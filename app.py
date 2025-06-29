import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, make_response
import logging
from datetime import datetime

app = Flask(__app__)

# Configure logging for the Flask app
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# --- NEW: Temporary In-Memory Storage for Saved Prompts ---
saved_prompts_in_memory = []

# --- NEW: Language Mapping for Gemini Instructions ---
LANGUAGE_MAP = {
    "en-US": "English",
    "en-GB": "English (UK)",
    "es-ES": "Spanish",
    "fr-FR": "French",
    "de-DE": "German",
    "it-IT": "Italian",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "zh-CN": "Simplified Chinese",
    "hi-IN": "Hindi"
}


# --- Gemini API Key and Configuration ---
GEMINI_API_CONFIGURED = False
GEMINI_API_KEY = None
gemini_model_instance = None

def configure_gemini_api():
    global GEMINI_API_KEY, GEMINI_API_CONFIGURED, gemini_model_instance
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Use gemini-1.5-flash for potentially better handling of longer contexts in recursion
            gemini_model_instance = genai.GenerativeModel('gemini-1.5-flash') 
            GEMINI_API_CONFIGURED = True
            app.logger.info("Gemini API configured successfully and model instance initialized.")
        except Exception as e:
            app.logger.error(f"ERROR: Failed to configure Gemini API: {e}")
            app.logger.error("Please ensure your API key environment variable (GEMINI_API_KEY) is correct and valid.")
            GEMINI_API_CONFIGURED = False
            gemini_model_instance = None
    else:
        app.logger.warning("\n" + "="*80)
        app.logger.warning("WARNING: GEMINI_API_KEY environment variable not set. Prompt generation features will be disabled.")
        app.logger.warning("Please set the GEMINI_API_KEY environment variable on Render.")
        app.logger.warning("="*80 + "\n")
        GEMINI_API_CONFIGURED = False
        gemini_model_instance = None

configure_gemini_api()

# --- Response Filtering Function ---
def filter_gemini_response(text):
    unauthorized_message = "I am not authorised to answer this question. My purpose is solely to refine your raw prompt into a machine-readable format."
    text_lower = text.lower()
    unauthorized_phrases = [
        "as a large language model", "i am an ai", "i was trained by", "my training data",
        "this application was built using", "the code for this app", "i cannot fulfill this request because",
        "i apologize, but i cannot", "i'm sorry, but i cannot", "i am unable to", "i do not have access",
        "i am not able to", "i cannot process", "i cannot provide", "i am not programmed",
        "i cannot generate", "i cannot give you details about my internal workings",
        "i cannot discuss my creation or operation", "i cannot explain the development of this tool",
        "my purpose is to", "i am designed to", "i don't have enough information to", "i lack the ability to"
    ]
    for phrase in unauthorized_phrases:
        if phrase in text_lower:
            # Allow "i don't have enough information to" if it's clearly related to the prompt itself
            if phrase == "i don't have enough information to" and \
               ("about the provided prompt" in text_lower or "based on your input" in text_lower or "to understand the context" in text_lower):
                continue
            return unauthorized_message
    
    bug_phrases = [
        "a bug occurred", "i encountered an error", "there was an issue in my processing",
        "i made an error", "my apologies", "i cannot respond to that"
    ]
    for phrase in bug_phrases:
        if phrase in text_lower:
            return unauthorized_message
    
    if "no response from model." in text_lower or "error communicating with gemini api:" in text_lower:
        return text
    return text

# --- Gemini API interaction functions ---
async def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=1024):
    if not GEMINI_API_CONFIGURED or gemini_model_instance is None:
        return "Gemini API Key is not configured or the AI model failed to initialize."

    try:
        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.1
        }

        response = await gemini_model_instance.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=generation_config
        )
        raw_gemini_text = response.text if response and response.text else "No response from model."
        return filter_gemini_response(raw_gemini_text).strip()
    except Exception as e:
        app.logger.error(f"DEBUG: Error calling Gemini API: {e}", exc_info=True)
        return filter_gemini_response(f"Error communicating with Gemini API: {e}")

# --- generate_prompts_async (original function, unchanged) ---
async def generate_prompts_async(raw_input, language_code="en-US"):
    if not raw_input.strip():
        return {
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        }

    target_language_name = LANGUAGE_MAP.get(language_code, "English")
    language_instruction_prefix = f"The output MUST be entirely in {target_language_name}. "

    polished_prompt_instruction = language_instruction_prefix + f"""Refine the following text into a clear, concise, and effective prompt for a large language model. Improve grammar, clarity, and structure. Do not add external information, only refine the given text.

Crucially, do NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided raw text into a better prompt.

Raw Text:
{raw_input}"""
    polished_prompt = await ask_gemini_for_prompt(polished_prompt_instruction)

    if "Error" in polished_prompt or "not configured" in polished_prompt:
        return {
            "polished": polished_prompt,
            "creative": "", "technical": "", "shorter": "", "additions": ""
        }

    strict_instruction_suffix = "\n\nDo NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided text."

    creative_coroutine = ask_gemini_for_prompt(language_instruction_prefix + f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_prompt}{strict_instruction_suffix}")
    technical_coroutine = ask_gemini_for_prompt(language_instruction_prefix + f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_prompt}{strict_instruction_suffix}")
    shorter_coroutine = ask_gemini_for_prompt(language_instruction_prefix + f"Condense the following prompt into its shortest possible form while retaining all essential meaning and instructions. Aim for brevity.:\n\n{polished_prompt}{strict_instruction_suffix}", max_output_tokens=512)

    additions_coroutine = ask_gemini_for_prompt(language_instruction_prefix + f"""Analyze the following prompt and suggest potential additions to improve its effectiveness for a large language model. Focus on elements like:
    -   Desired Tone (e.g., formal, informal, humorous, serious)
    -   Required Format (e.g., bullet points, essay, script, email, JSON)
    -   Target Audience (e.g., experts, general public, children)
    -   Specific Length (e.g., 500 words, 3 paragraphs, 2 sentences)
    -   Examples or Context (if applicable)
    -   Constraints (e.g., "Do not use X", "Avoid Y")
    -   Perspective (e.g., "Act as a marketing expert")

    Provide your suggestions concisely, perhaps as a list or brief paragraphs.
    {strict_instruction_suffix}

    Prompt: {polished_prompt}
    """)

    creative_result, technical_result, shorter_result, additions_result = await asyncio.gather(
        creative_coroutine, technical_coroutine, shorter_coroutine, additions_coroutine
    )

    return {
        "polished": polished_result, # Use polished_result from here
        "creative": creative_result,
        "technical": technical_result,
        "shorter": shorter_result,
        "additions": additions_result
    }

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_prompts_endpoint():
    raw_input = request.form.get('prompt_input', '').strip()
    language_code = request.form.get('language_code', 'en-US')

    if not raw_input:
        return jsonify({
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        })

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    results = loop.run_until_complete(generate_prompts_async(raw_input, language_code))
    return jsonify(results)

# --- Save Prompt Endpoint ---
@app.route('/save_prompt', methods=['POST'])
def save_prompt_endpoint():
    prompt_data = request.get_json()
    prompt_text = prompt_data.get('prompt_text')
    prompt_type = prompt_data.get('prompt_type', 'unknown')

    if not prompt_text:
        app.logger.warning("Attempted to save empty prompt text.")
        return jsonify({"success": False, "message": "No prompt text provided"}), 400

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved_prompts_in_memory.append({
        "timestamp": timestamp,
        "type": prompt_type,
        "text": prompt_text
    })

    app.logger.info(f"Prompt of type '{prompt_type}' saved to memory at {timestamp}.")
    return jsonify({"success": True, "message": "Prompt saved temporarily!"}), 200

# --- Get Saved Prompts Endpoint ---
@app.route('/get_saved_prompts', methods=['GET'])
def get_saved_prompts_endpoint():
    return jsonify(list(saved_prompts_in_memory)), 200

# --- Download Prompts as TXT Endpoint ---
@app.route('/download_prompts_txt', methods=['GET'])
def download_prompts_txt():
    if not saved_prompts_in_memory:
        return "No prompts to download.", 404

    lines = []
    for i, prompt in enumerate(saved_prompts_in_memory):
        lines.append(f"--- PROMPT {i+1} ---")
        lines.append(f"Type: {prompt['type'].capitalize()}")
        lines.append(f"Saved: {prompt['timestamp']}")
        lines.append("-" * 30)
        lines.append(prompt['text'])
        lines.append("-" * 30)
        lines.append("\n")

    text_content = "\n".join(lines).strip()

    response = make_response(text_content)
    response.headers["Content-Disposition"] = "attachment; filename=saved_prompts.txt"
    response.headers["Content-type"] = "text/plain"
    app.logger.info("Generated and sending saved_prompts.txt for download.")
    return response

# --- NEW: Recursive Polish Endpoint ---
@app.route('/recursive_polish', methods=['POST'])
async def recursive_polish_endpoint():
    raw_input = request.form.get('prompt_input', '').strip()
    language_code = request.form.get('language_code', 'en-US')
    num_iterations = 3 # Fixed number of iterations

    if not raw_input:
        return jsonify({"error": "Please provide input for recursive polishing."}), 400

    current_prompt_text = raw_input
    target_language_name = LANGUAGE_MAP.get(language_code, "English")
    final_polished_prompt = ""

    for i in range(num_iterations):
        app.logger.info(f"Recursive Polish - Iteration {i+1}/{num_iterations}")
        
        refinement_instruction = (
            f"You are tasked with iteratively refining a given prompt for a large language model. "
            f"On this step (Iteration {i+1} of {num_iterations}), focus on making it even more clear, "
            f"concise, and effective. Remove any redundancy, improve phrasing, and ensure it directly guides the LLM. "
            f"Do NOT add new information or answer questions about yourself or how this application works. "
            f"Your sole purpose is to refine the prompt. "
            f"The output MUST be entirely in {target_language_name}.\n\n"
            f"Refine this prompt:\n{current_prompt_text}"
        )
        
        try:
            refined_output = await ask_gemini_for_prompt(refinement_instruction)
            
            if "Error" in refined_output or "not configured" in refined_output or "not authorised" in refined_output:
                app.logger.error(f"Recursive polishing failed at iteration {i+1}: {refined_output}")
                return jsonify({"error": f"Recursive polishing stopped due to AI error at iteration {i+1}: {refined_output}"}), 500
            
            current_prompt_text = refined_output # The output of this step becomes the input for the next
            final_polished_prompt = refined_output # Keep track of the last successful refinement

        except Exception as e:
            app.logger.exception(f"Error during recursive polishing iteration {i+1}:")
            return jsonify({"error": f"An unexpected server error occurred during recursive polishing at iteration {i+1}: {e}"}), 500
            
    return jsonify({"final_polished_prompt": final_polished_prompt})


if __name__ == '__main__':
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))
