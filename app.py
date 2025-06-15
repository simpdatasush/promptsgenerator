import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime # Import datetime for unique filenames/timestamps

app = Flask(__name__)

# Configure logging for the Flask app
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# --- NEW: Temporary In-Memory Storage for Saved Prompts ---
# This list will hold saved prompts. It will reset when the server restarts.
saved_prompts_in_memory = []

# --- Gemini API Key and Configuration ---
GEMINI_API_CONFIGURED = False
GEMINI_API_KEY = None

# Global instance for the Gemini model
gemini_model_instance = None

def configure_gemini_api():
    global GEMINI_API_KEY, GEMINI_API_CONFIGURED, gemini_model_instance
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model_instance = genai.GenerativeModel('gemini-2.0-flash')
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

# --- Response Filtering Function (Unchanged) ---
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
    bug_phrases = [
        "a bug occurred", "i encountered an error", "there was an issue in my processing",
        "i made an error", "my apologies", "i cannot respond to that"
    ]
    for phrase in unauthorized_phrases:
        if phrase in text_lower:
            if phrase == "i don't have enough information to" and ("about the provided prompt" in text_lower or "based on your input" in text_lower or "to understand the context" in text_lower):
                continue
            return unauthorized_message
    for phrase in bug_phrases:
        if phrase in text_lower:
            return unauthorized_message
    if "no response from model." in text_lower or "error communicating with gemini api:" in text_lower:
        return text
    return text

# --- Gemini API interaction functions (Temperature added) ---
async def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=1024):
    if not GEMINI_API_CONFIGURED or gemini_model_instance is None:
        return "Gemini API Key is not configured or the AI model failed to initialize."

    try:
        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.1 # Ensure this is inside the dict
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

# --- generate_prompts_async (Unchanged from previous versions for core logic) ---
async def generate_prompts_async(raw_input):
    if not raw_input.strip():
        return {
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        }

    polished_prompt_instruction = f"""Refine the following text into a clear, concise, and effective prompt for a large language model. Improve grammar, clarity, and structure. Do not add external information, only refine the given text.

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

    creative_coroutine = ask_gemini_for_prompt(f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_prompt}{strict_instruction_suffix}")
    technical_coroutine = ask_gemini_for_prompt(f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_prompt}{strict_instruction_suffix}")
    shorter_coroutine = ask_gemini_for_prompt(f"Condense the following prompt into its shortest possible form while retaining all essential meaning and instructions. Aim for brevity.:\n\n{polished_prompt}{strict_instruction_suffix}", max_output_tokens=512)

    additions_coroutine = ask_gemini_for_prompt(f"""Analyze the following prompt and suggest potential additions to improve its effectiveness for a large language model. Focus on elements like:
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
        "polished": polished_prompt,
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

    if not raw_input:
        return jsonify({
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        })

    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(generate_prompts_async(raw_input))
        return jsonify(results)
    except Exception as e:
        app.logger.exception("Error during prompt generation in endpoint:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500

# --- NEW: Save Prompt Endpoint ---
@app.route('/save_prompt', methods=['POST'])
def save_prompt_endpoint():
    prompt_data = request.get_json() # Use get_json() for JSON payload
    prompt_text = prompt_data.get('prompt_text')
    prompt_type = prompt_data.get('prompt_type', 'unknown') # e.g., 'polished', 'creative'

    if not prompt_text:
        app.logger.warning("Attempted to save empty prompt text.")
        return jsonify({"success": False, "message": "No prompt text provided"}), 400

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Store a dictionary with type and timestamp for better context
    saved_prompts_in_memory.append({
        "timestamp": timestamp,
        "type": prompt_type,
        "text": prompt_text
    })

    app.logger.info(f"Prompt of type '{prompt_type}' saved to memory at {timestamp}.")
    return jsonify({"success": True, "message": "Prompt saved temporarily!"}), 200

# --- NEW: Get Saved Prompts Endpoint ---
@app.route('/get_saved_prompts', methods=['GET'])
def get_saved_prompts_endpoint():
    # Return the list of saved prompts. It's a copy to prevent external modification.
    return jsonify(list(saved_prompts_in_memory)), 200


if __name__ == '__main__':
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))
