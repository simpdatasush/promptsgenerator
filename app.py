import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# --- Limiter Configuration ---
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    strategy="fixed-window"
)

# --- Gemini API Key and Configuration ---
GEMINI_API_CONFIGURED = False
GEMINI_API_KEY = None

def configure_gemini_api():
    global GEMINI_API_KEY, GEMINI_API_CONFIGURED
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            GEMINI_API_CONFIGURED = True
            print("Gemini API configured successfully from environment variable.")
        except Exception as e:
            print(f"ERROR: Failed to configure Gemini API: {e}")
            print("Please ensure your API key environment variable (GEMINI_API_KEY) is correct and valid.")
            GEMINI_API_CONFIGURED = False
    else:
        print("\n" + "="*80)
        print("WARNING: GEMINI_API_KEY environment variable not set. Prompt generation features will be disabled.")
        print("Please set the GEMINI_API_KEY environment variable on Render.")
        print("="*80 + "\n")
        GEMINI_API_CONFIGURED = False

configure_gemini_api()

# --- NEW: Response Filtering Function ---
def filter_gemini_response(text):
    """
    Filters Gemini's response to prevent it from answering out-of-scope questions
    or discussing its own nature/errors.
    """
    unauthorized_message = "I am not authorised to answer this question. My purpose is solely to refine your raw prompt into a machine-readable format."
    text_lower = text.lower()

    # Phrases indicating self-reference, model capabilities, or app details
    unauthorized_phrases = [
        "as a large language model",
        "i am an ai",
        "i was trained by",
        "my training data",
        "this application was built using",
        "the code for this app",
        "i cannot fulfill this request because",
        "i apologize, but i cannot",
        "i'm sorry, but i cannot",
        "i am unable to",
        "i do not have access",
        "i am not able to",
        "i cannot process",
        "i cannot provide",
        "i am not programmed",
        "i cannot generate",
        "i cannot give you details about my internal workings",
        "i cannot discuss my creation or operation",
        "i cannot explain the development of this tool",
        "my purpose is to",
        "i am designed to",
        "i don't have enough information to", # If it refers to its own lack of info, not the user's prompt
        "i lack the ability to"
    ]

    # Phrases indicating it's discussing its own bugs/errors
    bug_phrases = [
        "a bug occurred",
        "i encountered an error",
        "there was an issue in my processing",
        "i made an error",
        "my apologies", # General apology, sometimes precedes self-correction or refusal
        "i cannot respond to that"
    ]

    # Check for specific refusal/self-referential phrases
    for phrase in unauthorized_phrases:
        if phrase in text_lower:
            # Special case: allow "i don't have enough information" if it's clearly about the user's input/prompt
            if phrase == "i don't have enough information to" and ("about the provided prompt" in text_lower or "based on your input" in text_lower or "to understand the context" in text_lower):
                continue
            return unauthorized_message

    # Check for bug-related phrases
    for phrase in bug_phrases:
        if phrase in text_lower:
            return unauthorized_message

    # If it's a general "no response" from our side, still show that error
    if "no response from model." in text_lower or "error communicating with gemini api:" in text_lower:
        return text # Don't filter our own internal error messages

    return text # If no unauthorized phrase is found, return the original text

# --- Gemini API interaction functions ---

async def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=1024):
    if not GEMINI_API_CONFIGURED:
        return "Gemini API Key is not configured. Please set your API key to use this feature."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

        generation_config = {"max_output_tokens": max_output_tokens}

        response = await model.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=generation_config
        )

        raw_gemini_text = response.text if response and response.text else "No response from model."
        return filter_gemini_response(raw_gemini_text).strip() # Apply filter here
    except Exception as e:
        print(f"DEBUG: Error calling Gemini API: {e}")
        return filter_gemini_response(f"Error communicating with Gemini API: {e}") # Apply filter here too

async def generate_prompts_async(raw_input):
    if not raw_input.strip():
        return {
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        }

    # 1. Polished Prompt - Await directly as it's a prerequisite for others
    # Add an instruction to the prompt itself to discourage off-topic replies
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

    # 2. Prompt Variants (Creative, Technical, Shorter) - Prepare coroutines
    # IMPORTANT: We need to pass the polished_prompt (which has already been filtered)
    # and re-add the strict instruction to each subsequent prompt as well,
    # as they also make separate calls to ask_gemini_for_prompt.
    strict_instruction_suffix = "\n\nDo NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided text."

    creative_coroutine = ask_gemini_for_prompt(f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_prompt}{strict_instruction_suffix}")
    technical_coroutine = ask_gemini_for_prompt(f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_prompt}{strict_instruction_suffix}")
    shorter_coroutine = ask_gemini_for_prompt(f"Condense the following prompt into its shortest possible form while retaining all essential meaning and instructions. Aim for brevity.:\n\n{polished_prompt}{strict_instruction_suffix}", max_output_tokens=512)

    # 3. Suggested Additions - Prepare coroutine
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

    # Await all *coroutines* concurrently using asyncio.gather
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
@limiter.limit("1 per 10 minutes")
def generate_prompts_endpoint():
    raw_input = request.form.get('prompt_input', '').strip()

    if not raw_input:
        return jsonify({
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        })

    results = asyncio.run(generate_prompts_async(raw_input))
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))
