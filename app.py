import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, g # Import g for app-level state

app = Flask(__name__)

# --- Gemini API Key and Configuration ---
GEMINI_API_CONFIGURED = False
GEMINI_API_KEY = None # Will be set from environment variable

def configure_gemini_api():
    global GEMINI_API_KEY, GEMINI_API_CONFIGURED
    # Prioritize reading from environment variable
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

# Call this once at app startup
configure_gemini_api()


# --- Asyncio Loop Management (for Flask context) ---
# Use a simple wrapper to ensure async functions run within the Flask context
# For production, a dedicated async worker might be more robust,
# but for this example, we'll run it in the main thread's loop.
async def run_async_function(coroutine_func):
    """Helper to run an async coroutine."""
    try:
        result = await coroutine_func
        return result
    except Exception as e:
        print(f"ERROR: An error occurred during async operation: {e}")
        return {"error": f"An error occurred during prompt generation: {e}"}

# --- Your existing Gemini API interaction functions (adapted for web) ---

async def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=4096):
    """
    Makes an asynchronous API call to Gemini for prompt generation.
    """
    if not GEMINI_API_CONFIGURED:
        return "Gemini API Key is not configured. Please set your API key to use this feature."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash') # Using a capable model

        generation_config = {"max_output_tokens": max_output_tokens}

        response = await model.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=generation_config
        )

        if response and response.text:
            return response.text.strip()
        else:
            print(f"DEBUG: Gemini returned empty response for prompt: {prompt_instruction[:50]}...")
            return "No response from model."
    except Exception as e:
        print(f"DEBUG: Error calling Gemini API: {e}")
        return f"Error communicating with Gemini API: {e}"

async def generate_prompts_async(raw_input):
    """
    Asynchronously generates a polished prompt, variants, and suggestions.
    """
    if not raw_input.strip():
        return {
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        }

    # 1. Polished Prompt
    polished_prompt_instruction = f"Refine the following text into a clear, concise, and effective prompt for a large language model. Improve grammar, clarity, and structure. Do not add external information, only refine the given text:\n\n{raw_input}"
    polished_prompt_task = asyncio.create_task(ask_gemini_for_prompt(polished_prompt_instruction))

    polished_prompt = await polished_prompt_task
    if "Error" in polished_prompt or "not configured" in polished_prompt:
        return {
            "polished": polished_prompt,
            "creative": "", "technical": "", "shorter": "", "additions": ""
        }

    # 2. Prompt Variants (Creative, Technical, Shorter)
    creative_prompt_instruction = f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_prompt}"
    technical_prompt_instruction = f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_prompt}"
    shorter_prompt_instruction = f"Condense the following prompt into its shortest possible form while retaining all essential meaning and instructions. Aim for brevity.:\n\n{polished_prompt}"

    creative_task = asyncio.create_task(ask_gemini_for_prompt(creative_prompt_instruction))
    technical_task = asyncio.create_task(ask_gemini_for_prompt(technical_prompt_instruction))
    shorter_task = asyncio.create_task(ask_gemini_for_prompt(shorter_prompt_instruction, max_output_tokens=512))

    # 3. Suggested Additions
    additions_instruction = f"""Analyze the following prompt and suggest potential additions to improve its effectiveness for a large language model. Focus on elements like:
    -   Desired Tone (e.g., formal, informal, humorous, serious)
    -   Required Format (e.g., bullet points, essay, script, email, JSON)
    -   Target Audience (e.g., experts, general public, children)
    -   Specific Length (e.g., 500 words, 3 paragraphs, 2 sentences)
    -   Examples or Context (if applicable)
    -   Constraints (e.g., "Do not use X", "Avoid Y")
    -   Perspective (e.g., "Act as a marketing expert")

    Provide your suggestions concisely, perhaps as a list or brief paragraphs.

    Prompt: {polished_prompt}
    """
    additions_task = asyncio.create_task(ask_gemini_for_prompt(additions_instruction))

    # Await all tasks concurrently
    creative_result, technical_result, shorter_result, additions_result = await asyncio.gather(
        creative_task, technical_task, shorter_task, additions_task
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
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
async def generate_prompts_endpoint():
    """Handles the prompt generation request from the web form."""
    raw_input = request.form.get('prompt_input', '').strip()

    if not raw_input:
        return jsonify({
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        })

    # Call your async function
    results = await generate_prompts_async(raw_input)
    return jsonify(results)

if __name__ == '__main__':
    # For local development, you can run: python app.py
    # Set your API key as an environment variable before running locally, e.g.:
    # export GEMINI_API_KEY="YOUR_ACTUAL_API_KEY"
    # python app.py
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))
