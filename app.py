import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash
import logging
from datetime import datetime

# --- NEW IMPORTS FOR AUTHENTICATION ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# --- END NEW IMPORTS ---


app = Flask(__name__)

# --- NEW: Flask-SQLAlchemy Configuration ---
# Configure SQLite database. This file will be created in your project directory.
# On Render, this database file will be ephemeral unless you attach a persistent disk.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress a warning
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_that_should_be_in_env') # Needed for Flask-Login sessions
db = SQLAlchemy(app)
# --- END NEW: Flask-SQLAlchemy Configuration ---

# --- NEW: Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # The view Flask-Login should redirect to for login
# --- END NEW: Flask-Login Configuration ---


# Configure logging for the Flask app
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# --- Temporary In-Memory Storage for Saved Prompts (Unchanged) ---
saved_prompts_in_memory = []

# --- Language Mapping for Gemini Instructions (Unchanged) ---
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


# --- Gemini API Key and Configuration (Unchanged) ---
GEMINI_API_CONFIGURED = False
GEMINI_API_KEY = None

def configure_gemini_api():
    global GEMINI_API_KEY, GEMINI_API_CONFIGURED
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            GEMINI_API_CONFIGURED = True
            app.logger.info("Gemini API configured successfully.")
        except Exception as e:
            app.logger.error(f"ERROR: Failed to configure Gemini API: {e}")
            app.logger.error("Please ensure your API key environment variable (GEMINI_API_KEY) is correct and valid.")
            GEMINI_API_CONFIGURED = False
    else:
        app.logger.warning("\n" + "="*80)
        app.logger.warning("WARNING: GEMINI_API_KEY environment variable not set. Prompt generation features will be disabled.")
        app.logger.warning("Please set the GEMINI_API_KEY environment variable on Render.")
        app.logger.warning("="*80 + "\n")
        GEMINI_API_CONFIGURED = False

configure_gemini_api()

# --- NEW: User Model for SQLAlchemy and Flask-Login ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

# --- NEW: Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id)) # Use db.session.get for primary key lookup
# --- END NEW: User Model and Loader ---


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
    for phrase in unauthorized_phrases:
        if phrase in text_lower:
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

# --- Gemini API interaction function ---
async def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=1024):
    if not GEMINI_API_CONFIGURED:
        return "Gemini API Key is not configured or the AI model failed to initialize."

    try:
        gemini_model_instance = genai.GenerativeModel('gemini-2.0-flash') 

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

# --- generate_prompts_async function (main async logic for prompt variations) ---
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
        "polished": polished_prompt,
        "creative": creative_result,
        "technical": technical_result,
        "shorter": shorter_result,
        "additions": additions_result
    }

# --- Flask Routes ---
@app.route('/')
def index():
    # Pass current_user object to the template to show login/logout status
    return render_template('index.html', current_user=current_user)

@app.route('/generate', methods=['POST'])
@login_required # Protect this route
def generate_prompts_endpoint():
    raw_input = request.form.get('prompt_input', '').strip()
    language_code = request.form.get('language_code', 'en-US')

    if not raw_input:
        return jsonify({
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        })

    try:
        results = asyncio.run(generate_prompts_async(raw_input, language_code))
        return jsonify(results)
    except Exception as e:
        app.logger.exception("Error during prompt generation in endpoint:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500

# --- Save Prompt Endpoint ---
@app.route('/save_prompt', methods=['POST'])
@login_required # Protect this route
def save_prompt_endpoint():
    # For now, saved prompts are still in-memory and not tied to users.
    # If persistent per-user storage is needed, this would require a DB change.
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
        "text": prompt_text,
        "user": current_user.username if current_user.is_authenticated else "anonymous" # Track user
    })

    app.logger.info(f"Prompt of type '{prompt_type}' saved to memory at {timestamp} by {current_user.username if current_user.is_authenticated else 'anonymous'}.")
    return jsonify({"success": True, "message": "Prompt saved temporarily!"}), 200

# --- Get Saved Prompts Endpoint ---
@app.route('/get_saved_prompts', methods=['GET'])
@login_required # Protect this route
def get_saved_prompts_endpoint():
    # Only return prompts saved by the current user if authenticated
    if current_user.is_authenticated:
        user_prompts = [p for p in saved_prompts_in_memory if p.get('user') == current_user.username]
        return jsonify(user_prompts), 200
    else:
        # If not authenticated, return only anonymous prompts or deny access
        # For this basic setup, let's just return anonymous ones if not logged in
        anonymous_prompts = [p for p in saved_prompts_in_memory if p.get('user') == "anonymous"]
        return jsonify(anonymous_prompts), 200


# --- Download Prompts as TXT Endpoint ---
@app.route('/download_prompts_txt', methods=['GET'])
@login_required # Protect this route
def download_prompts_txt():
    # Filter prompts by current user for download
    if current_user.is_authenticated:
        prompts_to_download = [p for p in saved_prompts_in_memory if p.get('user') == current_user.username]
    else:
        prompts_to_download = [p for p in saved_prompts_in_memory if p.get('user') == "anonymous"]

    if not prompts_to_download:
        return "No prompts to download for this user.", 404

    lines = []
    for i, prompt in enumerate(prompts_to_download):
        lines.append(f"--- PROMPT {i+1} ---")
        lines.append(f"Type: {prompt['type'].capitalize()}")
        lines.append(f"Saved: {prompt['timestamp']}")
        lines.append("-" * 30)
        lines.append(prompt['text'])
        lines.append("-" * 30)
        lines.append("\n")

    text_content = "\n".join(lines).strip()
    filename = f"saved_prompts_{current_user.username if current_user.is_authenticated else 'anonymous'}.txt"

    response = make_response(text_content)
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-type"] = "text/plain"
    app.logger.info(f"Generated and sending {filename} for download.")
    return response


# --- NEW: Authentication Routes ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists. Please choose a different one.', 'danger')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember_me = 'remember_me' in request.form

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember_me)
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next') # Redirect to the page user tried to access
            return redirect(next_page or url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required # Only logged-in users can log out
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))
# --- END NEW: Authentication Routes ---


# --- Database Initialization (Run once to create tables) ---
# This block ensures tables are created when the app starts.
# In production, you might use Flask-Migrate or a separate script.
with app.app_context():
    db.create_all()
    app.logger.info("Database tables created/checked.")

# --- Main App Run ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))

