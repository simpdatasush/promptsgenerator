import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash, session
import logging
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from asgiref.sync import async_to_sync

# --- NEW IMPORTS FOR GOOGLE AUTHENTICATION ---
from oauthlib.oauth2 import WebApplicationClient
import requests
import json # For parsing Google discovery document
# --- END NEW IMPORTS ---


app = Flask(__name__)

# --- Flask-SQLAlchemy Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_that_should_be_in_env')
db = SQLAlchemy(app)

# --- Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Configure logging for the Flask app
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# --- Temporary In-Memory Storage for Saved Prompts ---
saved_prompts_in_memory = []

# --- Language Mapping for Gemini Instructions ---
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
        app.logger.warning("WARNING: GEMINI_API_KEY environment variable not set. AI generation features will be disabled.")
        app.logger.warning("Please set the GEMINI_API_KEY environment variable on Render.")
        app.logger.warning("="*80 + "\n")
        GEMINI_API_CONFIGURED = False

configure_gemini_api()

# --- NEW: Google OAuth Configuration ---
app.config['GOOGLE_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")
app.config['GOOGLE_DISCOVERY_URL'] = "https://accounts.google.com/.well-known/openid-configuration"

# OAuth 2 client setup
google_client = WebApplicationClient(app.config['GOOGLE_CLIENT_ID'])
# --- END NEW: Google OAuth Configuration ---


# --- User Model for SQLAlchemy and Flask-Login ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True) # Make nullable for Google-only users
    google_id = db.Column(db.String(120), unique=True, nullable=True) # New field for Google ID
    email = db.Column(db.String(120), unique=True, nullable=True) # New field for email (from Google or manual)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        if self.google_id:
            return f'<User {self.username} (Google ID: {self.google_id})>'
        return f'<User {self.username}>'

# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


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
            return text # Keep the error message for debugging purposes
    
    if "no response from model." in text_lower or "error communicating with gemini api:" in text_lower:
        return text
    return text

# --- Gemini API interaction function (Creates model instance per call) ---
async def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=1024):
    if not GEMINI_API_CONFIGURED:
        return "Gemini API Key is not configured or the AI model failed to initialize."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash') 

        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.1
        }

        response = await model.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=generation_config
        )
        raw_gemini_text = response.text if response and response.text else "No response from model."
        return filter_gemini_response(raw_gemini_text).strip()
    except Exception as e:
        app.logger.error(f"DEBUG: Error calling Gemini API: {e}", exc_info=True)
        return filter_gemini_response(f"Error communicating with Gemini API: {e}")

# --- Prompt Generator Engine (No Changes) ---
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

# --- AI Cover Letter and CV Generator Engine ---

async def generate_cover_letter_async(job_description, language_code="en-US"):
    if not job_description.strip():
        return "Please provide a job description to generate a cover letter."
    
    target_language_name = LANGUAGE_MAP.get(language_code, "English")
    language_instruction_prefix = f"The output MUST be entirely in {target_language_name}. "

    cover_letter_instruction = language_instruction_prefix + f"""
    Generate a professional cover letter based on the following job description.
    Focus on highlighting relevant skills and experiences that match the job description.
    Structure:
    1. Your Contact Information (Name, Address, Phone, Email - use placeholders like [Your Name])
    2. Date
    3. Hiring Manager/Company Contact Information (use placeholders like [Hiring Manager Name], [Company Name], [Company Address] if not provided)
    4. Salutation (e.g., Dear [Hiring Manager Name] or Dear Hiring Team,)
    5. Opening Paragraph: State the position you're applying for and where you saw the advertisement. Briefly mention your enthusiasm.
    6. Body Paragraph(s): Connect your key skills and experiences to the requirements mentioned in the job description. Provide specific examples where possible. Highlight achievements.
    7. Closing Paragraph: Reiterate your interest, express eagerness for an interview, and thank them for your time and consideration.
    8. Professional Closing (e.g., Sincerely,)
    9. Your Typed Name (use placeholder [Your Name])

    Job Description:
    {job_description}

    Crucially, do NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to generate a professional cover letter.
    """
    return await ask_gemini_for_prompt(cover_letter_instruction, max_output_tokens=2048)

async def generate_cv_async(user_cv_text, job_description, language_code="en-US"):
    if not user_cv_text.strip() or not job_description.strip():
        return "Please provide both your CV text and the job description to generate a tailored CV."
    
    target_language_name = LANGUAGE_MAP.get(language_code, "English")
    language_instruction_prefix = f"The output MUST be entirely in {target_language_name}. "

    cv_instruction = language_instruction_prefix + f"""
    Given the candidate's existing CV text and a specific job description, generate a revised CV in plain text format.
    The goal is to highlight the most relevant skills, experiences, and achievements from the candidate's CV that directly align with the job description.
    Rephrase bullet points and descriptions to use keywords and phrasing from the job description where appropriate, without fabricating information.
    Maintain a clear, professional, and concise structure similar to a standard resume, using sections like:
    - SYNOPSIS/SUMMARY
    - TECHNICAL FORTE (if applicable)
    - SKILL SETS
    - WORK EXPERIENCE (most relevant first)
    - EDUCATION
    - CERTIFICATIONS

    Do NOT include personal contact details (like phone, email, address, LinkedIn URL, sex, date of birth, nationality) in the generated CV. Use placeholders like [Candidate Name] for the name.
    Ensure the output is a single, coherent plain text document, ready for download.

    Candidate's CV Text:
    {user_cv_text}

    Job Description:
    {job_description}

    Crucially, do NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to generate a tailored CV.
    """
    return await ask_gemini_for_prompt(cv_instruction, max_output_tokens=3072)


# --- Main Flask Routes ---

@app.route('/')
def root_redirect():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', current_user=current_user)

@app.route('/prompt_generator_app')
@login_required
def prompt_generator_app():
    return render_template('index.html', current_user=current_user)

@app.route('/cv_cover_letter_app')
@login_required
def cv_cover_letter_app():
    return render_template('cv_cover_letter_app.html', current_user=current_user, language_map=LANGUAGE_MAP)


@app.route('/generate', methods=['POST'])
@login_required
def generate_prompts_endpoint():
    raw_input = request.form.get('prompt_input', '').strip()
    language_code = request.form.get('language_code', 'en-US')

    if not raw_input:
        return jsonify({
            "polished": "Please enter some text to generate prompts.",
            "creative": "", "technical": "", "shorter": "", "additions": ""
        })

    try:
        results = async_to_sync(generate_prompts_async)(raw_input, language_code)
        return jsonify(results)
    except Exception as e:
        app.logger.exception("Error during prompt generation in endpoint:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500

# --- CV/Cover Letter Generation Endpoints ---
@app.route('/generate_cover_letter', methods=['POST'])
@login_required
def generate_cover_letter_endpoint():
    job_description = request.form.get('job_description', '').strip()
    language_code = request.form.get('language_code', 'en-US')

    if not job_description:
        return jsonify({"error": "Please provide a job description."}), 400

    try:
        cover_letter = async_to_sync(generate_cover_letter_async)(job_description, language_code)
        if "Error" in cover_letter or "not configured" in cover_letter:
            return jsonify({"error": cover_letter}), 500
        return jsonify({"cover_letter": cover_letter})
    except Exception as e:
        app.logger.exception("Error during cover letter generation:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500


@app.route('/generate_cv', methods=['POST'])
@login_required
def generate_cv_endpoint():
    user_cv_text = request.form.get('user_cv_text', '').strip()
    job_description = request.form.get('job_description', '').strip()
    language_code = request.form.get('language_code', 'en-US')

    if not user_cv_text or not job_description:
        return jsonify({"error": "Please provide both your CV text and the job description."}), 400

    try:
        tailored_cv = async_to_sync(generate_cv_async)(user_cv_text, job_description, language_code)
        if "Error" in tailored_cv or "not configured" in tailored_cv:
            return jsonify({"error": tailored_cv}), 500
        return jsonify({"tailored_cv": tailored_cv})
    except Exception as e:
        app.logger.exception("Error during CV generation:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500


# --- Save Prompt Endpoint ---
@app.route('/save_prompt', methods=['POST'])
@login_required
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
        "text": prompt_text,
        "user_id": current_user.id # Store user ID for association
    })

    app.logger.info(f"Prompt of type '{prompt_type}' saved to memory at {timestamp} by user ID {current_user.id}.")
    return jsonify({"success": True, "message": "Prompt saved temporarily!"}), 200

# --- Get Saved Prompts Endpoint ---
@app.route('/get_saved_prompts', methods=['GET'])
@login_required
def get_saved_prompts_endpoint():
    # Only return prompts saved by the current user
    user_prompts = [p for p in saved_prompts_in_memory if p.get('user_id') == current_user.id]
    return jsonify(user_prompts), 200


# --- Download Prompts as TXT Endpoint ---
@app.route('/download_prompts_txt', methods=['GET'])
@login_required
def download_prompts_txt():
    # Filter prompts by current user for download
    prompts_to_download = [p for p in saved_prompts_in_memory if p.get('user_id') == current_user.id]

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
    filename = f"saved_prompts_{current_user.username}.txt"

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
        return redirect(url_for('dashboard')) # Changed redirect to dashboard

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
        return redirect(url_for('dashboard')) # Changed redirect to dashboard

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember_me = 'remember_me' in request.form

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember_me)
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard')) # Changed redirect to dashboard
        else:
            flash('Login Unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required # Only logged-in users can log out
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login')) # Changed redirect to login page after logout

# --- NEW: Google Login Routes ---
@app.route('/google_login')
def google_login():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('dashboard'))

    # Get Google's OpenID Connect discovery document
    try:
        resp = requests.get(app.config['GOOGLE_DISCOVERY_URL'])
        resp.raise_for_status()
        google_config = resp.json()
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching Google discovery document: {e}")
        flash('Failed to connect to Google for login. Please try again later.', 'danger')
        return redirect(url_for('login'))

    # Prepare the OAuth request
    request_uri = google_client.prepare_request_uri(
        google_config['authorization_endpoint'],
        redirect_uri=request.base_url + '/callback',
        scope=["openid", "email", "profile"],
        prompt="select_account" # Force account selection
    )
    return redirect(request_uri)

@app.route('/google_login/callback')
def google_callback():
    # Get the authorization code from the request
    code = request.args.get('code')
    if not code:
        flash('Google login failed: No authorization code received.', 'danger')
        return redirect(url_for('login'))

    # Get Google's OpenID Connect discovery document
    try:
        resp = requests.get(app.config['GOOGLE_DISCOVERY_URL'])
        resp.raise_for_status()
        google_config = resp.json()
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching Google discovery document in callback: {e}")
        flash('Failed to connect to Google for login. Please try again later.', 'danger')
        return redirect(url_for('login'))

    try:
        # Prepare and send token request
        token_url, headers, body = google_client.prepare_token_request(
            google_config['token_endpoint'],
            authorization_response=request.url,
            redirect_url=request.base_url,
            code=code,
            client_id=app.config['GOOGLE_CLIENT_ID'],
            client_secret=app.config['GOOGLE_CLIENT_SECRET']
        )
        token_response = requests.post(
            token_url,
            headers=headers,
            data=body,
            auth=(app.config['GOOGLE_CLIENT_ID'], app.config['GOOGLE_CLIENT_SECRET']),
        )
        token_response.raise_for_status()
        google_client.parse_request_body_response(token_response.text)

        # Get user info from ID token
        userinfo_endpoint = google_config['userinfo_endpoint']
        uri, headers, body = google_client.add_token(userinfo_endpoint)
        userinfo_response = requests.get(uri, headers=headers, data=body)
        userinfo_response.raise_for_status()
        user_info = userinfo_response.json()

        google_id = user_info['sub']
        email = user_info.get('email')
        username_from_google = user_info.get('name', user_info.get('email', f"google_user_{google_id[:8]}")) # Fallback username

        # Find or create user
        user = User.query.filter_by(google_id=google_id).first()
        if not user:
            # Check if an existing user with the same email already exists (for linking accounts)
            if email:
                existing_user_with_email = User.query.filter_by(email=email).first()
                if existing_user_with_email:
                    # Link existing account to Google
                    existing_user_with_email.google_id = google_id
                    db.session.commit()
                    user = existing_user_with_email
                    flash(f'Your existing account ({user.username}) has been linked with Google!', 'success')
                else:
                    # Create new user
                    user = User(username=username_from_google, email=email, google_id=google_id, password_hash=None) # No password for Google-only
                    db.session.add(user)
                    db.session.commit()
                    flash('Registered and logged in with Google successfully!', 'success')
            else:
                # Fallback if no email provided by Google (less common, but possible)
                flash('Google login failed: Could not retrieve email. Please ensure your Google account has an email associated.', 'danger')
                return redirect(url_for('login'))
        else:
            flash('Logged in with Google successfully!', 'success')

        login_user(user)
        return redirect(url_for('dashboard'))

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error during Google OAuth token/userinfo exchange: {e}")
        flash('Google login failed due to a network error. Please try again.', 'danger')
        return redirect(url_for('login'))
    except Exception as e:
        app.logger.exception("An unexpected error occurred during Google callback:")
        flash('An unexpected error occurred during Google login. Please try again.', 'danger')
        return redirect(url_for('login'))
# --- END NEW: Google Login Routes ---


# --- Database Initialization (Run once to create tables) ---
with app.app_context():
    db.create_all()
    app.logger.info("Database tables created/checked.")

# --- Main App Run ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))
