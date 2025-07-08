import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash, session
import logging
from datetime import datetime
import requests # NEW: For making HTTP requests to Imagen API

# --- NEW IMPORTS FOR AUTHENTICATION ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# --- NEW: Authlib for Google OAuth ---
from authlib.integrations.flask_client import OAuth
import json # Ensure json is imported
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

# --- NEW: Google OAuth Configuration ---
oauth = OAuth(app)

# Load Google Client ID and Client Secret from environment variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Register Google OAuth client
# server_metadata_url automatically discovers endpoints from Google's OpenID Connect discovery document
# client_kwargs specifies the requested scopes: openid (for ID token), email, and profile information
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)
# --- END NEW: Google OAuth Configuration ---


# Configure logging for the Flask app
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# --- Temporary In-Memory Storage for Saved Prompts (Unchanged) ---
# Note: For production, this should be replaced with a persistent database
# and linked to the User model for proper per-user storage.
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

# --- NEW: User Model for SQLAlchemy and Flask-Login (MODIFIED for Google) ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    # Username is nullable as Google users might not have a traditional username
    username = db.Column(db.String(80), unique=True, nullable=True) 
    # Email is crucial for both local and Google users, must be unique
    email = db.Column(db.String(120), unique=True, nullable=False) 
    # password_hash is nullable for users who only log in via Google
    password_hash = db.Column(db.String(128), nullable=True) 
    # google_id stores the unique identifier from Google OAuth
    google_id = db.Column(db.String(128), unique=True, nullable=True) 

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        # Representation updated to show username or email for Google users
        return f'<User {self.username or self.email}>'

# --- NEW: Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    # Use db.session.get for primary key lookup, which is more efficient
    return db.session.get(User, int(user_id)) 
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
        "my purpose is to", "i am designed to", "i don't have enough information to" # Keep this last, as it's more general
    ]
    for phrase in unauthorized_phrases:
        if phrase in text_lower:
            # Special handling for "i don't have enough information to" to allow legitimate responses
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

# --- Gemini API interaction function (MODIFIED to be synchronous) ---
def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=1024):
    if not GEMINI_API_CONFIGURED:
        return "Gemini API Key is not configured or the AI model failed to initialize."

    try:
        gemini_model_instance = genai.GenerativeModel('gemini-2.0-flash') 

        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.1
        }
        
        # Use asyncio.run to execute the async method in a new, isolated event loop
        # This is crucial for running async code from a synchronous Flask context
        response = asyncio.run(gemini_model_instance.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=generation_config
        ))
        raw_gemini_text = response.text if response and response.text else "No response from model."
        return filter_gemini_response(raw_gemini_text).strip()
    except Exception as e:
        app.logger.error(f"DEBUG: Error calling Gemini API: {e}", exc_info=True)
        return filter_gemini_response(f"Error communicating with Gemini API: {e}")

# --- NEW: Imagen API interaction function (MODIFIED to be synchronous) ---
def generate_image_with_imagen(prompt):
    # The API key is automatically provided by Canvas if empty string
    apiKey = "" 
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={apiKey}"
    
    payload = { "instances": { "prompt": prompt }, "parameters": { "sampleCount": 1} }

    try:
        # Use asyncio.run to execute the async request in a new, isolated event loop
        # asyncio.to_thread is used to run the synchronous requests.post call within the async context
        result = asyncio.run(asyncio.to_thread(
            requests.post,
            apiUrl,
            headers={ 'Content-Type': 'application/json' },
            json=payload
        ).json()) # .json() needs to be called on the response object from requests.post
        
        if result.get('predictions') and len(result['predictions']) > 0 and result['predictions'][0].get('bytesBase64Encoded'):
            image_base64 = result['predictions'][0]['bytesBase64Encoded']
            return f"data:image/png;base64,{image_base64}"
        else:
            app.logger.warning("Imagen API did not return a valid image.")
            return None
    except Exception as e:
        app.logger.error(f"Error calling Imagen API: {e}", exc_info=True)
        return None

# --- generate_prompts_async function (MODIFIED for sequential calls) ---
def generate_prompts_sync_wrapper(raw_input, language_code="en-US"):
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
    polished_prompt = ask_gemini_for_prompt(polished_prompt_instruction)

    if "Error" in polished_prompt or "not configured" in polished_prompt:
        return {
            "polished": polished_prompt,
            "creative": "", "technical": "", "shorter": "", "additions": ""
        }

    strict_instruction_suffix = "\n\nDo NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided text."

    # Making sequential calls now since ask_gemini_for_prompt is synchronous
    creative_result = ask_gemini_for_prompt(language_instruction_prefix + f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_prompt}{strict_instruction_suffix}")
    technical_result = ask_gemini_for_prompt(language_instruction_prefix + f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_prompt}{strict_instruction_suffix}")
    shorter_result = ask_gemini_for_prompt(language_instruction_prefix + f"Condense the following prompt into its shortest possible form while retaining all essential meaning and instructions. Aim for brevity.:\n\n{polished_prompt}{strict_instruction_suffix}", max_output_tokens=512)

    additions_result = ask_gemini_for_prompt(language_instruction_prefix + f"""Analyze the following prompt and suggest potential additions to improve its effectiveness for a large language model. Focus on elements like:
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
    # The root route now serves the new landing page
    return render_template('landing.html', current_user=current_user)

@app.route('/app') # NEW: Route for the main AI Prompt Generator application
@login_required # Protect this route, only accessible after login
def app_generator_page():
    # This route now serves the renamed app_generator.html
    return render_template('app_generator.html', current_user=current_user)


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
        # Call the synchronous wrapper function
        results = generate_prompts_sync_wrapper(raw_input, language_code)
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
        return redirect(url_for('app_generator_page')) # Redirect to app page after register if already logged in

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form['password']
        email = request.form.get('email') # Assuming you add an email field to register.html

        if not username and not email:
            flash('Username or Email is required.', 'danger')
            return render_template('register.html')

        # Check for existing username (if provided)
        if username and User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('register.html')
        
        # Check for existing email
        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists. Please log in or use a different email.', 'danger')
            return render_template('register.html')

        new_user = User(username=username, email=email)
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
        return redirect(url_for('app_generator_page')) # Redirect to app page after login if already logged in

    if request.method == 'POST':
        username_or_email = request.form['username_or_email'] # Changed to handle both
        password = request.form['password']
        remember_me = 'remember_me' in request.form

        # Try to find user by username or email
        user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()

        if user and user.check_password(password):
            login_user(user, remember=remember_me)
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next') # Redirect to the page user tried to access
            return redirect(next_page or url_for('app_generator_page')) # Redirect to app page
        else:
            flash('Login Unsuccessful. Please check username/email and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required # Only logged-in users can log out
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index')) # Redirect to the new landing page after logout

# --- NEW: Google Authentication Routes ---
@app.route('/login/google')
def login_google():
    # If the user is already authenticated, redirect them
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('app_generator_page')) # Redirect to app page if already logged in
    
    # Generate a nonce to prevent replay attacks
    # The nonce is stored in the session and checked upon callback
    nonce = os.urandom(16).hex()
    session['oauth_nonce'] = nonce

    # Generate the redirect URI for Google OAuth callback
    # _external=True ensures a full URL is generated, necessary for OAuth
    redirect_uri = url_for('auth_google', _external=True)
    
    # Redirect the user to Google's authorization endpoint, including the nonce
    return oauth.google.authorize_redirect(redirect_uri, nonce=nonce)

@app.route('/auth/google')
def auth_google():
    try:
        # Attempt to authorize the access token from Google's response
        token = oauth.google.authorize_access_token()
    except Exception as e:
        # Log any errors during the OAuth token exchange
        app.logger.error(f"Error during Google authentication: {e}")
        flash('Google login failed. Please try again.', 'danger')
        return redirect(url_for('login'))

    # Retrieve the nonce from the session
    nonce = session.pop('oauth_nonce', None)
    if not nonce:
        # If nonce is missing from session, it could be a security issue or session problem
        app.logger.error("Nonce missing from session during Google OAuth callback.")
        flash('Google login failed due to a security issue. Please try again.', 'danger')
        return redirect(url_for('login'))

    # Parse the ID token to get user information (email, name, Google ID 'sub')
    # Pass the retrieved nonce to the parse_id_token method
    userinfo = oauth.google.parse_id_token(token, nonce=nonce)
    
    # Check if a user already exists in our database with this Google ID
    user = User.query.filter_by(google_id=userinfo['sub']).first()

    if user is None:
        # If no user found by Google ID, check if a user exists with this email
        # This handles cases where a user might have a local account and then tries Google login
        user = User.query.filter_by(email=userinfo['email']).first()
        if user:
            # If an existing local account is found with the same email, link the Google ID
            user.google_id = userinfo['sub']
            # Update username if it was null or generic, using Google's provided name
            if not user.username:
                user.username = userinfo.get('name', userinfo['email'].split('@')[0])
            db.session.commit()
            flash('Your Google account has been linked to your existing account!', 'success')
        else:
            # If no existing account (local or Google) is found, create a new user
            user = User(
                google_id=userinfo['sub'],
                email=userinfo['email'],
                # Use Google's provided name as username, or derive from email if name is not available
                username=userinfo.get('name', userinfo['email'].split('@')[0]) 
            )
            db.session.add(user)
            db.session.commit()
            flash('Account created successfully via Google!', 'success')

    # Log the user into Flask-Login session
    login_user(user)
    flash('Logged in successfully with Google!', 'success')
    return redirect(url_for('app_generator_page')) # Redirect to the main app page after Google login

# --- NEW: Endpoint for dynamic daily content (motivational prompt and image) ---
@app.route('/generate_daily_content', methods=['GET'])
def generate_daily_content():
    current_hour = datetime.now().hour
    
    # Determine time of day for prompt/image context
    if 5 <= current_hour < 12:
        time_of_day = "morning"
        prompt_theme = "new beginnings, fresh start, productivity, sunrise"
    elif 12 <= current_hour < 18:
        time_of_day = "afternoon"
        prompt_theme = "focus, progress, overcoming challenges, bright sky"
    else: # 18 <= current_hour < 5
        time_of_day = "evening"
        prompt_theme = "reflection, relaxation, future dreams, starry night"

    # Generate motivational prompt using Gemini (now synchronous)
    motivational_prompt_instruction = f"Generate a short, inspiring, and motivational quote or sentence suitable for the {time_of_day}, focusing on themes of {prompt_theme}. Keep it concise, under 20 words."
    motivational_text = ask_gemini_for_prompt(motivational_prompt_instruction, max_output_tokens=50)

    # Generate image using Imagen (now synchronous)
    image_prompt = f"A beautiful, inspiring image representing a {time_of_day} with elements of {prompt_theme}. Artistic, serene, high quality. Digital art."
    image_url = generate_image_with_imagen(image_prompt)
    
    if not image_url:
        # Fallback to a placeholder if image generation fails
        if time_of_day == "morning":
            image_url = "https://placehold.co/800x400/ADD8E6/000000?text=Morning+Inspiration"
        elif time_of_day == "afternoon":
            image_url = "https://placehold.co/800x400/90EE90/000000?text=Afternoon+Focus"
        else: # evening
            image_url = "https://placehold.co/800x400/8A2BE2/FFFFFF?text=Evening+Reflection"

    return jsonify({
        "motivational_text": motivational_text,
        "image_url": image_url
    })
# --- END NEW: Endpoint for dynamic daily content ---


# --- Database Initialization (Run once to create tables) ---
# This block ensures tables are created when the app starts.
# In production, you might use Flask-Migrate or a separate script.
with app.app_context():
    db.create_all()
    app.logger.info("Database tables created/checked.")

# --- Main App Run ---
if __name__ == '__main__':
    # Run the Flask app in debug mode, accessible from any IP on port 5000 (or specified by PORT env var)
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))
