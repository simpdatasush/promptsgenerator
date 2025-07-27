# REMOVE THESE TWO LINES IF THEY ARE PRESENT AT THE TOP OF YOUR FILE
# import nest_asyncio
# nest_asyncio.apply()

import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash
import logging
from datetime import datetime, timedelta # Import timedelta for time calculations
import re # Import for regular expressions
from functools import wraps # Import wraps for decorators
import base64 # Import base64 for image processing

# --- NEW IMPORTS FOR AUTHENTICATION ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# --- END NEW IMPORTS ---

# NEW: Import for specific Gemini API exceptions
from google.api_core import exceptions as google_api_exceptions


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
# Corrected logging setup:
stream_handler = logging.StreamHandler() # Create a StreamHandler instance
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Create a Formatter instance
stream_handler.setFormatter(formatter) # Set the formatter on the handler
app.logger.addHandler(stream_handler) # Add the configured handler to the app logger


# --- Temporary In-Memory Storage for Saved Prompts (Unchanged) ---
saved_prompts_in_memory = []

# --- NEW: Cooldown configuration (no longer in-memory dict for last_request_time) ---
COOLDOWN_SECONDS = 60 # 60 seconds cooldown

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
       app.logger.warning("WARNING: GEMINI_API_KEY environment variable not set. Prompt generation features will be disabled.")
       app.logger.warning("Please set the GEMINI_API_KEY environment variable on Render or your deployment environment.")
       app.logger.warning("="*80 + "\n")
       GEMINI_API_CONFIGURED = False

configure_gemini_api()

# --- UPDATED: User Model for SQLAlchemy and Flask-Login ---
class User(db.Model, UserMixin):
   id = db.Column(db.Integer, primary_key=True)
   username = db.Column(db.String(80), unique=True, nullable=False)
   password_hash = db.Column(db.String(128), nullable=False)
   is_admin = db.Column(db.Boolean, default=False)
   last_prompt_request = db.Column(db.DateTime, nullable=True) # For cooldown
   daily_prompt_count = db.Column(db.Integer, default=0, nullable=False) # NEW: Daily prompt count
   last_count_reset_date = db.Column(db.Date, nullable=True) # NEW: Date when count was last reset

   def set_password(self, password):
       self.password_hash = generate_password_hash(password)

   def check_password(self, password):
       return check_password_hash(self.password_hash, password)

   def __repr__(self):
       return f'<User {self.username}>'

# --- NEW: RawPrompt Model for storing user's raw input requests ---
class RawPrompt(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
   raw_text = db.Column(db.Text, nullable=False)
   timestamp = db.Column(db.DateTime, default=datetime.utcnow)

   user = db.relationship('User', backref=db.backref('raw_prompts', lazy=True))

   def __repr__(self):
       return f'<RawPrompt {self.id} by User {self.user_id}>'
# --- END NEW: RawPrompt Model ---

# --- UPDATED: News Model for storing news items ---
class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow) # When added to our app
    published_date = db.Column(db.DateTime, nullable=True) # Actual publication date
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Who added it

    user = db.relationship('User', backref=db.backref('news_items', lazy=True))

    def __repr__(self):
        return f'<News {self.title}>'
# --- END UPDATED: News Model ---

# --- UPDATED: Job Model for storing job listings (added published_date) ---
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    company = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255), nullable=True)
    url = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow) # For sorting/reposting
    published_date = db.Column(db.DateTime, nullable=True) # NEW: Actual publication date for jobs
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Who added it

    user = db.relationship('User', backref=db.backref('job_listings', lazy=True))

    def __repr__(self):
        return f'<Job {self.title} at {self.company}>'
# --- END UPDATED: Job Model ---


# --- NEW: Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
   return db.session.get(User, int(user_id)) # Use db.session.get for primary key lookup
# --- END NEW: User Model and Loader ---

# --- NEW: Admin Required Decorator ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('app_home')) # Redirect to app home or a specific unauthorized page
        return f(*args, **kwargs)
    return decorated_function
# --- END NEW: Admin Required Decorator ---


# --- Response Filtering Function (UPDATED) ---
def filter_gemini_response(text):
    unauthorized_message = "I am not authorised to answer this question. My purpose is solely to refine your raw prompt into a machine-readable format."
    text_lower = text.lower()

    # Generic unauthorized phrases
    unauthorized_phrases = [
        "as a large language model", "i am an ai", "i was trained by", "my training data",
        "this application was built using", "the code for this app", "i cannot fulfill this request because",
        "i apologize, but i cannot", "i'm sorry, but i cannot", "i am unable to", "i do not have access",
        "i am not able to", "i cannot process", "i cannot provide", "i am not programmed",
        "i cannot generate", "i cannot give you details about my internal workings",
        "i cannot discuss my creation or operation", "i cannot explain the development of this tool",
        "my purpose is to", "i am designed to", "i don't have enough information to"
    ]
    for phrase in unauthorized_phrases:
        if phrase in text_lower:
            if phrase == "i don't have enough information to" and \
               ("about the provided prompt" in text_lower or "based on your input" in text_lower or "to understand the context" in text_lower):
                continue
            return unauthorized_message

    # Generic bug/error phrases
    bug_phrases = [
        "a bug occurred", "i encountered an error", "there was an issue in my processing",
        "i made an error", "my apologies", "i cannot respond to that"
    ]
    for phrase in bug_phrases:
        if phrase in text_lower:
            return unauthorized_message

    # Specific filtering for Gemini API quota/internal errors
    # This will remove any detailed JSON or specific model/API references
    if "you exceeded your current quota" in text_lower:
        return "You exceeded your current quota. Please try again later or check your plan and billing details."
    
    # Catch-all for any API-related error details
    if "error communicating with gemini api:" in text_lower or "no response from model." in text_lower:
        # Remove any specific model names, API keys, or detailed error structures
        filtered_text = text
        filtered_text = re.sub(r"model: \"[a-zA-Z0-9-.]+\"", "model: \"[REDACTED]\"", filtered_text)
        filtered_text = re.sub(r"quota_metric: \"[^\"]+\"", "quota_metric: \"[REDACTED]\"", filtered_text)
        filtered_text = re.sub(r"quota_id: \"[^\"]+\"", "quota_id: \"[REDACTED]\"", filtered_text)
        filtered_text = re.sub(r"quota_dimensions \{[^\}]+\}", "quota_dimensions { [REDACTED] }", filtered_text)
        filtered_text = re.sub(r"links \{\s*description: \"[^\"]+\"\s*url: \"[^\"]+\"\s*\}", "links { [REDACTED] }", filtered_text)
        filtered_text = re.sub(r"retry_delay \{\s*seconds: \d+\s*\}", "retry_delay { [REDACTED] }", filtered_text)
        filtered_text = re.sub(r"\[violations \{.*?\}\s*,?\s*links \{.*?\}\s*,?\s*retry_delay \{.*?\}\s*\]", "", filtered_text, flags=re.DOTALL)
        filtered_text = re.sub(r"\[violations \{.*?\}\s*\]", "", filtered_text, flags=re.DOTALL) # In case only violations are present

        # If after filtering, it's still too verbose or contains sensitive info, generalize
        if "google.api_core.exceptions" in filtered_text.lower() or "api_key" in filtered_text.lower():
            return "There was an issue with the AI service. Please try again later."
        
        return filtered_text.strip() # Return the filtered text

    return text

# --- Gemini API interaction function (NOW SYNCHRONOUS) ---
def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=512):
   if not GEMINI_API_CONFIGURED:
       # This check is also done in the endpoint, but kept here for robustness
       return "Gemini API Key is not configured or the AI model failed to initialize."

   try:
       gemini_model_instance = genai.GenerativeModel('gemini-2.0-flash')

       generation_config = {
           "max_output_tokens": max_output_tokens,
           "temperature": 0.1
       }

       # USE THE SYNCHRONOUS generate_content METHOD
       response = gemini_model_instance.generate_content(
           contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
           generation_config=generation_config
       )
       raw_gemini_text = response.text if response and response.text else "No response from model."
       return filter_gemini_response(raw_gemini_text).strip()
   except google_api_exceptions.GoogleAPICallError as e: # Catch specific API errors
       app.logger.error(f"DEBUG: Google API Call Error: {e}", exc_info=True)
       # Pass the string representation of the exception to the filter
       return filter_gemini_response(f"Error communicating with Gemini API: {str(e)}")
   except Exception as e: # Catch any other unexpected errors
       app.logger.error(f"DEBUG: Unexpected Error calling Gemini API: {e}", exc_info=True)
       return filter_gemini_response(f"An unexpected error occurred: {str(e)}")


# --- NEW: Gemini API for Image Understanding ---
def ask_gemini_for_image_text(image_data_bytes):
   if not GEMINI_API_CONFIGURED:
       return "Gemini API Key is not configured or the AI model failed to initialize."

   try:
       gemini_model_instance = genai.GenerativeModel('gemini-2.0-flash')
      
       # Prepare the image for the Gemini API
       image_part = {
           "mime_type": "image/jpeg", # Assuming JPEG for simplicity, can be dynamic
           "data": image_data_bytes
       }

       # Instruction for the model to extract text
       prompt_parts = [
           image_part,
           "Extract all text from this image, including handwritten text. Provide only the extracted text, without any additional commentary or formatting."
       ]

       response = gemini_model_instance.generate_content(prompt_parts)
       extracted_text = response.text if response and response.text else ""
       return filter_gemini_response(extracted_text).strip() # Filter image response too
   except google_api_exceptions.GoogleAPICallError as e: # Catch specific API errors
       app.logger.error(f"Error calling Gemini API for image text extraction: {e}", exc_info=True)
       return filter_gemini_response(f"Error extracting text from image: {str(e)}")
   except Exception as e: # Catch any other unexpected errors
       app.logger.error(f"Unexpected Error calling Gemini API for image text extraction: {e}", exc_info=True)
       return filter_gemini_response(f"An unexpected error occurred during image text extraction: {str(e)}")
# --- END NEW ---


# --- generate_prompts_async function (main async logic for prompt variations) ---
async def generate_prompts_async(raw_input, language_code="en-US"):
   if not raw_input.strip():
       return {
           "polished": "Please enter some text to generate prompts.",
           "creative": "",
           "technical": "",
       }

   target_language_name = LANGUAGE_MAP.get(language_code, "English")
   language_instruction_prefix = f"The output MUST be entirely in {target_language_name}. "

   polished_prompt_instruction = language_instruction_prefix + f"""Refine the following text into a clear, concise, and effective prompt for a large language model. Improve grammar, clarity, and structure. Do not add external information, only refine the given text. Crucially, do NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided raw text into a better prompt. Raw Text: {raw_input}"""

   # CALL SYNCHRONOUS ask_gemini_for_prompt IN A SEPARATE THREAD
   polished_prompt_result = await asyncio.to_thread(ask_gemini_for_prompt, polished_prompt_instruction)

   if "Error" in polished_prompt_result or "not configured" in polished_prompt_result or "quota" in polished_prompt_result.lower(): # Check for quota error
       return {
           "polished": polished_prompt_result,
           "creative": "",
           "technical": "",
       }

   strict_instruction_suffix = "\n\nDo NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided text."

   # Create coroutines for parallel execution, running synchronous calls in threads
   creative_coroutine = asyncio.to_thread(ask_gemini_for_prompt, language_instruction_prefix + f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_prompt_result}{strict_instruction_suffix}")
   technical_coroutine = asyncio.to_thread(ask_gemini_for_prompt, language_instruction_prefix + f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_prompt_result}{strict_instruction_suffix}")

   creative_result, technical_result = await asyncio.gather(
       creative_coroutine, technical_coroutine
   )

   return {
       "polished": polished_prompt_result,
       "creative": creative_result,
       "technical": technical_result,
   }

# --- NEW: Reverse Prompting function ---
async def generate_reverse_prompt_async(input_text, language_code="en-US"):
    if not input_text.strip():
        return "Please provide text or code to infer a prompt from."

    # Enforce character limit
    MAX_REVERSE_PROMPT_CHARS = 10000
    if len(input_text) > MAX_REVERSE_PROMPT_CHARS:
        return f"Input for reverse prompting exceeds the {MAX_REVERSE_PROMPT_CHARS} character limit. Please shorten your input."

    target_language_name = LANGUAGE_MAP.get(language_code, "English")
    language_instruction_prefix = f"The output MUST be entirely in {target_language_name}. "

    # Escape curly braces in input_text to prevent f-string parsing errors
    # This replaces single { with {{ and single } with }}
    escaped_input_text = input_text.replace('{', '{{').replace('}', '}}')

    # The core instruction for reverse prompting
    # Using concatenation to avoid f-string parsing issues with embedded user input
    reverse_prompt_instruction = (
        language_instruction_prefix +
        "Given the following text or code, infer the most effective and concise prompt that would have generated it. Focus on the core instruction, any implied constraints, style requirements, or specific formats. Do not add conversational filler, explanations, or preambles. Provide only the inferred prompt.\n\n"
        "Input Text/Code:\n"
        "---\n"
        + escaped_input_text +
        "\n---\n\n"
        "Inferred Prompt:"
    )

    # Corrected f-string for logging: double the literal curly brace
    app.logger.info(f"Sending reverse prompt instruction to Gemini (length: {len(reverse_prompt_instruction)} chars)}}")

    # Call synchronous ask_gemini_for_prompt in a separate thread
    reverse_prompt_result = await asyncio.to_thread(ask_gemini_for_prompt, reverse_prompt_instruction, max_output_tokens=512) # Use a reasonable max_output_tokens for a prompt

    return reverse_prompt_result
# --- END NEW: Reverse Prompting function ---


# --- Flask Routes ---

# UPDATED: Landing page route to fetch more news AND jobs
@app.route('/')
def landing():
   # Fetch latest 10 news items for the landing page
   news_items = News.query.order_by(News.timestamp.desc()).limit(10).all()
   # Fetch latest 10 job listings for the landing page
   job_listings = Job.query.order_by(Job.timestamp.desc()).limit(10).all()
   return render_template('landing.html', news_items=news_items, job_listings=job_listings, current_user=current_user)

# Renamed original index route to /app_home
@app.route('/app_home')
def app_home():
   # Pass current_user object to the template to show login/logout status
   return render_template('index.html', current_user=current_user)

# NEW: LLM Benchmark Page Route
@app.route('/llm_benchmark')
def llm_benchmark():
    return render_template('llm_benchmark.html', current_user=current_user)


@app.route('/generate', methods=['POST'])
@login_required # Protect this route
async def generate_prompts_endpoint(): # This remains async
   user = current_user # Get the current user object
   now = datetime.utcnow() # Use utcnow for consistency with database default

   # --- UPDATED: Cooldown Check using database timestamp ---
   if user.last_prompt_request:
       time_since_last_request = (now - user.last_prompt_request).total_seconds()
       if time_since_last_request < COOLDOWN_SECONDS:
           remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
           app.logger.info(f"User {user.username} is on cooldown. Remaining: {remaining_time}s")
           return jsonify({
               "error": f"Please wait {remaining_time} seconds before generating new prompts.",
               "cooldown_active": True,
               "remaining_time": remaining_time
           }), 429 # 429 Too Many Requests
   # --- END UPDATED ---

   # --- NEW: Daily Limit Check ---
   if not user.is_admin: # Admins are exempt from the daily limit
       today = now.date()
       if user.last_count_reset_date != today:
           user.daily_prompt_count = 0
           user.last_count_reset_date = today
           db.session.add(user) # Mark user as modified
           db.session.commit() # Commit reset immediately to prevent race conditions on count

       if user.daily_prompt_count >= 10: # Max 10 generations per day
           app.logger.info(f"User {user.username} exceeded daily prompt limit.")
           return jsonify({
               "error": "You have reached your daily limit of 10 prompt generations. Please try again tomorrow.",
               "daily_limit_reached": True
           }), 429 # 429 Too Many Requests
   # --- END NEW: Daily Limit Check ---


   raw_input = request.form.get('prompt_input', '').strip()
   language_code = request.form.get('language_code', 'en-US')

   if not raw_input:
       # This is handled client-side in index.html, but kept as a server-side fallback
       return jsonify({
           "polished": "Please enter some text to generate prompts.",
           "creative": "",
           "technical": "",
       })

   # --- ADDED: Explicit check for Gemini API configuration ---
   if not GEMINI_API_CONFIGURED:
       app.logger.error("Gemini API Key is not configured. Cannot generate prompts.")
       return jsonify({
           "error": "Gemini API Key is not configured. Please set the GEMINI_API_KEY environment variable in your deployment environment."
       }), 500 # Return 500 Internal Server Error as it's a server-side configuration issue
   # --- END ADDED ---

   try:
       # Await the async function directly
       results = await generate_prompts_async(raw_input, language_code)

       # --- UPDATED: Update last_prompt_request in database and Save raw_input ---
       user.last_prompt_request = now # Record the time of this successful request
       if not user.is_admin: # Only increment count for non-admin users
           user.daily_prompt_count += 1
       db.session.add(user) # Add the user object back to the session to mark it as modified
       db.session.commit()
       app.logger.info(f"User {user.username}'s last prompt request time updated and count incremented. (Forward Prompt)")

       if current_user.is_authenticated:
           try:
               new_raw_prompt = RawPrompt(user_id=current_user.id, raw_text=raw_input)
               db.session.add(new_raw_prompt)
               db.session.commit()
               app.logger.info(f"Raw prompt saved for user {current_user.username}")
           except Exception as e:
               app.logger.error(f"Error saving raw prompt for user {current_user.username}: {e}")
               db.session.rollback() # Rollback in case of error
       # --- END UPDATED ---

       return jsonify(results)
   except Exception as e:
       app.logger.exception("Error during prompt generation in endpoint:")
       return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500

# --- NEW: Reverse Prompt Endpoint ---
@app.route('/reverse_prompt', methods=['POST'])
@login_required
async def reverse_prompt_endpoint():
    user = current_user
    now = datetime.utcnow()

    # Apply cooldown to reverse prompting as well
    if user.last_prompt_request: # Reusing the same cooldown for simplicity
        time_since_last_request = (now - user.last_prompt_request).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            app.logger.info(f"User {user.username} is on cooldown for reverse prompt. Remaining: {remaining_time}s")
            return jsonify({
                "error": f"Please wait {remaining_time} seconds before performing another reverse prompt.",
                "cooldown_active": True,
                "remaining_time": remaining_time
            }), 429

    # --- NEW: Daily Limit Check for Reverse Prompt ---
    if not user.is_admin: # Admins are exempt from the daily limit
        today = now.date()
        if user.last_count_reset_date != today:
            user.daily_prompt_count = 0
            user.last_count_reset_date = today
            db.session.add(user) # Mark user as modified
            db.session.commit() # Commit reset immediately to prevent race conditions on count

        if user.daily_prompt_count >= 10: # Max 10 generations per day
            app.logger.info(f"User {user.username} exceeded daily reverse prompt limit.")
            return jsonify({
                "error": "You have reached your daily limit of 10 prompt generations. Please try again tomorrow.",
                "daily_limit_reached": True
            }), 429 # 429 Too Many Requests
    # --- END NEW: Daily Limit Check ---

    data = request.get_json()
    input_text = data.get('input_text', '').strip()
    language_code = data.get('language_code', 'en-US')

    if not input_text:
        # This is handled client-side in index.html, but kept as a server-side fallback
        return jsonify({"error": "Please provide text or code to infer a prompt from."}), 400

    if not GEMINI_API_CONFIGURED:
        app.logger.error("Gemini API Key is not configured. Cannot perform reverse prompting.")
        return jsonify({
            "error": "Gemini API Key is not configured. Please set the GEMINI_API_KEY environment variable."
        }), 500

    try:
        inferred_prompt = await generate_reverse_prompt_async(input_text, language_code)

        # Update last_prompt_request after successful reverse prompt
        user.last_prompt_request = now
        if not user.is_admin: # Only increment count for non-admin users
            user.daily_prompt_count += 1
        db.session.add(user)
        db.session.commit()
        app.logger.info(f"User {user.username}'s last prompt request time updated and count incremented after reverse prompt.")

        return jsonify({"inferred_prompt": inferred_prompt})
    except Exception as e:
        app.logger.exception("Error during reverse prompt generation in endpoint:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500
# --- END NEW: Reverse Prompt Endpoint ---

# --- NEW: Image Processing Endpoint ---
@app.route('/process_image_prompt', methods=['POST'])
@login_required
async def process_image_prompt_endpoint():
    user = current_user
    now = datetime.utcnow()

    # Apply cooldown to image processing as well
    if user.last_prompt_request:
        time_since_last_request = (now - user.last_prompt_request).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            app.logger.info(f"User {user.username} is on cooldown for image processing. Remaining: {remaining_time}s")
            return jsonify({
                "error": f"Please wait {remaining_time} seconds before processing another image.",
                "cooldown_active": True,
                "remaining_time": remaining_time
            }), 429

    # Daily limit check for image processing
    if not user.is_admin:
        today = now.date()
        if user.last_count_reset_date != today:
            user.daily_prompt_count = 0
            user.last_count_reset_date = today
            db.session.add(user)
            db.session.commit()

        if user.daily_prompt_count >= 10:
            app.logger.info(f"User {user.username} exceeded daily image processing limit.")
            return jsonify({
                "error": "You have reached your daily limit of 10 image processing requests. Please try again tomorrow.",
                "daily_limit_reached": True
            }), 429

    data = request.get_json()
    image_data_b64 = data.get('image_data')
    language_code = data.get('language_code', 'en-US') # Not directly used by Gemini Vision, but good to pass

    if not image_data_b64:
        return jsonify({"error": "No image data provided."}), 400

    try:
        image_data_bytes = base64.b64decode(image_data_b64) # Decode base64 string to bytes
        
        # Call the Gemini API for image understanding
        recognized_text = await asyncio.to_thread(ask_gemini_for_image_text, image_data_bytes)

        # Update last_prompt_request after successful image processing
        user.last_prompt_request = now
        if not user.is_admin:
            user.daily_prompt_count += 1
        db.session.add(user)
        db.session.commit()
        app.logger.info(f"User {user.username}'s last prompt request time updated and count incremented after image processing.")

        return jsonify({"recognized_text": recognized_text})
    except Exception as e:
        app.logger.exception("Error during image processing endpoint:")
        return jsonify({"error": f"An unexpected server error occurred during image processing: {e}. Please check server logs for details."}), 500

# --- END NEW: Image Processing Endpoint ---


# --- UPDATED: Endpoint to check cooldown status for frontend ---
@app.route('/check_cooldown', methods=['GET'])
@login_required
def check_cooldown_endpoint():
    user = current_user
    now = datetime.utcnow() # Use utcnow for consistency

    cooldown_active = False
    remaining_time = 0
    if user.last_prompt_request:
        time_since_last_request = (now - user.last_prompt_request).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            cooldown_active = True
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)

    daily_limit_reached = False
    daily_count = 0
    if not user.is_admin: # Check daily limit only for non-admins
        today = now.date()
        if user.last_count_reset_date != today:
            # If the last reset date is not today, reset the count for the current session's check
            # (The actual DB reset happens on the next prompt generation)
            daily_count = 0
        else:
            daily_count = user.daily_prompt_count
        
        if daily_count >= 10:
            daily_limit_reached = True

    return jsonify({
        "cooldown_active": cooldown_active,
        "remaining_time": remaining_time,
        "daily_limit_reached": daily_limit_reached,
        "daily_count": daily_count,
        "is_admin": user.is_admin
    }), 200
# --- END UPDATED ---


# --- UPDATED: Admin News Management Routes ---
@app.route('/admin/news', methods=['GET'])
@login_required
@admin_required
def admin_news():
    news_items = News.query.order_by(News.timestamp.desc()).all()
    return render_template('admin_news.html', news_items=news_items, current_user=current_user)

@app.route('/admin/news/add', methods=['POST'])
@login_required
@admin_required
def add_news():
    title = request.form.get('title')
    url = request.form.get('url')
    description = request.form.get('description')
    published_date_str = request.form.get('published_date') # NEW: Get published_date string

    if not title or not url:
        flash('Title and URL are required to add news.', 'danger')
        return redirect(url_for('admin_news'))

    published_date = None
    if published_date_str:
        try:
            published_date = datetime.strptime(published_date_str, '%Y-%m-%d') # Parse date
        except ValueError:
            flash('Invalid Published Date format. Please use YYYY-MM-DD.', 'danger')
            return redirect(url_for('admin_news'))

    try:
        new_news = News(title=title, url=url, description=description, published_date=published_date, user_id=current_user.id) # Use published_date
        db.session.add(new_news)
        db.session.commit()
        flash('News item added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding news item: {e}', 'danger')
    return redirect(url_for('admin_news'))

@app.route('/admin/news/delete/<int:news_id>', methods=['POST'])
@login_required
@admin_required
def delete_news(news_id):
    news_item = News.query.get_or_404(news_id)
    try:
        db.session.delete(news_item)
        db.session.commit()
        flash('News item deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting news item: {e}', 'danger')
    return redirect(url_for('admin_news'))

# --- NEW: Repost News Route ---
@app.route('/admin/news/repost/<int:news_id>', methods=['POST'])
@login_required
@admin_required
def repost_news(news_id):
    news_item = News.query.get_or_404(news_id)
    try:
        news_item.timestamp = datetime.utcnow() # Update timestamp to current UTC time
        db.session.commit()
        flash(f'News item "{news_item.title}" reposted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error reposting news item: {e}', 'danger')
    return redirect(url_for('admin_news'))
# --- END NEW: Repost News Route ---
# --- END UPDATED: Admin News Management Routes ---

# --- UPDATED: Admin Jobs Management Routes (added published_date) ---
@app.route('/admin/jobs', methods=['GET'])
@login_required
@admin_required
def admin_jobs():
    job_listings = Job.query.order_by(Job.timestamp.desc()).all()
    return render_template('admin_jobs.html', job_listings=job_listings, current_user=current_user)

@app.route('/admin/jobs/add', methods=['POST'])
@login_required
@admin_required
def add_job():
    title = request.form.get('title')
    company = request.form.get('company')
    location = request.form.get('location')
    url = request.form.get('url')
    description = request.form.get('description')
    published_date_str = request.form.get('published_date') # NEW: Get published_date string for jobs

    if not title or not company or not url:
        flash('Title, Company, and URL are required to add a job listing.', 'danger')
        return redirect(url_for('admin_jobs'))

    published_date = None
    if published_date_str:
        try:
            published_date = datetime.strptime(published_date_str, '%Y-%m-%d') # Parse date
        except ValueError:
            flash('Invalid Published Date format for job. Please use YYYY-MM-DD.', 'danger')
            return redirect(url_for('admin_jobs'))

    try:
        new_job = Job(title=title, company=company, location=location, url=url, description=description, published_date=published_date, user_id=current_user.id) # Use published_date
        db.session.add(new_job)
        db.session.commit()
        flash('Job listing added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding job listing: {e}', 'danger')
    return redirect(url_for('admin_jobs'))

@app.route('/admin/jobs/delete/<int:job_id>', methods=['POST'])
@login_required
@admin_required
def delete_job(job_id):
    job_listing = Job.query.get_or_404(job_id)
    try:
        db.session.delete(job_listing)
        db.session.commit()
        flash('Job listing deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting job listing: {e}', 'danger')
    return redirect(url_for('admin_jobs'))

@app.route('/admin/jobs/repost/<int:job_id>', methods=['POST'])
@login_required
@admin_required
def repost_job(job_id):
    job_listing = Job.query.get_or_404(job_id)
    try:
        job_listing.timestamp = datetime.utcnow() # Update timestamp to current UTC time
        db.session.commit()
        flash(f'Job listing "{job_listing.title}" reposted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error reposting job listing: {e}', 'danger')
    return redirect(url_for('admin_jobs'))
# --- END UPDATED: Admin Jobs Management Routes ---


# --- NEW: Change Password Route ---
@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_new_password = request.form.get('confirm_new_password')

        if not current_password or not new_password or not confirm_new_password:
            flash('All fields are required.', 'danger')
            return render_template('change_password.html', current_user=current_user)

        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'danger')
            return render_template('change_password.html', current_user=current_user)

        if new_password != confirm_new_password:
            flash('New password and confirmation do not match.', 'danger')
            return render_template('change_password.html', current_user=current_user)

        if len(new_password) < 6: # Example: enforce minimum password length
            flash('New password must be at least 6 characters long.', 'danger')
            return render_template('change_password.html', current_user=current_user)

        try:
            current_user.set_password(new_password)
            db.session.commit()
            flash('Your password has been changed successfully!', 'success')
            return redirect(url_for('app_home'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while changing your password: {e}', 'danger')
            app.logger.error(f"Error changing password for user {current_user.username}: {e}")

    return render_template('change_password.html', current_user=current_user)
# --- END NEW: Change Password Route ---


# --- Save Prompt Endpoint ---
@app.route('/save_prompt', methods=['POST'])
@login_required
def save_prompt():
   data = request.get_json()
   # --- FIX: Match frontend keys 'prompt_text' and 'prompt_type' ---
   prompt_type = data.get('prompt_type')
   prompt_content = data.get('prompt_text')

   if not prompt_content or not prompt_type:
       app.logger.warning(f"Attempted to save empty prompt text or type. Type: '{prompt_type}', Content: '{prompt_content[:50] if prompt_content else ''}'")
       return jsonify({"success": False, "message": "No content or type provided for saving."}), 400

   timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   saved_prompts_in_memory.append({
       "timestamp": timestamp,
       "type": prompt_type, # Store as 'type' for consistency with existing display logic
       "text": prompt_content, # Store as 'text' for consistency with existing display logic
       "user": current_user.username if current_user.is_authenticated else "anonymous" # Track user
   })

   app.logger.info(f"Prompt of type '{prompt_type}' saved to memory at {timestamp} by {current_user.username if current_user.is_authenticated else 'anonymous'}. Content: '{prompt_content[:50]}...'")
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

# --- NEW: Get Raw Prompts Endpoint ---
@app.route('/get_raw_prompts', methods=['GET'])
@login_required
def get_raw_prompts_endpoint():
   if not current_user.is_authenticated:
       # If not logged in, return an empty list or redirect to login
       # For this case, returning empty list is fine for UI
       return jsonify([]), 200

   try:
       # Fetch last 10 raw prompts for the current user, ordered by timestamp descending
       raw_prompts = RawPrompt.query.filter_by(user_id=current_user.id) \
                                    .order_by(RawPrompt.timestamp.desc()) \
                                    .limit(10) \
                                    .all()

       # Format for JSON response
       formatted_prompts = [{
           "id": p.id,
           "raw_text": p.raw_text,
           "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S")
       } for p in raw_prompts]

       return jsonify(formatted_prompts), 200
   except Exception as e:
       app.logger.error(f"Error fetching raw prompts for user {current_user.username}: {e}")
       return jsonify({"error": "Failed to retrieve past raw requests."}), 500
# --- END NEW ---


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


# --- UPDATED: Authentication Routes for automatic redirect ---
@app.route('/register', methods=['GET', 'POST'])
def register():
   if current_user.is_authenticated:
       flash('You are already registered and logged in.', 'info')
       return redirect(url_for('app_home'))

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
           login_user(new_user) # Automatically log in the new user
           flash('Registration successful! You are now logged in.', 'success')
           return redirect(url_for('app_home')) # Redirect directly to app_home
   return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
   if current_user.is_authenticated:
       flash('You are already logged in.', 'info')
       return redirect(url_for('app_home'))

   if request.method == 'POST':
       username = request.form['username']
       password = request.form['password']
       remember_me = 'remember_me' in request.form

       user = User.query.filter_by(username=username).first()
       if user and user.check_password(password):
           login_user(user, remember=remember_me)
           flash('Logged in successfully!', 'success')
           return redirect(url_for('app_home')) # Redirect directly to app_home
       else:
           flash('Login Unsuccessful. Please check username and password.', 'danger')
   return render_template('login.html')

@app.route('/logout')
@login_required # Only logged-in users can log out
def logout():
   logout_user()
   flash('You have been logged out.', 'info')
   return redirect(url_for('landing')) # Redirect to landing page after logout
# --- END UPDATED: Authentication Routes ---


# --- Database Initialization (Run once to create tables) ---
# This block ensures tables are created when the app starts.
# In production, you might use Flask-Migrate or a separate script.
with app.app_context():
   db.create_all()
   app.logger.info("Database tables created/checked.")

   # NEW: Create an admin user if one doesn't exist for easy testing
   if not User.query.filter_by(username='admin').first():
       admin_user = User(username='admin', is_admin=True)
       admin_user.set_password('adminpass') # Set a default password for the admin
       db.session.add(admin_user)
       db.session.commit()
       app.logger.info("Default admin user 'admin' created with password 'adminpass'.")


# --- Main App Run ---
if __name__ == '__main__':
   # Important: For async Flask routes, you should use an ASGI server in production.
   # For local development with auto-reloading, Hypercorn is a good choice.
   # To run with Hypercorn:
   # 1. Install it: pip install hypercorn
   # 2. Run: hypercorn app:app --reload
   # If you must use app.run() for quick tests and encounter the 'event loop closed' error,
   # you can use `nest_asyncio.apply()` (install with `pip install nest-asyncio`), but this is
   # generally not recommended for production as it can hide underlying architectural issues.
   app.run(debug=True)
