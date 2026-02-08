# REMOVE THESE TWO LINES IF THEY ARE PRESENT AT THE TOP OF YOUR FILE
# import nest_asyncio
# nest_asyncio.apply()

import json
import asyncio
import os
import time
import io
import wave
import re
import threading
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash, session, send_file
import logging
from datetime import datetime, timedelta # Import timedelta for time calculations
import re # Import for regular expressions
from functools import wraps # Import wraps for decorators
import base64 # Import base64 for image processing
import uuid # For generating unique reset tokens
import random # NEW: For generating random username suggestions
import string # NEW: For string manipulation in username generation
from google import genai as gemma_genai
from google.genai import types as gemma_types   # Required for GenerateContentConfig
from zai import ZaiClient as ZhipuAI
# 1. Use absolute import
import secrets
from datetime import datetime
# Assuming News and ApiRequestLog models are imported

# --- NEW IMPORTS FOR AUTHENTICATION ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import text
# --- END NEW IMPORTS ---


# NEW: Import for specific Gemini API exceptions
from google.api_core import exceptions as google_api_exceptions


# NEW: Flask-Mail imports for email sending
from flask_mail import Mail, Message

# NEW: Flask-Sock for Live API
from flask_sock import Sock




app = Flask(__name__)


# --- NEW: Flask-SQLAlchemy Configuration ---
# Configure SQLite database. This file will be created in your project directory.
# On Render, this database file will be ephemeral unless you attach a persistent disk.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////var/data/site.db' # 'sqlite:////var/data/site.db' #'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress a warning
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_that_should_be_in_env') # Needed for Flask-Login sessions
db = SQLAlchemy(app)
# --- END NEW: Flask-SQLAlchemy Configuration ---


# --- NEW: Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # The view Flask-Login should redirect to for login
# --- END NEW: Flask-Login Configuration ---


# --- NEW: Flask-Mail Configuration ---
app.config['MAIL_SERVER'] = 'smtp.hostinger.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'info@promptsgenerator.ai'
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD') # IMPORTANT: Set this environment variable!
mail = Mail(app)
# --- END NEW: Flask-Mail Configuration ---




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
ZAIPU_API_CONFIGURED = False
GEMINI_API_KEY = None
gemma_client = None  # Start as 
# Updated Configuration
zai_client = None

def configure_ai_apis():
    global GEMINI_API_CONFIGURED, gemma_client, zai_client
    # Existing Gemini Keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    # New Z.ai Key
    zai_api_key = os.getenv("ZAI_API_KEY")

    if gemini_key:
        try:
            # 1. Configure the Gemini (Legacy) SDK
            genai.configure(api_key=gemini_key)
            
            # 2. Configure the Gemma (New) SDK Client
            # Use the verified api_key variable here
            gemma_client = gemma_genai.Client(api_key=gemini_key)
            
            GEMINI_API_CONFIGURED = True
            app.logger.info("Both Gemini and Gemma APIs configured successfully.")
        except Exception as e:
            app.logger.error(f"ERROR: AI Configuration failed: {e}")
            GEMINI_API_CONFIGURED = False
      
    elif zai_api_key:
        try:
            # Initialize GLM client alongside Gemini
            zai_client = ZhipuAI(api_key=zai_api_key)
            app.logger.info(" glm-flash-4.7 configured successfully.")
        except Exception as e:
            app.logger.error(f"GLM Config Error: {e}")
            ZAIPU_API_CONFIGURED = False
    
    else:
        app.logger.warning("API_KEY not found. AI features disabled.")

# Call the consolidated function
configure_ai_apis()

# Model Usage Tracker
class ModelUsageTracker:
    def __init__(self):
        self.counts = {
            'gemma-3-1b-it': 0,
            'gemma-3-4b-it': 0,
            'gemma-3-12b-it': 0,
            'gemma-3-27b-it': 0
        }
        self.limit = 12000
        self.last_reset = datetime.now().date()
        self.lock = threading.Lock()

    def _check_reset(self):
        """Resets counters if a new day has started."""
        now = datetime.now().date()
        if now > self.last_reset:
            with self.lock:
                for model in self.counts:
                    self.counts[model] = 0
                self.last_reset = now

    def get_and_increment(self, preferred_model):
        self._check_reset()
        with self.lock:
            # Check if preferred model is under limit
            if self.counts[preferred_model] < self.limit:
                self.counts[preferred_model] += 1
                return preferred_model
            
            # Fallback Logic: If preferred is full, try the next smallest model
            fallbacks = ['gemma-3-12b-it', 'gemma-3-4b-it', 'gemma-3-1b-it']
            for model in fallbacks:
                if self.counts[model] < self.limit:
                    self.counts[model] += 1
                    return model
            
            return None # All quotas exhausted

# Initialize the global tracker
usage_tracker = ModelUsageTracker()

def get_dynamic_model_name(prompt_instruction: str) -> str:
    length = len(prompt_instruction)

    # 1. Determine "ideal" model based on length
    if length > 7500:
        preferred >= 'glm-4.7-flash'
    elif length > 5400:
        preferred = 'gemma-3-27b-it'
    elif length >= 2700:
        preferred = 'gemma-3-12b-it'
    elif length >= 1800:
        preferred = 'gemma-3-4b-it'
    else:
        preferred = 'gemma-3-1b-it'

    # 2. Check quota and get final model name
    final_model = usage_tracker.get_and_increment(preferred)
    
    if not final_model:
        raise Exception("Daily quota exceeded for all models.")
        
    return final_model


# --- UPDATED: User Model for SQLAlchemy and Flask-Login ---
class User(db.Model, UserMixin):
  id = db.Column(db.Integer, primary_key=True)
  username = db.Column(db.String(80), unique=True, nullable=False)
  password_hash = db.Column(db.String(128), nullable=False)
  is_admin = db.Column(db.Boolean, default=False)
  # --- ADD THESE NEW FIELDS ---
  api_key = db.Column(db.String(100), unique=True, nullable=True) # For API Authentication
  is_locked = db.Column(db.Boolean, default=False) # To disable API access if needed
  # ----------------------------
  last_prompt_request = db.Column(db.DateTime, nullable=True) # For cooldown
  daily_prompt_count = db.Column(db.Integer, default=0, nullable=False) # NEW: Daily prompt count
  last_count_reset_date = db.Column(db.Date, nullable=True) # NEW: Date when count was last reset
  # NEW: Fields for password reset
  password_reset_token = db.Column(db.String(36), nullable=True) # UUID is 36 chars
  password_reset_expiration = db.Column(db.DateTime, nullable=True)
  email = db.Column(db.String(120), unique=True, nullable=True) # NEW: Email field for password reset




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

# --- ADD THIS NOW ---
class ApiRequestLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    endpoint = db.Column(db.String(255), nullable=False)
    method = db.Column(db.String(10), nullable=False)
    status_code = db.Column(db.Integer)
    response_time = db.Column(db.Float) 
    request_timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('api_logs', lazy=True))

# Initialize variable
blog_id_tracker = []

with app.app_context():
    try:
        # 1. Create tables that don't exist yet (like ApiRequestLog)
        db.create_all()

        # 2. Patch existing 'user' table
        from sqlalchemy import text, inspect
        inspector = inspect(db.engine)
        existing_columns = [col['name'] for col in inspector.get_columns('user')]
        
        with db.engine.connect() as conn:
            # FIX FOR API_KEY (Split into two steps for SQLite)
            if 'api_key' not in existing_columns:
                app.logger.info("Patching: Adding api_key to user table.")
                # Step A: Add the column as a plain string
                conn.execute(text('ALTER TABLE "user" ADD COLUMN api_key VARCHAR(100)'))
                # Step B: Create a Unique Index (This is how SQLite handles Uniqueness on existing tables)
                conn.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS idx_user_api_key ON "user" (api_key)'))
            
            # FIX FOR IS_LOCKED
            if 'is_locked' not in existing_columns:
                app.logger.info("Patching: Adding is_locked to user table.")
                conn.execute(text('ALTER TABLE "user" ADD COLUMN is_locked BOOLEAN DEFAULT FALSE'))
            
            conn.commit()
            app.logger.info("Database patching completed.")

        # 3. Load the tracker (This will work now because the columns exist!)
        internal_blogs = News.query.filter(News.url.ilike('%blog_content%')).all()
        blog_id_tracker = [b.id for b in internal_blogs]
        app.logger.info(f"Startup: Loaded {len(blog_id_tracker)} blogs into tracker.")

    except Exception as e:
        # We use a broad exception here to ensure the app still attempts to start
        app.logger.error(f"Startup failed: {e}")

# RUNS ON EVERY REQUEST: Keeps the list fresh if a new blog is added or deleted
@app.before_request
def sync_tracker_from_disk():
    global blog_id_tracker
    try:
        # Match the startup logic exactly
        blogs = News.query.filter(News.url.ilike('%blog_content%')).all()
        blog_id_tracker = [b.id for b in blogs]
    except Exception as e:
        # If DB is locked or busy, we just keep the previous list
        pass


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

def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=1024):
    """
    Consolidated function that replaces ask_gemini_for_prompt.
    It routes to the correct SDK based on the model name.
    """
    try:
        # This now handles both length-based tiering AND the 12k RPD check
        selected_model = get_dynamic_model_name(prompt_instruction)

        # --- Route 1: GLM Model (ZhipuAI SDK) ---
        if "glm" in selected_model:
            response = zai_client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt_instruction}],
                max_tokens=max_output_tokens
            )
            return response.choices[0].message.content.strip()

      
        else:
            response = gemma_client.models.generate_content(
                model=selected_model,
                contents=prompt_instruction,
                config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": 0.1
                }
            )
        return filter_gemini_response(response.text).strip()
      
    except Exception as e:
        app.logger.error(f"Gemma Routing Error: {e}")
        return "SuperPrompter AI Service temporarily unavailable due to high demand."

# --- NEW: Gemini API for Image Understanding ---
def ask_gemini_for_image_text(image_data_bytes):
  if not GEMINI_API_CONFIGURED:
      return "Gemini API Key is not configured or the AI model failed to initialize."


  try:
      gemini_model_instance = genai.GenerativeModel('gemini-2.5-flash')
    
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
      return filter_gemini_response(f"Error communicating with SuperPrompter AI, please try after sometime.")
  except Exception as e: # Catch any other unexpected errors
      app.logger.error(f"Unexpected Error calling Gemini API for image text extraction: {e}", exc_info=True)
      return filter_gemini_response(f"Error communicating with SuperPrompter AI, please try after sometime.")
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


# In app.py
from datetime import datetime, timedelta, timezone # Import necessary time libraries

@app.route('/')
def landing():
    # Define "recent" as the last 24 hours (adjust timedelta as needed)
    # Use timezone-aware comparison if your database timestamps are timezone-aware
    RECENT_THRESHOLD = datetime.now(timezone.utc) - timedelta(hours=24)
    has_new_news = False
    has_new_jobs = False

    # 1. Fetch latest 6 news items
    raw_news_items = News.query.filter(~News.description.like('[AI_APP]%'),~News.description.like('[APP_LOG]%'),~News.description.like('[PROMPT]%'),~News.description.like('[AI_HUB]%')).order_by(News.timestamp.desc()).limit(6).all()
    
    # Process news items to clean the description for the summary view
    cleaned_news_items = []
    SUMMARY_LENGTH = 150
    
    for item in raw_news_items:
        
        # --- FRESHNESS CHECK FOR NEWS ---
        if item.timestamp.replace(tzinfo=timezone.utc) > RECENT_THRESHOLD:
             has_new_news = True
        # -------------------------------
        
        # Check if the item is a blog post (i.e., requires HTML cleaning)
        if item.id in blog_id_tracker:
            clean_text = strip_html_tags(item.description) if item.description else "No content provided."
        else:
            # If it's a regular news item, assume the content is already plain text
            clean_text = item.description if item.description else "No content provided."

        # Truncate the clean text to create the final summary snippet
        summary_snippet = clean_text[:SUMMARY_LENGTH] + ('...' if len(clean_text) > SUMMARY_LENGTH else '')
        
        # Create a dictionary to pass the cleaned data to the template
        cleaned_news_items.append({
            "id": item.id,
            "title": item.title,
            "summary_snippet": summary_snippet,
            "url": item.url,
            "timestamp": item.timestamp,
            "description": item.description 
        })
    
    # 2. Fetch latest 6 job listings
    job_listings = Job.query.order_by(Job.timestamp.desc()).limit(6).all()
    
    for job in job_listings:
        # --- FRESHNESS CHECK FOR JOBS ---
        if job.timestamp.replace(tzinfo=timezone.utc) > RECENT_THRESHOLD:
            has_new_jobs = True
        # -------------------------------

    # --- NEW LOGIC: Determine the Single Strip Status ---
    if not raw_news_items and not job_listings:
        # Case 1: Database is totally empty (most critical warning)
        status_message = "The site is currently empty. No items to display in any section."
        status_class = "alert-danger"
    elif has_new_news and has_new_jobs:
        # Case 2: Both have recent additions
        status_message = "New News and Job listings added recently!"
        status_class = "alert-success"
    elif has_new_news:
        # Case 3: Only news has recent additions
        status_message = "New News items added recently. Displaying latest available jobs."
        status_class = "alert-success"
    elif has_new_jobs:
        # Case 4: Only jobs has recent additions
        status_message = "New Job listings added recently. Displaying latest available news."
        status_class = "alert-success"
    else:
        # Case 5: Content exists, but none of the 6 displayed items are less than 24 hours old
        status_message = "Displaying latest available items. No new additions in the last 24 hours."
        status_class = "alert-info"
    # --- END NEW LOGIC ---
    
    return render_template('landing.html', 
                            news_items=cleaned_news_items,
                            job_listings=job_listings, 
                            current_user=current_user,
                            blog_id_tracker=blog_id_tracker,
                            # Pass the message and styling class for the single strip
                            status_message=status_message,
                            status_class=status_class)


# Renamed original index route to /app_home
@app.route('/app_home')
def app_home():
  # Pass current_user object to the template to show login/logout status
  return render_template('index.html', current_user=current_user)


# NEW: LLM Benchmark Page Route
@app.route('/llm_benchmark')
def llm_benchmark():
   return render_template('llm_benchmark.html', current_user=current_user)

def log_app_event(user_id, message, category="info"):
    """
    Systematic logging using the News database model.
    Format: [APP_LOG][category][HH:MM:SS] Message
    """
    from datetime import datetime
    # Get current time for the log entry
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    
    # Construct the tagged description based on your AI App methodology
    # Example: [APP_LOG][success][14:30:05] Prompts generated successfully.
    tagged_description = f"[APP_LOG][{category}][{timestamp}] {message}"
    
    # Create the entry in the News table
    new_log = News(
        title="System Log Entry",
        url="#",
        description=tagged_description,
        user_id=user_id
    )
    
    try:
        db.session.add(new_log)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to save system log: {e}")

@app.route('/get_app_logs', methods=['GET'])
@login_required
def get_app_logs():
    """
    Retrieves all persistent logs for the current user from the News table.
    Parses the [APP_LOG][category][timestamp] format.
    """
    # Fetch all entries for this user tagged as app logs
    log_entries = News.query.filter(
        News.user_id == current_user.id,
        News.description.like('[APP_LOG]%')
    ).order_by(News.id.asc()).all()

    parsed_logs = []
    for entry in log_entries:
        try:
            # Format is: [APP_LOG][category][timestamp] message
            # We split by ']' to isolate the parts
            parts = entry.description.split(']', 3)
            if len(parts) >= 4:
                category = parts[1].replace('[', '')
                timestamp = parts[2].replace('[', '')
                message = parts[3].strip()
                
                parsed_logs.append({
                    'category': category,
                    'timestamp': timestamp,
                    'message': message
                })
        except Exception as e:
            app.logger.error(f"Error parsing log entry {entry.id}: {e}")
            continue

    return jsonify(parsed_logs)

def clean_prompt_text(text):
    """
    Removes Markdown formatting and excessive special characters 
    to ensure a clean, 'non-messy' output.
    """
    if not text:
        return ""
    
    # 1. Remove Markdown code blocks (```python ... ```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # 2. Remove bold/italic markers (* or _)
    text = text.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
    
    # 3. Remove header markers (#)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # 4. Remove backticks (`)
    text = text.replace('`', '')

    # 5. Clean up extra newlines and spaces
    text = re.sub(r'\n\s*\n', '\n', text) # Remove empty lines
    return text.strip()

@app.route('/generate', methods=['POST'])
@login_required
async def generate_prompts_endpoint():
    user = current_user
    now = datetime.utcnow()

    # --- Cooldown Check ---
    if user.last_prompt_request:
        time_since_last_request = (now - user.last_prompt_request).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            # LOG THE COOLDOWN
            log_app_event(user.id, f"Generation blocked: Please wait {remaining_time}s.", "warning")
            return jsonify({
                "error": f"Please wait {remaining_time} seconds.",
                "cooldown_active": True,
                "remaining_time": remaining_time
            }), 429

    # --- Daily Limit Check ---
    if not user.is_admin:
        today = now.date()
        if user.last_count_reset_date != today:
            user.daily_prompt_count = 0
            user.last_count_reset_date = today
            db.session.commit()

        if user.daily_prompt_count >= 2:
            # LOG THE LIMIT REACHED
            log_app_event(user.id, "Daily limit of 2 generations reached.", "danger")
            return jsonify({"error": "Daily limit reached.", "daily_limit_reached": True}), 429

    raw_input = request.form.get('prompt_input', '').strip()
    language_code = request.form.get('language_code', 'en-US')

    if not raw_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        
        raw_results = await generate_prompts_async(raw_input, language_code)
        results = {
            "polished": clean_prompt_text(raw_results.get("polished", "")),
            "creative": clean_prompt_text(raw_results.get("creative", "")),
            "technical": clean_prompt_text(raw_results.get("technical", ""))
        }

        # Update stats
        user.last_prompt_request = now
        if not user.is_admin:
            user.daily_prompt_count += 1
        
        # Save Raw Request (Your existing logic)
        new_raw_prompt = RawPrompt(user_id=user.id, raw_text=raw_input)
        db.session.add(new_raw_prompt)
        
        # LOG THE SUCCESS
        log_app_event(user.id, f"Success: Generated prompts for '{raw_input[:20]}...'", "success")
        
        db.session.commit()
        return jsonify(results)

    except Exception as e:
        db.session.rollback()
        # LOG THE SYSTEM ERROR
        log_app_event(user.id, f"System Error: {str(e)[:50]}", "danger")
        return jsonify({"error": str(e)}), 500

@app.route('/reverse_prompt', methods=['POST'])
@login_required
async def reverse_prompt_endpoint():
    user = current_user
    now = datetime.utcnow()

    # --- Cooldown Check ---
    if user.last_prompt_request:
        time_since_last_request = (now - user.last_prompt_request).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            # LOG THE COOLDOWN BLOCK
            log_app_event(user.id, f"Reverse Prompt blocked: Wait {remaining_time}s.", "warning")
            return jsonify({
                "error": f"Please wait {remaining_time} seconds.",
                "cooldown_active": True,
                "remaining_time": remaining_time
            }), 429

    # --- Daily Limit Check ---
    if not user.is_admin:
        if user.daily_prompt_count >= 1: # Assuming 1 for Reverse Prompt
            # LOG THE LIMIT BLOCK
            log_app_event(user.id, "Daily limit for Reverse Prompt reached.", "danger")
            return jsonify({"error": "Daily limit reached.", "daily_limit_reached": True}), 429

    data = request.get_json()
    input_text = data.get('input_text', '').strip()
    language_code = data.get('language_code', 'en-US')

    try:
        inferred_prompt = await generate_reverse_prompt_async(input_text, language_code)

        # Update stats
        user.last_prompt_request = now
        if not user.is_admin:
            user.daily_prompt_count += 1
        
        # LOG THE SUCCESS
        log_app_event(user.id, f"Success: Reverse prompt inferred for '{input_text[:20]}...'", "success")
        
        db.session.commit()
        return jsonify({"inferred_prompt": inferred_prompt})

    except Exception as e:
        db.session.rollback()
        # LOG THE ERROR
        log_app_event(user.id, f"Reverse Prompt Error: {str(e)[:50]}", "danger")
        return jsonify({"error": str(e)}), 500



@app.route('/admin/users')
@admin_required
def admin_users():
    """
    Renders the admin dashboard to manage users and their API keys.
    """
    users = User.query.all()
    users_data = []
    
    for user in users:
        users_data.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'api_key': user.api_key, # New field from supporting code
            'is_admin': user.is_admin,
            'is_locked': getattr(user, 'is_locked', False), # Safety check for field
            'daily_limit': getattr(user, 'daily_limit', 0)
        })

    return render_template('admin_users.html', users=users_data, current_user=current_user)


def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time() # Start the timer
        
        # 1. Get the key from the header
        api_key = request.headers.get('X-API-KEY')
        
        if not api_key:
            return jsonify({"error": "API Key is missing"}), 401
        
        # 2. Authenticate User
        user = User.query.filter_by(api_key=api_key).first()
        
        if not user or user.is_locked:
            status = 403 if user and user.is_locked else 401
            return jsonify({"error": "Unauthorized or Locked"}), status

        # 3. Execute the actual route function
        response = f(user, *args, **kwargs)

        # 4. LOGGING LOGIC: Save the request details
        try:
            duration = time.time() - start_time
            # Determine status code (handles both tuple and response object)
            status_code = response[1] if isinstance(response, tuple) else response.status_code
            
            new_log = ApiRequestLog(
                user_id=user.id,
                endpoint=request.path,
                method=request.method,
                status_code=status_code,
                response_time=round(duration, 4)
            )
            db.session.add(new_log)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Failed to log API request: {e}")

        return response
    return decorated_function

@app.route('/admin/users/generate_api_key/<int:user_id>', methods=['POST'])
@admin_required
def generate_api_key(user_id):
    """
    Generates a secure random API key for a specific user.
    """
    user = User.query.get_or_404(user_id)
    
    # Generate a cryptographically secure random 64-character hex string
    # 'sp_live_' prefix helps identify your app's keys in logs/code
    random_key = f"sp_live_{secrets.token_hex(32)}"
    
    try:
        user.api_key = random_key
        db.session.commit()
        flash(f"New API key generated for {user.username}.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error generating API key: {str(e)}", "danger")
        
    return redirect(url_for('admin_users'))

@app.route('/admin/users/toggle_access/<int:user_id>', methods=['POST'])
@admin_required
def toggle_user_access(user_id):
    """
    Locks or unlocks a user account to control API access.
    """
    user = User.query.get_or_404(user_id)
    
    # Toggle the boolean status
    user.is_locked = not user.is_locked
    
    status = "locked" if user.is_locked else "unlocked"
    db.session.commit()
    flash(f"User {user.username} has been {status}.", "info")
    
    return redirect(url_for('admin_users'))

@app.route('/api/v1/news/search', methods=['GET'])
@api_key_required
def api_search_news(user):
    # 1. Get query parameters
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    # 2. Perform the search
    results = News.query.filter(News.title.ilike(f'%{query}%')).limit(limit).all()
  
    
    # 3. Return the JSON response
    # The DECORATOR will now take over, calculate the time, 
    # and save the log to the database automatically.
    return jsonify({
        "results": [
            {"title": n.title, "url": n.url, "description": n.description} 
            for n in results
        ]
    })

@app.route('/admin/api-performance')
@admin_required
def admin_api_performance():
    """
    Renders the API Performance Dashboard with the latest request logs.
    """
    # Fetch the 100 most recent logs to keep the dashboard snappy
    logs = ApiRequestLog.query.order_by(ApiRequestLog.request_timestamp.desc()).limit(100).all()
    
    # Create a lookup map for usernames to avoid multiple DB queries in the template
    users = {user.id: user for user in User.query.all()}
    
    return render_template('admin_api_performance.html', api_logs=logs, users=users)
      

@app.route('/process_image_prompt', methods=['POST'])
@login_required
async def process_image_prompt_endpoint():
    user = current_user
    now = datetime.utcnow()

    # --- Cooldown Check ---
    if user.last_prompt_request:
        time_since_last_request = (now - user.last_prompt_request).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            # LOG THE COOLDOWN
            log_app_event(user.id, f"Image processing blocked: Wait {remaining_time}s.", "warning")
            return jsonify({"error": "Cooldown", "remaining_time": remaining_time}), 429

    # --- Daily Limit Check ---
    if not user.is_admin and user.daily_prompt_count >= 2:
        # LOG THE LIMIT
        log_app_event(user.id, "Image processing blocked: Daily limit reached.", "danger")
        return jsonify({"error": "Daily limit reached"}), 429

    data = request.get_json()
    image_data_b64 = data.get('image_data')

    if not image_data_b64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode base64 image data
        image_data_bytes = base64.b64decode(image_data_b64)
        
        # Process image with Gemini (assumed utility function)
        recognized_text = await asyncio.to_thread(ask_gemini_for_image_text, image_data_bytes)

        # Update stats
        user.last_prompt_request = now
        if not user.is_admin:
            user.daily_prompt_count += 1
        
        # LOG THE SUCCESS
        log_app_event(user.id, "Image processed successfully. Text extracted.", "success")
        
        db.session.commit()
        return jsonify({"recognized_text": recognized_text})

    except Exception as e:
        db.session.rollback()
        # LOG THE SYSTEM ERROR
        log_app_event(user.id, f"Image Error: {str(e)[:50]}", "danger")
        return jsonify({"error": str(e)}), 500



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
      
       if daily_count >= 2:
           daily_limit_reached = True


   return jsonify({
       "cooldown_active": cooldown_active,
       "remaining_time": remaining_time,
       "daily_limit_reached": daily_limit_reached,
       "daily_count": daily_count,
       "is_admin": user.is_admin
   }), 200
# --- END UPDATED ---

def strip_html_tags(html_content):
    """Removes HTML tags and entities from a string to create a clean text summary."""
    if html_content:
        # 1. Replace common HTML entities (&nbsp;, &lt;, etc.) with spaces
        content = html_content.replace('&nbsp;', ' ')
        # You may need more robust entity handling depending on your needs, 
        # but this simple replace often works for RTE output.
        
        # 2. Use regex to strip all HTML tags (<...>)
        clean = re.compile('<.*?>')
        text_without_tags = re.sub(clean, '', content)
        
        return text_without_tags.strip() # Remove leading/trailing whitespace
    return ""

# In app.py

@app.route('/news_item/<int:item_id>', methods=['GET'])
def view_news_item(item_id):
    """
    Finds the News item by ID and redirects the user immediately 
    to the original external source URL (News.url).
    """
    news_item = News.query.get_or_404(item_id)
    
    # 1. Check if it's a blog post; if so, redirect to the internal blog route
    # (This step is good defensive coding to prevent accidentally redirecting internal blogs)
    global blog_id_tracker
    if news_item.id in blog_id_tracker:
        blog_uuid = news_item.url.split('/')[-1]
        return redirect(url_for('view_blog_content', blog_uuid=blog_uuid))
        
    # 2. PERFORM THE REDIRECT
    # This line is the key change. Flask's redirect() function tells the browser 
    # to load the URL stored in news_item.url (e.g., the CNN link).
    return redirect(news_item.url) 
    # By default, this uses a 302 Found status code. You can optionally 
    # use redirect(news_item.url, code=301) for a permanent redirect if needed.

# In app.py, add a constant near the top:
ITEMS_PER_PAGE = 10
##query = News.query.order_by(News.timestamp.desc()).limit(6).all()

@app.route('/all_news', methods=['GET'])
def all_news():
    search_query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)

    # 1. THE BASE QUERY OBJECT (No .all(), No .limit() here!)
    # We must keep this as a 'BaseQuery' object for pagination to work
    news_query_object = News.query.filter(
        News.description.isnot(None), 
        ~News.description.like('[AI_APP]%'),
        ~News.description.like('[APP_LOG]%'),
        ~News.description.like('[PROMPT]%'),
        ~News.description.like('[AI_HUB]%')
    ).order_by(News.timestamp.desc())

    # 2. ADD SEARCH FILTERS (If search_query exists)
    if search_query:
        search_term = f"%{search_query}%"
        news_query_object = news_query_object.filter(
            (News.title.ilike(search_term)) | (News.description.ilike(search_term))
        )

    # 3. PAGINATE THE QUERY OBJECT
    # This is where the error was happening. 'news_query_object' MUST NOT be a list.
    pagination = news_query_object.paginate(page=page, per_page=ITEMS_PER_PAGE, error_out=False)
    
    # 4. GET THE ACTUAL ITEMS FROM PAGINATION
    articles_from_db = pagination.items

    formatted_articles = []
    SUMMARY_LENGTH = 200

    for article in articles_from_db:
        # Generate the Internal Shareable Link
        if article.id in blog_id_tracker:
             blog_uuid = article.url.split('/')[-1]
             share_link = url_for('view_blog_content', blog_uuid=blog_uuid, _external=True)
        else:
             share_link = url_for('view_news_item', item_id=article.id, _external=True)

        # 1. Get raw description
        raw_desc = article.description if article.description else ""

        # 2. STRIP TAGS HERE IN PYTHON (using your existing strip_html_tags function)
        # This ensures no <p> or <span> tags make it into the 'summary' field
        clean_description = strip_html_tags(raw_desc)

        summary_snippet = clean_description[:SUMMARY_LENGTH] + ('...' if len(clean_description) > SUMMARY_LENGTH else '')

        formatted_articles.append({
            "id": article.id,
            "title": article.title,
            "summary": summary_snippet,
            "source_url": article.url,
            "share_url": share_link,
            "date_published": article.published_date if article.published_date else article.timestamp
        })
    
    return render_template('all_news.html',
                            articles=formatted_articles,
                            pagination=pagination,
                            search_query=search_query,
                            blog_id_tracker=blog_id_tracker,
                            current_user=current_user)
  

# --- NEW: Autocomplete API for News Titles ---
@app.route('/api/news_autocomplete', methods=['GET'])
def news_autocomplete():
    """Returns a JSON list of news titles matching the query."""
    query_term = request.args.get('q', '').strip()
    if not query_term:
        return jsonify([])

    # Use ilike for case-insensitive partial matching
    search_pattern = f"%{query_term}%"
    
    # Fetch top 10 matching titles
    results = News.query.filter(News.title.ilike(search_pattern)) \
                        .limit(10) \
                        .all()
    
    # Format results as a list of strings (titles)
    titles = [item.title for item in results]
    
    return jsonify(titles)
# --- END NEW: Autocomplete API for News Titles ---

# --- UPDATED: All Jobs Public Page Route with Search and Pagination ---
@app.route('/all_jobs', methods=['GET'])
@login_required
def all_jobs():
    search_query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int) # Get current page number, defaults to 1

    query = Job.query.order_by(Job.timestamp.desc())

    if search_query:
        search_term = f"%{search_query}%"
        query = query.filter(
            (Job.title.ilike(search_term)) |
            (Job.company.ilike(search_term)) |
            (Job.location.ilike(search_term)) |
            (Job.description.ilike(search_term))
        )

    # Apply pagination
    pagination = query.paginate(page=page, per_page=ITEMS_PER_PAGE, error_out=False)
    job_listings = pagination.items

    # Format the job listings for the template
    formatted_jobs = [{
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "description_summary": job.description if job.description else "No description provided.",
        "job_url": job.url,
        "date_posted": job.published_date if job.published_date else job.timestamp
    } for job in job_listings]

    return render_template('all_jobs.html', 
                           jobs=formatted_jobs, 
                           pagination=pagination, # Pass pagination object to template
                           search_query=search_query, # Pass search query to preserve it in pagination links
                           current_user=current_user)
# --- END UPDATED: All Jobs Public Page Route ---


# --- UPDATED: Admin News Management Routes ---
@app.route('/admin/news', methods=['GET'])
@login_required
@admin_required
def admin_news():
   news_items = News.query.order_by(News.timestamp.desc()).all()
   return render_template('admin_news.html', news_items=news_items, current_user=current_user)


# In app.py

@app.route('/admin/news/add', methods=['POST'])
@login_required
@admin_required
def add_news():
    title = request.form.get('title')
    url = request.form.get('url')
    description = request.form.get('description')
    published_date_str = request.form.get('published_date')

    if not title or not url:
        flash('Title and URL are required to add news.', 'danger')
        return redirect(url_for('admin_news'))

    published_date = None
    if published_date_str:
        try:
            published_date = datetime.strptime(published_date_str, '%Y-%m-%d')
        except ValueError:
            flash('Invalid Published Date format. Please use YYYY-MM-DD.', 'danger')
            return redirect(url_for('admin_news'))

    try:
        new_news = News(
            title=title, 
            url=url, 
            description=description, 
            published_date=published_date, 
            user_id=current_user.id
        )
        db.session.add(new_news)
        db.session.commit()
        
        # --- ENHANCEMENT (Optional): Explicitly confirm this is NOT a blog ---
        # Since this route is for external news, we ensure its ID is NOT in the tracker.
        # This is primarily defensive programming, as the blog route should handle adding.
        global blog_id_tracker
        if new_news.id in blog_id_tracker:
             # If somehow the ID got added (which shouldn't happen here), remove it.
             blog_id_tracker.remove(new_news.id)
             
        # --- END ENHANCEMENT ---
        
        flash('News item added successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding news item: {e}', 'danger')
        
    return redirect(url_for('admin_news'))

# In app.py

@app.route('/admin/news/delete/<int:news_id>', methods=['POST'])
@login_required
@admin_required
def delete_news(news_id):
    news_item = News.query.get_or_404(news_id)
    
    # 1. Check the type of item BEFORE deleting it (we need its ID)
    global blog_id_tracker
    was_blog = news_item.id in blog_id_tracker
    
    try:
        # Perform the database deletion
        db.session.delete(news_item)
        db.session.commit()
        
        # 2. Update the tracker if it was a blog
        if was_blog:
            blog_id_tracker.remove(news_item.id)
            flash(f'Blog Post "{news_item.title}" deleted successfully!', 'success')
        else:
            flash('News item deleted successfully!', 'success')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting item: {e}', 'danger')
        
        # Fallback redirect: If the deletion failed, return to the default news page
        return redirect(url_for('admin_news'))
        
    # 3. CRITICAL REDIRECT LOGIC: Redirect based on the item type
    if was_blog:
        # If it was a blog, redirect back to the blog management page
        return redirect(url_for('admin_blogs')) # <-- FIX 1: Redirect to admin_blogs
    else:
        # If it was a regular news item, redirect back to the news management page
        return redirect(url_for('admin_news')) # <-- FIX 2: Redirect to admin_news


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

# In app.py

@app.route('/admin/blogs', methods=['GET'])
@login_required
@admin_required
def admin_blogs():
    # 1. Get current page number from the URL query string (default is 1)
    page = request.args.get('page', 1, type=int)
    
    # 2. Use .paginate() instead of .all()
    # Parameters: page=page number, per_page=number of results per page
    pagination = News.query.order_by(News.timestamp.desc()).paginate(
        page=page, 
        per_page=10, 
        error_out=False
    )
    
    # 3. Extract the items for the current page
    blog_items = pagination.items
    
    return render_template('admin_blogs.html', 
                           blog_items=blog_items,
                           pagination=pagination, # Pass the whole object for UI controls
                           blog_id_tracker=blog_id_tracker,
                           current_user=current_user)


@app.route('/admin/blogs/add', methods=['POST'])
@login_required
@admin_required
def add_blog_post():
    title = request.form.get('title')
    description = request.form.get('description')
    published_date_str = request.form.get('published_date')
    
    # Create a unique, permanent link for this content
    blog_url = f"/blog_content/{uuid.uuid4()}" 
    
    if not title or not description:
        flash('Title and Blog Content are required.', 'danger')
        return redirect(url_for('admin_blogs'))

    published_date = None
    if published_date_str:
        try:
            published_date = datetime.strptime(published_date_str, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format. Use YYYY-MM-DD.', 'danger')
            return redirect(url_for('admin_blogs'))

    try:
        new_blog = News(
            title=title, 
            url=blog_url, 
            description=description, 
            published_date=published_date, 
            user_id=current_user.id
        ) 
        db.session.add(new_blog)
        # CRITICAL: Saves to /var/data/site.db permanently
        db.session.commit()
        
        flash('Blog Post saved permanently to disk!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding blog post: {e}', 'danger')
        
    return redirect(url_for('admin_blogs'))

@app.route('/admin/blogs/edit/<int:post_id>', methods=['POST'])
@login_required
@admin_required
def edit_blog_post(post_id):
    # Fetch the existing blog from the disk
    blog_to_edit = News.query.get_or_404(post_id)
    
    title = request.form.get('title')
    description = request.form.get('description')
    published_date_str = request.form.get('published_date')

    if not title or not description:
        flash('Title and Content cannot be empty.', 'danger')
        return redirect(url_for('admin_blogs'))

    try:
        # Update the object fields
        blog_to_edit.title = title
        blog_to_edit.description = description
        
        if published_date_str:
            blog_to_edit.published_date = datetime.strptime(published_date_str, '%Y-%m-%d')

        # Save the changes permanently
        db.session.commit()
        flash('Blog Post updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating blog: {e}', 'danger')

    return redirect(url_for('admin_blogs'))

@app.route('/blog_content/<uuid:blog_uuid>')
def view_blog_content(blog_uuid):
    placeholder_url = f"/blog_content/{str(blog_uuid)}"
    blog_post = News.query.filter_by(url=placeholder_url).first_or_404()

    # --- 1. DYNAMIC AI KEYWORD GENERATOR (FIXED) ---
    # Combine title and description
    raw_content = (blog_post.title + " " + (blog_post.description or ""))
    
    # STEP A: Remove <style> blocks (this is where the #Dirltr gibberish comes from)
    clean_content = re.sub(r'<style.*?>.*?</style>', '', raw_content, flags=re.DOTALL | re.IGNORECASE)
    
    # STEP B: Remove all other HTML tags (like <div>, <p>, <span>)
    clean_content = re.sub(r'<[^>]+>', '', clean_content)
    
    # STEP C: Standard alphanumeric cleaning
    words = ''.join(e for e in clean_content if e.isalnum() or e.isspace()).split()
    
    # STEP D: Filtering logic
    stop_words = {'this', 'that', 'with', 'from', 'their', 'could', 'would', 'about', 'these'}
    css_junk = {'Margin', 'Padding', 'Height', 'Weight', 'Color', 'Style', 'Textalign'}
    
    unique_keywords = []
    for w in words:
        w_clean = w.capitalize()
        # Only take long words that aren't stop words or lingering CSS terms
        if len(w) > 5 and w.lower() not in stop_words and w_clean not in unique_keywords and w_clean not in css_junk:
            unique_keywords.append(w_clean)
            if len(unique_keywords) >= 5: break 
    # ----------------------------------------
    
    # --- 2. EXISTING SIDEBAR LOGIC ---
    raw_blogs = News.query.filter(News.url.contains('/blog_content/')).order_by(News.timestamp.desc()).limit(20).all()

    processed_sidebar = []
    for b in raw_blogs:
        uuid_part = b.url.split('/')[-1]
        processed_sidebar.append({
            'id': b.id,
            'title': b.title,
            'url': url_for('view_blog_content', blog_uuid=uuid_part)
        })

    # Pass both processed_sidebar AND unique_keywords to the template
    return render_template('single_blog_post.html', 
                           post=blog_post, 
                           latest_blogs=processed_sidebar, 
                           ai_tags=unique_keywords)


  

@app.route('/api/recommended/<int:post_id>')
def get_recommended(post_id):
    try:
        current_post = News.query.get_or_404(post_id)
        keywords = [w.lower() for w in current_post.title.split() if len(w) > 4]
        
        # Query matching IDs from your existing blog_id_tracker
        query = News.query.filter(News.id != post_id, News.id.in_(blog_id_tracker))

        if keywords:
            query = query.filter(db.or_(*[News.title.ilike(f'%{kw}%') for kw in keywords]))

        related = query.order_by(News.timestamp.desc()).limit(5).all()

        results = []
        for p in related:
            # We use Regex to pull the UUID out of your stored URL string
            # This looks for the part after /blog_content/
            match = re.search(r'blog_content/([a-f0-9\-]+)', p.url)
            uuid_to_use = match.group(1) if match else p.id 

            results.append({
                "title": p.title[:45] + "..." if len(p.title) > 45 else p.title,
                "url": url_for('view_blog_content', blog_uuid=uuid_to_use)
            })

        return jsonify(results)

    except Exception as e:
        print(f"Error in get_recommended: {e}")
        return jsonify([])


@app.route('/api/ai-summary', methods=['GET'])
def ai_summary():
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({"summary": "Please enter a search term."}), 400

    try:
        # 1. DATABASE SEARCH
        # Replace 'News' with whatever your class name is (e.g., Article, Post, etc.)
        # If your model is in another file, ensure you have: from models import News
        search_results = News.query.filter(
            (News.title.ilike(f'%{query}%')) | 
            (News.description.ilike(f'%{query}%'))
        ).limit(5).all()

        # 2. CONTEXT PREPARATION
        if search_results:
            db_context = "Relevant data from our database:\n" + "\n".join(
                [f"Title: {n.title} - Details: {n.description[:200]}" for n in search_results]
            )
        else:
            db_context = "No specific match found in the local database."

        # 3. AI INSTRUCTION
        ai_instruction = f"""
        Task: Summary of '{query}' based on recent: {db_context} and your training data.
        Rules: Max 70 words. Plain text only. NO special characters like * or #.
        """

        # 4. CALL YOUR EXISTING GEMINI FUNCTION
        summary_text = ask_gemini_for_prompt(ai_instruction, max_output_tokens=250)
        
        # Clean special characters just in case
        clean_summary = summary_text.replace('*', '').replace('#', '').replace('_', '')
        
        return jsonify({"summary": clean_summary})

    except Exception as e:
        app.logger.error(f"AI Summary Error: {e}")
        return jsonify({"summary": "AI is taking a moment to breathe. Please try again."}), 500

# 1. ANALYZER: Summarizes the blog post
@app.route('/api/analyze-post/<int:post_id>')
def analyze_post(post_id):
    post = News.query.get_or_404(post_id)
    prompt = f"Summarize this article in 3 bullet points of key takeaways: {post.description[:800]}"
    summary = ask_gemini_for_prompt(prompt)
    return jsonify({"summary": summary})

# 2. POLL GENERATOR: Using Gemini 3 Flash
@app.route('/api/generate-blog-poll/<int:post_id>')
def generate_blog_poll(post_id):
    post = News.query.get_or_404(post_id)
    
    # Lexi-style system context for cleaner generation
    prompt = (
        f"Context: {post.title}\n"
        "Task: Create 1 sophisticated poll question and 3 distinct options.\n"
        "Constraint: No markdown. Format: Question | Opt1 | Opt2 | Opt3"
    )
    
    try:
        # Switching to Gemini 3 Flash
        response = gemma_client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        
        raw_text = response.text.strip()
        parts = [p.strip() for p in raw_text.split('|')]
        
        # Fallback if splitting fails
        if len(parts) < 4:
            return jsonify({"question": "What is your view on this?", "options": ["Agree", "Disagree", "Neutral"]})

        return jsonify({
            "question": parts[0], 
            "options": parts[1:4],
            "status": "Published" # Ensures the 'None' issue is fixed at the source
        })
    except Exception as e:
        app.logger.error(f"Poll Gen Error: {e}")
        return jsonify({"error": "Failed to generate poll"}), 500

# 3. SMART ANSWER: Using Gemini 3 Flash
@app.route('/api/poll-review')
def poll_review():
    choice = request.args.get('choice')
    question = request.args.get('question')
    
    prompt = f"""
    The user was asked: '{question}'
    The user chose: '{choice}'
    Task: Provide a 2-sentence review. 
    Sentence 1: Estimate what % of people agree with this choice based on current trends.
    Sentence 2: Briefly analyze why this choice is logical or controversial.
    Format: You are on the [Side] side (Approx [X]%). [Analysis].
    Constraint: No markdown formatting.
    """
    
    try:
        response = gemma_client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        
        review = response.text.strip()
        return jsonify({
            "review": review,
            "status": "Published" # Corrects the card badge
        })
    except Exception as e:
        app.logger.error(f"Poll Review Error: {e}")
        return jsonify({"review": "I apologize, but I cannot assess the consensus right now."}), 500


def process_toy_response(raw_text):
    # More robust regex to handle spaces and various bracket styles
    new_questions = re.findall(r'\[Q:\s*(.*?)\]', raw_text)
    # Remove the question tags and asterisks for a clean typewriter effect
    clean_text = re.sub(r'\[Q:.*?\]', '', raw_text).replace('*', '').strip()
    return clean_text, new_questions

# 2. Page Route
@app.route('/toy-builder')
def toy_builder():
    return render_template('toy_builder.html')

@app.route('/ignite_toy', methods=['POST'])
def ignite_toy():
    # Use silent=True to prevent 400 errors if JSON is malformed
    data = request.get_json(silent=True) or {}
    
    brain = data.get('brain', 'Space Pirate') # Added default
    vibe = data.get('vibe', 'Happy') # Added default
    
    # Force a clean string for the session
    session['toy_identity'] = f"Role: {str(brain)}. Vibe: {str(vibe)}."
    session.modified = True 
    
    question_sets = {
        "Space Pirate": ["Where's the gold?", "Your ship stinks!", "Tell me about space krakens.", "Give me a star-map.", "Can I join your crew?", "Show me your hook!", "What's the best drink?", "Seen a black hole?", "Who's your rival?", "Is your parrot real?"],
        "Mad Scientist": ["Is the experiment safe?", "Can you turn me into a frog?", "Is time travel real?", "What's in that tube?", "Why is your lab smoking?", "Is the moon cheese?", "Tell me about your monster.", "Build a robot!", "Secret of eternal life?", "Who funds your lab?"],
        "Zen Master": ["Meaning of life?", "Show me how to breathe.", "Is the mountain tall?", "Why does the river flow?", "Secret of the stars?", "Tell a silent story.", "How to find peace?", "Is the tea ready?", "Where does wind go?", "Let go of anger."],
        "Detective": ["Who's the killer?", "Trust the butler?", "The missing sock mystery?", "Explain this clue.", "Spot a liar.", "Your hardest case?", "What's in the notebook?", "Last night's alibi?", "A shadow follows us!", "Is the case closed?"],
        "Medieval Knight": ["Where is the dragon?", "How heavy is that armor?", "Who is the king?", "Code of chivalry?", "Seen a wizard?", "The Great Tournament?", "Is my castle safe?", "Your horse's name?", "Become a squire.", "Find the Holy Grail."],
        "Cyberpunk Hacker": ["Can you bypass the firewall?", "Is the Matrix real?", "Who's tracking my IP?", "What's in the encrypted file?", "Can we crash the mainframe?", "Teach me to code in binary.", "Is my webcam on?", "Who's the best netrunner?", "What's the price of data?", "Can you hack a toaster?"],
        "Victorian Ghost": ["How did you pass away?", "Is there tea in the afterlife?", "Who's living in your mansion?", "Why are you rattling chains?", "Can you walk through walls?", "Do you miss the sun?", "Are you seeking revenge?", "Write a message in the dust.", "Why is the attic so cold?", "Do you remember 1888?"],
        "Conspiracy Theorist": ["Is the moon a hologram?", "Who really runs the world?", "What's in the chemtrails?", "Are birds actually drones?", "Tell me about Area 51.", "Why are they hiding the truth?", "Is Elvis still alive?", "What's the secret frequency?", "Who's watching us right now?", "Is the Earth flat or hollow?"],
        "Grumpy Cat": ["Where is my tuna?", "Stop touching my ears.", "Is it nap time yet?", "Why is the red dot winning?", "Can I knock this off the table?", "Who let the dog in?", "Rub my belly (but only twice).", "Why are you looking at me?", "Is the vacuum gone?", "I require treats. Now."],
        "Super Villain": ["What's your master plan?", "How do I build a doomsday device?", "Why is the hero so annoying?", "Where is your secret lair?", "Can I be your henchman?", "How do you handle capes?", "Is the world worth saving?", "Why do you monologue so much?", "What's your tragic backstory?", "Is the lava pit ready?"]
                }
    
    initial_qs = question_sets.get(brain, ["Who are you?", "What next?"])
    
    # Return a very explicit success signal
    return jsonify({
        "status": "ignited", 
        "initial_questions": initial_qs
    })

@app.route('/chat_toy', methods=['POST'])
def chat_toy():
    data = request.get_json()
    user_input = data.get('message')
    toy_identity = session.get('toy_identity', "A friendly toy.")
    
    prompt = (
        f"Act as: {toy_identity}\nUser: {user_input}\n"
        "Instructions: Respond in character. End with 3 follow-up questions in this EXACT format: [Q: Question]"
    )

    try:
        response = gemma_client.models.generate_content(model='gemma-3-4b-it', contents=prompt)
        reply_text, follow_up_qs = process_toy_response(response.text)
        
        return jsonify({
            "reply": reply_text,
            "new_questions": follow_up_qs if follow_up_qs else ["Tell me more!", "What else?"]
        })
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"reply": "My gears jammed!", "new_questions": []}), 500


# 5. Reset Route
@app.route('/reset_toy', methods=['POST'])
def reset_toy():
    session.pop('toy_identity', None)
    return jsonify({"status": "disassembled"})


@app.route('/generate_audio/<int:post_id>', methods=['POST'])
def generate_audio(post_id):
    try:
        # 1. DATABASE FETCH
        post = News.query.get_or_404(post_id)

        # 2. AI SUMMARIZATION (The Intermediate Step)
        # We create a specific instruction for the Audio Summary
        summary_instruction = (
            f"You are a professional news reader. Summarize the following blog post into a "
            f"clear, engaging 3-sentence summary for an audio broadcast. "
            f"Do not use bullet points or special characters. "
            f"Post Content: {post.description[:2000]}"
        )
        
        # Call your existing function (Block 2) to get the text summary
        audio_script = ask_gemini_for_prompt(summary_instruction)

        # 3. TEXT TO SPEECH (Block 1 Logic)
        model_id = "gemini-2.5-flash-preview-tts"
        response = gemma_client.models.generate_content(
            model=model_id,
            contents=audio_script, # We now read the summary, not the full post
            config=gemma_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=gemma_types.SpeechConfig(
                    voice_config=gemma_types.VoiceConfig(
                        prebuilt_voice_config=gemma_types.PrebuiltVoiceConfig(
                            voice_name="Puck" 
                        )
                    )
                )
            )
        )

        # 4. EXTRACT PCM AND STREAM WAV
        pcm_data = response.candidates[0].content.parts[0].inline_data.data
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_data)
        
        wav_io.seek(0)
        return send_file(wav_io, mimetype="audio/wav")

    except Exception as e:
        app.logger.error(f"Integrated TTS Error for post {post_id}: {e}")
        return str(e), 500


# gemma based alex

@app.route('/ask-lexi', methods=['POST'])
def ask_lexi():
    data = request.get_json()
    user_msg = data.get('message', '')

    instruction = "You are Lexi, a professional British Concierge. Be brief and elegant."
    prompt = f"{instruction}\n\nUser: {user_msg}\nLexi:"

    try:
        # Check if client exists
        response = gemma_client.models.generate_content(
            model='gemma-3-4b-it', 
            contents=prompt
        )
        return jsonify({"reply": response.text.strip()})

    except Exception as e:
        # THIS LINE IS CRITICAL: Check your Render/Terminal logs for this output
        print(f"!!! GEMMA SYSTEM ERROR: {str(e)}") 
        return jsonify({"reply": "I apologize, but I've encountered an error."}), 500


@app.route('/ai-apps')
def all_ai_apps():
    # Only fetch entries that were tagged as AI Apps via the admin_ai_apps page
    ai_apps = News.query.filter(News.description.isnot(None),News.description.like('[AI_APP]%')).order_by(News.title.asc()).all()
    return render_template('all_ai_apps.html', ai_apps=ai_apps)

@app.route('/admin/ai-apps')
@login_required
@admin_required
def admin_ai_apps():
    # Only fetch items that have our special tag in the description
    ai_apps = News.query.filter(News.description.like('[AI_APP]%')).order_by(News.timestamp.desc()).all()
    return render_template('admin_ai_apps.html', ai_apps=ai_apps)

@app.route('/admin/ai-apps/add', methods=['POST'])
@login_required
@admin_required
def add_ai_app():
    title = request.form.get('title')
    url = request.form.get('url')

    category = request.form.get('category') # New field from dropdown
    raw_desc = request.form.get('description')
    
    # Prepend the tag so we can distinguish this from news later
    description = f"[AI_APP][{category}] {raw_desc}"
    
    new_app = News(
        title=title, 
        url=url, 
        description=description, 
        user_id=current_user.id
    )
    db.session.add(new_app)
    db.session.commit()
    flash('AI App added to directory successfully!', 'success')
    return redirect(url_for('admin_ai_apps'))

@app.route('/admin/ai-apps/edit/<int:app_id>', methods=['POST'])
@login_required
@admin_required
def edit_ai_app(app_id):
    app_to_edit = News.query.get_or_404(app_id)
    
    # Get the updated data from the modal form
    app_to_edit.title = request.form.get('title')
    app_to_edit.url = request.form.get('url')
    
    category = request.form.get('category')
    raw_desc = request.form.get('description')
    
    # Re-apply the tagging structure
    app_to_edit.description = f"[AI_APP][{category}] {raw_desc}"
    
    db.session.commit()
    flash(f'Updated {app_to_edit.title} successfully!', 'info')
    return redirect(url_for('admin_ai_apps'))

@app.route('/admin/ai-apps/delete/<int:app_id>', methods=['POST'])
@login_required
@admin_required
def delete_ai_app(app_id):
    app_to_delete = News.query.get_or_404(app_id)
    db.session.delete(app_to_delete)
    db.session.commit()
    flash('App removed from directory.', 'info')
    return redirect(url_for('admin_ai_apps'))

@app.route('/donate')
def donate():
    return render_template('donate.html')

# 1. Public Marketplace View
@app.route('/prompts')
def prompts_marketplace():
    # Fetch everything tagged with [PROMPT]
    prompts = News.query.filter(News.description.like('[PROMPT]%')).order_by(News.timestamp.desc()).all()
    return render_template('prompts_marketplace.html', prompts=prompts)

# 2. Admin Management Page
@app.route('/admin/prompts')
@login_required
@admin_required
def admin_prompts():
    prompts = News.query.filter(News.description.like('[PROMPT]%')).order_by(News.timestamp.desc()).all()
    return render_template('admin_prompts.html', prompts=prompts)

# 3. Add Prompt (Used by Admin or User)
@app.route('/admin/prompts/add', methods=['POST'])
@login_required
@admin_required
def add_prompt():
    title = request.form.get('title')
    category = request.form.get('category') # e.g., Coding, Writing, Marketing
    visibility = request.form.get('visibility', 'OPEN') # OPEN or LOCKED
    content = request.form.get('content') # The actual prompt text
    
    # Structure: [PROMPT][CATEGORY][VISIBILITY] Content
    description = f"[PROMPT][{category}][{visibility}] {content}"
    
    new_prompt = News(
        title=title,
        description=description,
        user_id=current_user.id,
        url="#" # Prompts don't necessarily need a URL, but we'll fill it
    )
    db.session.add(new_prompt)
    db.session.commit()
    flash('Prompt submitted successfully!', 'success')
    return redirect(request.referrer) # Returns user to wherever they were

# 4. Edit Prompt (Admin only)
@app.route('/admin/prompts/edit/<int:prompt_id>', methods=['POST'])
@login_required
@admin_required
def edit_prompt(prompt_id):
    prompt = News.query.get_or_404(prompt_id)
    category = request.form.get('category')
    visibility = request.form.get('visibility')
    content = request.form.get('content')
    
    prompt.title = request.form.get('title')
    prompt.description = f"[PROMPT][{category}][{visibility}] {content}"
    
    db.session.commit()
    flash('Prompt updated!', 'info')
    return redirect(url_for('admin_prompts'))

@app.route('/delete_prompt/<int:prompt_id>', methods=['POST'])
@login_required
@admin_required
def delete_prompt(prompt_id):
    prompt = SavedPrompt.query.get_or_404(prompt_id)
    # Check if user is owner or admin
    if prompt.user_id != current_user.id and not current_user.is_admin:
        flash('Permission denied.', 'danger')
        return redirect(url_for('index'))
    
    db.session.delete(prompt)
    db.session.commit()
    flash('Prompt deleted successfully.', 'success')
    return redirect(request.referrer or url_for('index'))

@app.route('/ai-hub')
def ai_hub():
    # Fetch only entries tagged with [AI_HUB] in the description
    # Filter(News.description.isnot(None)) ensures we don't hit null errors
    hub_resources = News.query.filter(
        News.description.isnot(None),
        News.description.like('[AI_HUB]%')
    ).order_by(News.title.asc()).all()
    
    return render_template('ai_hub.html', ai_apps=hub_resources)

@app.route('/admin/ai-hub')
@login_required
@admin_required
def admin_ai_hub():
    # Show all hub items in the admin management table
    hub_resources = News.query.filter(News.description.like('[AI_HUB]%')).order_by(News.timestamp.desc()).all()
    return render_template('admin_ai_hub.html', ai_apps=hub_resources)

@app.route('/admin/ai-hub/add', methods=['POST'])
@login_required
@admin_required
def add_hub_resource():
    title = request.form.get('title')
    url = request.form.get('url')
    category = request.form.get('category')  # Dropdown: Newsletter, Podcast, Book, etc.
    raw_desc = request.form.get('description')
    
    # Store with the specific Hub tag: [AI_HUB][Category] Description
    description = f"[AI_HUB][{category}] {raw_desc}"
    
    new_resource = News(
        title=title, 
        url=url, 
        description=description, 
        user_id=current_user.id
    )
    db.session.add(new_resource)
    db.session.commit()
    flash(f'"{title}" added to Discovery Hub successfully!', 'success')
    return redirect(url_for('admin_ai_hub'))

@app.route('/admin/ai-hub/edit/<int:app_id>', methods=['POST'])
@login_required
@admin_required
def edit_hub_resource(app_id):
    resource = News.query.get_or_404(app_id)
    
    # Update core details
    resource.title = request.form.get('title')
    resource.url = request.form.get('url')
    
    # Re-build the tagged description
    category = request.form.get('category')
    raw_desc = request.form.get('description')
    resource.description = f"[AI_HUB][{category}] {raw_desc}"
    
    db.session.commit()
    flash(f'Updated {resource.title} successfully!', 'info')
    return redirect(url_for('admin_ai_hub'))

@app.route('/admin/ai-hub/delete/<int:app_id>', methods=['POST'])
@login_required
@admin_required
def delete_hub_resource(app_id):
    resource = News.query.get_or_404(app_id)
    title = resource.title
    
    db.session.delete(resource)
    db.session.commit()
    flash(f'"{title}" removed from the Hub.', 'warning')
    return redirect(url_for('admin_ai_hub'))

    
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


# --- NEW: Forgot Password Routes ---
@app.route('/forgot_password', methods=['GET'])
def forgot_password():
   return render_template('forgot_password.html')


@app.route('/send_reset_link', methods=['POST'])
async def send_reset_link():
   username = request.form.get('username')
   user = User.query.filter_by(username=username).first()
  
   if user:
       if not user.email:
           flash('This account does not have an email address associated for password reset. Please contact support.', 'danger')
           return redirect(url_for('forgot_password'))


       # Generate a unique token
       token = str(uuid.uuid4())
       # Set token expiration (e.g., 1 hour from now)
       expiration = datetime.utcnow() + timedelta(hours=1)
      
       user.password_reset_token = token
       user.password_reset_expiration = expiration
       db.session.commit()
      
       reset_link = url_for('reset_password', token=token, _external=True)
      
       try:
           msg = Message('Password Reset Request for SuperPrompter',
                         sender=app.config['MAIL_USERNAME'],
                         recipients=[user.email])
           msg.body = f"""
Dear {user.username},


You have requested a password reset for your SuperPrompter account.


Please click on the following link to reset your password:
{reset_link}


This link will expire in 1 hour.


If you did not request a password reset, please ignore this email.


Sincerely,
The SuperPrompter Team
"""
           mail.send(msg)
           app.logger.info(f"Password reset email sent to {user.email} for user {user.username}")
           flash('A password reset link has been sent to your email address. Please check your inbox (and spam folder).', 'info')
       except Exception as e:
           app.logger.error(f"Failed to send password reset email to {user.email}: {e}", exc_info=True)
           flash('Failed to send password reset email. Please try again later or contact support.', 'danger')
   else:
       # For security, always give a generic success message even if the user doesn't exist
       flash('If an account with that username exists, a password reset link has been sent to the associated email address.', 'info')
  
   return redirect(url_for('login'))


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
   user = User.query.filter_by(password_reset_token=token).first()
   now = datetime.utcnow()


   if not user or user.password_reset_expiration < now:
       flash('The password reset link is invalid or has expired.', 'danger')
       return redirect(url_for('forgot_password'))


   if request.method == 'POST':
       new_password = request.form.get('new_password')
       confirm_new_password = request.form.get('confirm_new_password')


       if not new_password or not confirm_new_password:
           flash('Both new password fields are required.', 'danger')
           return render_template('reset_password.html', token=token)


       if new_password != confirm_new_password:
           flash('New password and confirmation do not match.', 'danger')
           return render_template('reset_password.html', token=token)


       if len(new_password) < 6: # Example: enforce minimum password length
           flash('New password must be at least 6 characters long.', 'danger')
           return render_template('reset_password.html', token=token)


       user.set_password(new_password)
       user.password_reset_token = None # Invalidate the token after use
       user.password_reset_expiration = None
       db.session.commit()
       flash('Your password has been reset successfully! Please log in with your new password.', 'success')
       return redirect(url_for('login'))


   return render_template('reset_password.html', token=token)
# --- END NEW: Forgot Password Routes ---

from datetime import datetime # Ensure this import is at the top of app.py

@app.route('/submit_to_community', methods=['POST'])
@login_required
def submit_to_community():
    try:
        data = request.get_json()
        prompt_text = data.get('prompt_text')
        variant = data.get('variant', 'General')
        
        if not prompt_text:
            return jsonify({"success": False, "error": "Prompt content is empty"}), 400

        # Generate a short title for the news entry
        words = prompt_text.split()
        generated_title = " ".join(words[:5]) + ("..." if len(words) > 5 else "")

        # Create the database record
        new_entry = News(
            title=f"Community: {generated_title}",
            description=f"[PROMPT][{variant.capitalize()}][OPEN] {prompt_text}",
            url="#",
            # FIX: Use Python datetime objects, not strings
            timestamp=datetime.now(), 
            published_date=datetime.now().date(), 
            user_id=current_user.id
        )

        db.session.add(new_entry)
        db.session.commit()
        return jsonify({"success": True})

    except Exception as e:
        db.session.rollback()
        print(f"Database Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


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
      # NEW: Get email from registration form
      email = request.form.get('email') # Make sure to add this field to your register.html


      user = User.query.filter_by(username=username).first()
      if user:
          flash('this username already exists', 'danger') # Updated message
          # Generate suggestions if username exists
          suggestions = generate_unique_username_suggestions(username)
          return render_template('register.html', suggestions=suggestions) # Pass suggestions
      else:
          # NEW: Check if email already exists
          if email and User.query.filter_by(email=email).first():
              flash('Email already registered. Please use a different email or log in.', 'danger')
              return render_template('register.html') # Re-render without suggestions for email conflict
          else:
              new_user = User(username=username, email=email) # Pass email to User constructor
              new_user.set_password(password)
              db.session.add(new_user)
              db.session.commit()
              login_user(new_user) # Automatically log in the new user
              flash('Registration successful! You are now logged in.', 'success')
              return redirect(url_for('app_home')) # Redirect directly to app_home
  return render_template('register.html') # Initial GET request, no suggestions


# NEW: Helper function to generate unique username suggestions
def generate_unique_username_suggestions(base_username, num_suggestions=3):
   suggestions = []
   attempts = 0
   max_attempts_per_suggestion = 10 # Prevent infinite loops


   while len(suggestions) < num_suggestions and attempts < num_suggestions * max_attempts_per_suggestion:
       suffix = ''.join(random.choices(string.digits, k=4)) # 4 random digits
       new_username = f"{base_username}{suffix}"
      
       # Ensure the suggestion is not too long
       if len(new_username) > 80: # Max length for username field
           new_username = f"{base_username[:76]}{suffix}" # Truncate base_username if needed


       if not User.query.filter_by(username=new_username).first():
           suggestions.append(new_username)
       attempts += 1
  
   # If we still don't have enough suggestions, try more generic ones
   while len(suggestions) < num_suggestions:
       random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
       generic_username = f"user_{random_suffix}"
       if not User.query.filter_by(username=generic_username).first():
           suggestions.append(generic_username)
  
   return suggestions

@app.route('/send_request', methods=['GET', 'POST'])
@login_required
async def send_request():
    if request.method == 'POST':
        request_type = request.form.get('request_type')
        company = request.form.get('company_name', 'Not Provided')
        message_body = request.form.get('message')

        try:
            # --- EMAIL 1: Notification to You (Admin) ---
            admin_msg = Message(
                subject=f"New {request_type} from {current_user.username}",
                sender=app.config['MAIL_USERNAME'],
                recipients=['info@promptsgenerator.ai'] # Your inbox
            )
            admin_msg.body = f"User {current_user.username} sent a {request_type} request.\n\nMessage:\n{message_body}"
            mail.send(admin_msg)

            # --- EMAIL 2: Confirmation to the User (Auto-Reply) ---
            if current_user.email: # Check if user has an email saved
                user_msg = Message(
                    subject="We've received your SuperPrompter request!",
                    sender=app.config['MAIL_USERNAME'],
                    recipients=[current_user.email] # The user's inbox
                )
                user_msg.body = f"""
Hi {current_user.username},

Thank you for reaching out to SuperPrompter! 

We have received your request regarding "{request_type}". Our team is currently reviewing the details, and we will get back to you as soon as possible.

Your Message Summary:
-----------------------------------
Category: {request_type}
Company: {company}
Message: {message_body}
-----------------------------------

Best regards,
The SuperPrompter Team
"""
                mail.send(user_msg)

            flash("Your request has been sent! Check your email for a confirmation.", "success")
            return redirect(url_for('app_home'))

        except Exception as e:
            app.logger.error(f"Mail failure: {e}")
            flash("Request sent, but we couldn't send a confirmation email.", "warning")
            return redirect(url_for('app_home'))

    return render_template('send_request.html')



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
            
            # --- UPDATED LOGIC TO HANDLE REDIRECTION ---
            # 1. Get the 'next' URL parameter passed by Flask-Login (e.g., /all_jobs)
            next_page = request.args.get('next')
            
            # 2. Redirect to the 'next' page if it exists, otherwise redirect to 'app_home'
            # (In a production environment, you should use 'is_safe_url' here to prevent phishing)
            return redirect(next_page or url_for('app_home')) 
            
        else:
            flash('Login Unsuccessful. Please check username and password.', 'danger')
            # For POST failure, re-render the login page
            return render_template('login.html') 

    # For GET request, render the login page
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
      # For admin, set a dummy email or leave None if not required for testing password reset
      admin_user.email = 'admin@example.com' # Assign a dummy email for admin
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
