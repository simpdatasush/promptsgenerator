import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
import asyncio
import threading
import google.generativeai as genai
import os # For API key check

# --- Gemini API Key and Configuration ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual API key from Google AI Studio.
# Get your API key here: https://aistudio.google.com/app/apikey
# If you leave this as the placeholder, Gemini features WILL NOT WORK and you will see warnings.
GEMINI_API_KEY = "YOUR_API_KEY_HERE" # <<< REPLACE THIS PLACEHOLDER KEY WITH YOUR REAL KEY >>>

# Global flag to track if Gemini API was successfully configured
GEMINI_API_CONFIGURED = False

# Function to safely get API key at startup
def get_gemini_api_key_at_startup():
    global GEMINI_API_KEY, GEMINI_API_CONFIGURED
    
    # Check if the key is already set (e.g., from a previous run or environment variable)
    if GEMINI_API_KEY != "YOUR_API_KEY_HERE" and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            GEMINI_API_CONFIGURED = True
            print("Gemini API configured successfully from predefined key.")
            return
        except Exception as e:
            print(f"ERROR: Failed to configure Gemini API with predefined key: {e}")
            GEMINI_API_KEY = "" # Clear invalid key

    # If not configured, prompt the user
    temp_root_for_dialog = None
    if not tk._default_root: # Check if Tkinter root is not yet initialized
        temp_root_for_dialog = tk.Tk()
        temp_root_for_dialog.withdraw() # Hide the main window

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        try:
            dialog_key = simpledialog.askstring(
                "Gemini API Key Required",
                "Please enter your Google Gemini API Key. \n"
                "You can get one from: https://aistudio.google.com/app/apikey\n\n"
                "This key is necessary for generating prompts.",
                parent=temp_root_for_dialog
            )
            if dialog_key:
                GEMINI_API_KEY = dialog_key.strip()
                print("DEBUG: Gemini API Key received from user input.")
            else:
                print("WARNING: User cancelled API Key input or provided an empty key.")
                GEMINI_API_KEY = ""
        except Exception as e:
            print(f"ERROR: Failed to get API key from user dialog: {e}")
            GEMINI_API_KEY = ""

    if temp_root_for_dialog:
        temp_root_for_dialog.destroy()

    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            GEMINI_API_CONFIGURED = True
            print("Gemini API configured successfully.")
        except Exception as e:
            print(f"ERROR: Failed to configure Gemini API: {e}")
            print("Please ensure your API key is correct and valid in Google AI Studio.")
            GEMINI_API_CONFIGURED = False
    else:
        print("\n" + "="*80)
        print("WARNING: Gemini API Key is empty. Prompt generation features will be disabled.")
        print("="*80 + "\n")

# Global variables for the dedicated asyncio loop and thread
async_loop = None
async_thread = None

def start_async_loop():
    """Starts a new asyncio event loop in a dedicated thread."""
    global async_loop
    async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(async_loop)
    async_loop.run_forever()

def submit_async_task(coroutine):
    """Submits an asynchronous coroutine to the dedicated asyncio loop from another thread."""
    if async_loop and async_loop.is_running():
        return asyncio.run_coroutine_threadsafe(coroutine, async_loop)
    else:
        raise RuntimeError("Asyncio event loop is not running. Cannot submit task.")

def run_async_task_in_thread(coroutine_func):
    """Helper to run an async coroutine in the dedicated async thread and handle results/exceptions."""
    try:
        future = submit_async_task(coroutine_func)
        result = future.result() # This will block until the coroutine finishes or raises an exception
        return result
    except asyncio.CancelledError:
        root.after(0, lambda: app_output.insert(tk.END, "\n❗ Operation Cancelled.\n"))
        root.after(0, lambda: app_output.see(tk.END))
        return "Operation Cancelled."
    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Async Task Error", f"An error occurred during the async operation: {e}"))
        root.after(0, lambda: app_output.insert(tk.END, f"\n❌ Async task failed: {e}\n"))
        root.after(0, lambda: app_output.see(tk.END))
        return f"Error: {e}"

async def ask_gemini_for_prompt(prompt_instruction, max_output_tokens=4096):
    """
    Makes an asynchronous API call to Gemini for prompt generation.
    max_output_tokens is set high to allow for detailed prompt outputs.
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
    
    app_output.insert(tk.END, "\n🚀 Generating prompts and variants...\n")
    app_output.see(tk.END)

    # 1. Polished Prompt
    polished_prompt_instruction = f"Refine the following text into a clear, concise, and effective prompt for a large language model. Improve grammar, clarity, and structure. Do not add external information, only refine the given text:\n\n{raw_input}"
    polished_prompt_task = asyncio.create_task(ask_gemini_for_prompt(polished_prompt_instruction))

    # Wait for the polished prompt before generating variants and additions
    polished_prompt = await polished_prompt_task
    if "Error" in polished_prompt or "not configured" in polished_prompt:
        app_output.insert(tk.END, f"❌ Failed to generate polished prompt: {polished_prompt}\n")
        app_output.see(tk.END)
        return {
            "polished": polished_prompt,
            "creative": "", "technical": "", "shorter": "", "additions": ""
        }
    
    app_output.insert(tk.END, "✅ Polished prompt generated.\n")
    app_output.see(tk.END)

    # 2. Prompt Variants (Creative, Technical, Shorter)
    creative_prompt_instruction = f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_prompt}"
    technical_prompt_instruction = f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_prompt}"
    shorter_prompt_instruction = f"Condense the following prompt into its shortest possible form while retaining all essential meaning and instructions. Aim for brevity.:\n\n{polished_prompt}"
    
    creative_task = asyncio.create_task(ask_gemini_for_prompt(creative_prompt_instruction))
    technical_task = asyncio.create_task(ask_gemini_for_prompt(technical_prompt_instruction))
    shorter_task = asyncio.create_task(ask_gemini_for_prompt(shorter_prompt_instruction, max_output_tokens=512)) # Shorter variant can have a smaller token limit

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
    
    app_output.insert(tk.END, "✅ Variants and additions generated.\n")
    app_output.see(tk.END)

    return {
        "polished": polished_prompt,
        "creative": creative_result,
        "technical": technical_result,
        "shorter": shorter_result,
        "additions": additions_result
    }

def generate_prompts_gui():
    """
    GUI wrapper to get user input and initiate asynchronous prompt generation.
    """
    raw_input = input_text_area.get(1.0, tk.END).strip()
    
    def update_gui_with_results(results):
        polished_output_area.delete(1.0, tk.END)
        polished_output_area.insert(tk.END, results["polished"])

        creative_output_area.delete(1.0, tk.END)
        creative_output_area.insert(tk.END, results["creative"])

        technical_output_area.delete(1.0, tk.END)
        technical_output_area.insert(tk.END, results["technical"])

        shorter_output_area.delete(1.0, tk.END)
        shorter_output_area.insert(tk.END, results["shorter"])

        additions_output_area.delete(1.0, tk.END)
        additions_output_area.insert(tk.END, results["additions"])

        app_output.insert(tk.END, "\n✨ All prompt generation tasks complete.\n")
        app_output.see(tk.END)

    # Run the async generation in the dedicated thread
    threading.Thread(target=lambda: root.after(0, update_gui_with_results,
        run_async_task_in_thread(generate_prompts_async(raw_input))
    )).start()

# --- GUI Setup ---
root = tk.Tk()
root.title("Prompt Generator")
root.geometry("1000x800") # Set initial window size

# Configure a modern, clean theme
THEME_COLORS = {
    "bg": "#F0F2F5",          # Light grey background
    "fg": "#333333",          # Dark grey text
    "button_bg": "#4CAF50",   # Green button
    "button_fg": "#FFFFFF",   # White button text
    "button_active_bg": "#45A049", # Darker green on active
    "text_bg": "#FFFFFF",     # White text area background
    "text_fg": "#333333",     # Dark grey text area text
    "frame_bg": "#E0E2E5",    # Slightly darker frame background
    "header_fg": "#2C3E50"    # Dark blue-grey for headers
}

def apply_default_theme():
    """Applies the default theme to all relevant widgets."""
    root.config(bg=THEME_COLORS["bg"])

    # Apply to frames
    for frame in [main_container_frame, input_frame, output_frame, 
                   polished_frame, creative_frame, technical_frame, 
                   shorter_frame, additions_frame]:
        frame.config(bg=THEME_COLORS["frame_bg"], bd=1, relief="solid")

    # Apply to labels
    for label in [title_label, input_label, polished_label, creative_label,
                  technical_label, shorter_label, additions_label]:
        label.config(bg=THEME_COLORS["frame_bg"], fg=THEME_COLORS["header_fg"], font=("Inter", 10, "bold"))
    
    # Apply to text areas
    for text_area in [input_text_area, polished_output_area, creative_output_area,
                      technical_output_area, shorter_output_area, additions_output_area,
                      app_output]:
        text_area.config(bg=THEME_COLORS["text_bg"], fg=THEME_COLORS["text_fg"], 
                         insertbackground=THEME_COLORS["text_fg"], 
                         highlightbackground=THEME_COLORS["frame_bg"], 
                         highlightcolor=THEME_COLORS["button_bg"])
    
    # Apply to buttons
    generate_button.config(bg=THEME_COLORS["button_bg"], fg=THEME_COLORS["button_fg"],
                           activebackground=THEME_COLORS["button_active_bg"],
                           activeforeground=THEME_COLORS["button_fg"],
                           relief=tk.FLAT, bd=0, padx=10, pady=5)
    generate_button.bind("<Enter>", lambda e: generate_button.config(bg=THEME_COLORS["button_active_bg"]))
    generate_button.bind("<Leave>", lambda e: generate_button.config(bg=THEME_COLORS["button_bg"]))


# Main container frame
main_container_frame = tk.Frame(root, bg=THEME_COLORS["bg"], padx=10, pady=10)
main_container_frame.pack(fill=tk.BOTH, expand=True)

# Title Label
title_label = tk.Label(main_container_frame, text="Prompt Generator", font=("Inter", 18, "bold"), fg=THEME_COLORS["header_fg"])
title_label.pack(pady=(0, 15))

# Input Frame
input_frame = tk.LabelFrame(main_container_frame, text="Your Initial Prompt Idea", padx=10, pady=10, font=("Inter", 10, "bold"))
input_frame.pack(fill=tk.X, pady=10)

input_label = tk.Label(input_frame, text="Enter your raw prompt text here:")
input_label.pack(anchor=tk.W, pady=(0, 5))

input_text_area = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=6, font=("Inter", 10))
input_text_area.pack(fill=tk.BOTH, expand=True)

# Generate Button
generate_button = tk.Button(main_container_frame, text="Generate Prompts", command=generate_prompts_gui, font=("Inter", 12, "bold"), bg="#4CAF50", fg="black")
generate_button.pack(pady=15)

# Output Frame
output_frame = tk.Frame(main_container_frame)
output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

# Configure output_frame grid for 2 columns, 3 rows
output_frame.grid_columnconfigure(0, weight=1)
output_frame.grid_columnconfigure(1, weight=1)
output_frame.grid_rowconfigure(0, weight=1)
output_frame.grid_rowconfigure(1, weight=1)
output_frame.grid_rowconfigure(2, weight=1)


# Polished Prompt
polished_frame = tk.LabelFrame(output_frame, text="Polished Prompt", padx=10, pady=10, font=("Inter", 10, "bold"))
polished_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
polished_label = tk.Label(polished_frame, text="A refined version of your input:")
polished_label.pack(anchor=tk.W, pady=(0, 5))
polished_output_area = scrolledtext.ScrolledText(polished_frame, wrap=tk.WORD, height=8, font=("Inter", 10))
polished_output_area.pack(fill=tk.BOTH, expand=True)

# Creative Variant
creative_frame = tk.LabelFrame(output_frame, text="Creative Variant", padx=10, pady=10, font=("Inter", 10, "bold"))
creative_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
creative_label = tk.Label(creative_frame, text="A more imaginative approach:")
creative_label.pack(anchor=tk.W, pady=(0, 5))
creative_output_area = scrolledtext.ScrolledText(creative_frame, wrap=tk.WORD, height=8, font=("Inter", 10))
creative_output_area.pack(fill=tk.BOTH, expand=True)

# Technical Variant
technical_frame = tk.LabelFrame(output_frame, text="Technical Variant", padx=10, pady=10, font=("Inter", 10, "bold"))
technical_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
technical_label = tk.Label(technical_frame, text="A precise and detailed version:")
technical_label.pack(anchor=tk.W, pady=(0, 5))
technical_output_area = scrolledtext.ScrolledText(technical_frame, wrap=tk.WORD, height=8, font=("Inter", 10))
technical_output_area.pack(fill=tk.BOTH, expand=True)

# Shorter Variant
shorter_frame = tk.LabelFrame(output_frame, text="Shorter Variant", padx=10, pady=10, font=("Inter", 10, "bold"))
shorter_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
shorter_label = tk.Label(shorter_frame, text="A concise and brief alternative:")
shorter_label.pack(anchor=tk.W, pady=(0, 5))
shorter_output_area = scrolledtext.ScrolledText(shorter_frame, wrap=tk.WORD, height=8, font=("Inter", 10))
shorter_output_area.pack(fill=tk.BOTH, expand=True)

# Suggested Additions (spans two columns)
additions_frame = tk.LabelFrame(output_frame, text="Suggested Additions", padx=10, pady=10, font=("Inter", 10, "bold"))
additions_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
additions_label = tk.Label(additions_frame, text="Ideas to enhance your prompt (e.g., tone, format, audience):")
additions_label.pack(anchor=tk.W, pady=(0, 5))
additions_output_area = scrolledtext.ScrolledText(additions_frame, wrap=tk.WORD, height=8, font=("Inter", 10))
additions_output_area.pack(fill=tk.BOTH, expand=True)

# Application Output Console
app_output = scrolledtext.ScrolledText(
    main_container_frame,
    wrap=tk.WORD,
    height=4,
    font=("Inter", 9),
    bg=THEME_COLORS["text_bg"],
    fg=THEME_COLORS["text_fg"]
)
app_output.pack(pady=10, fill=tk.X, expand=False)


# --- Asynchronous Event Loop Setup ---
async_thread = threading.Thread(target=start_async_loop, daemon=True)
async_thread.start()

# Function to handle graceful shutdown when the window is closed
def on_closing():
    """Stops the asyncio loop and destroys the Tkinter root window."""
    if async_loop and async_loop.is_running():
        async_loop.call_soon_threadsafe(async_loop.stop)
    root.destroy()

# Bind the on_closing function to the window close protocol
root.protocol("WM_DELETE_WINDOW", on_closing)

# --- Initialization ---
# Attempt to get and configure API key at startup
get_gemini_api_key_at_startup()

# Apply theme after all widgets are created
apply_default_theme()

root.mainloop()
