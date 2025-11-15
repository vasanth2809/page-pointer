import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)


def call_gemini(prompt: str, model_name="gemini-2.5-flash"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text
