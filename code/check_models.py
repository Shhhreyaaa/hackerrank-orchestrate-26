import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

models_to_test = ["gemini-flash-latest", "gemini-3.5-flash"]

for model in models_to_test:
    print(f"Testing model: {model}")
    try:
        response = client.models.generate_content(
            model=model,
            contents="Say hello in one word."
        )
        print(f"SUCCESS for {model}: {response.text.strip()}")
    except Exception as e:
        print(f"FAILED for {model}: {e}")
