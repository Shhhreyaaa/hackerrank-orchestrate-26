import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

print("Listing available models:")
try:
    for m in client.models.list():
        # print the model representation or just the name
        print(f"Model Name: {m.name}")
except Exception as e:
    print(f"Failed to list models: {e}")
