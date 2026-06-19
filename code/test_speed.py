import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

models = ["gemini-flash-latest", "gemini-2.0-flash-lite", "gemini-3.5-flash"]

for model in models:
    print(f"\nTesting {model}...")
    success = 0
    total_time = 0.0
    for i in range(3):
        start = time.time()
        try:
            response = client.models.generate_content(
                model=model,
                contents="Hello"
            )
            elapsed = time.time() - start
            total_time += elapsed
            success += 1
            print(f"  Call {i+1}: Success in {elapsed:.2f}s -> {response.text.strip()}")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  Call {i+1}: Failed in {elapsed:.2f}s -> {e}")
        time.sleep(1.0)
    
    if success > 0:
        print(f"Summary for {model}: {success}/3 succeeded, Avg Time: {total_time/success:.2f}s")
    else:
        print(f"Summary for {model}: 0/3 succeeded")
