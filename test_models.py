import os
from dotenv import load_dotenv
import requests

load_dotenv()
token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

models_to_test = [
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-3.1-8B"
]

for model in models_to_test:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": "What is 2+2?",
        "options": {"wait_for_model": True}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"✅ {model} - WORKS")
        else:
            print(f"❌ {model} - Error {response.status_code}")
    except Exception as e:
        print(f"❌ {model} - {str(e)}")