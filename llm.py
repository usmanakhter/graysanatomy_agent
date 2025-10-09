"""
LLM interaction - supports HuggingFace and OpenAI
"""
from dotenv import load_dotenv
import os
import requests
from config import USE_OPENAI, LLM_MODEL, PROMPT_TEMPLATE


def get_answer_openai(context, question):
    """Get answer using OpenAI API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    client = OpenAI(api_key=api_key)
    
    prompt = PROMPT_TEMPLATE.format(context=context[:4000], question=question)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )
    
    return response.choices[0].message.content


def get_answer_huggingface(context, question):
    """Get answer using HuggingFace Inference API"""
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment")
    
    API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # Format prompt for Mistral
    prompt = f"""<s>[INST] {PROMPT_TEMPLATE.format(context=context[:3000], question=question)} [/INST]"""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.1,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "Error: No text generated")
        return str(result)
    else:
        raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")


def get_answer(context, question):
    """Get answer from LLM (routes to appropriate API)"""
    if USE_OPENAI:
        return get_answer_openai(context, question)
    else:
        return get_answer_huggingface(context, question)