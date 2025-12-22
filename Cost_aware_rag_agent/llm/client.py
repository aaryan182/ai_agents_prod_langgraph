import os
from openai import OpenAI

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def call_llm(prompt: str, model: str, temperature: float = 0.2) -> str:
    client = get_client()
    res = client.chat.completions.create(
        model = model,
        temperature = temperature,
        messages=[{"role": "user", "content" : prompt}],
    )
    
    return res.choices[0].message.content or ""