import os
from openai import OpenAI

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def call_llm(prompt: str, model: str = "gpt-4.1-mini", temperature: float = 0.2) -> str:
    """
    Deterministic LLM call for production use.
    Low temperature to reduce variance.
    """
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty content")
    return content
