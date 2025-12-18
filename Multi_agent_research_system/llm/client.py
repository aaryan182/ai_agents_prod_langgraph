from openai import OpenAI
client = OpenAI()

def call_llm(prompt: str, model= "gpt-4.1"):
    res = client.chat.completions.create(
        model = model,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content