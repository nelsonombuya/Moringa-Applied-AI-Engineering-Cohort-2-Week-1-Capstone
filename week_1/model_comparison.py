# model_comparison.py (starter)
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# I'll be using OpenRouter for the cloud model
# if os.getenv("OPENROUTER_API_KEY"):
#     client = OpenAI(
#         api_key=os.getenv("OPENROUTER_API_KEY"),
#         base_url="https://openrouter.ai/api/v1",
#     )
#     model = "openrouter/free"
# else:
#     client = OpenAI(
#         api_key="ollama",
#         base_url="http://localhost:11434/v1"
#     )
#     model = "llama3.2"
models = [
    {
        "name": "Local Model",
        "client": OpenAI(api_key="ollama", base_url="http://localhost:11434/v1"),
        "model": "llama3.2",
    },
    {
        "name": "Cloud Model",
        "client": OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        ),
        "model": "openrouter/free",
    },
]


SYSTEM_PROMPT = "You are a health assistant. Provide brief, safe guidance."
patient_message = "I have chest pain when I breathe deeply"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": patient_message},
]

for model_info in models:
    client = model_info["client"]
    model = model_info["model"]

    print(f"\n\n--- {model_info['name']} ---")
    start_time = time.time()
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.3  # type: ignore
    )
    end_time = time.time()
    time_taken = end_time - start_time

    assert (
        response.choices[0].message.content is not None
    ), "Model response content is missing."
    truncated_response = response.choices[0].message.content[:200]
    model_response_length = len(response.choices[0].message.content)

    print("\n\n--- Model Response ---")
    print(f"{truncated_response}...")
    print(f"Model {model_info['name']} time: {time_taken:.2f} seconds")
