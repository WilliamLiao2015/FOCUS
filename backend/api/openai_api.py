from openai import OpenAI


client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def chat_completions(prompt: str, system_prompt: str = None) -> str:
    messages = [] if not system_prompt else [{"role": "system", "content": system_prompt}]
    messages = messages + [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-BPE-fix-GGUF",
        messages=messages
    )
    return completion.choices[0].message.content
