import os

from openai import OpenAI


client = OpenAI(base_url="https://api.openai.com/v1", api_key=os.environ.get("OPENAI_API_KEY"))


def get_chat_completions(messages: list) -> str:
    messages = [message for message in messages if message["content"] is not None]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return completion.choices[0].message.content
