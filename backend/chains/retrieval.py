from langchain_core.runnables import Runnable, chain
from openai import OpenAI

from api import chat_completions
from retriever import get_retriever


client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


@chain
def get_retrieval_chain(inputs: dict) -> Runnable:
    retriever = get_retriever()
    prompt = inputs["input"]
    context = "".join([f"{document.page_content}\n\n" for document in retriever.invoke(prompt)])
    system_prompt = f"Answer any use questions based solely on the context below:\n\n{context}"
    return {"input": chat_completions(prompt, system_prompt)}
