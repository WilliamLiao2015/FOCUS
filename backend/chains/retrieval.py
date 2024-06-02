from langchain_core.runnables import Runnable
from openai import OpenAI

from retriever import get_retriever


client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def get_retrieval_chain(input) -> Runnable:
    retriever = get_retriever()
    context = "".join([f"{document.page_content}\n\n" for document in retriever.invoke(input["input"])])
    system_prompt = f"Answer any use questions based solely on the context below:\n\n{context}"
    completion = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-BPE-fix-GGUF",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input["input"]}
        ],
        temperature=0.7
    )
    return {"input": completion.choices[0].message.content}
