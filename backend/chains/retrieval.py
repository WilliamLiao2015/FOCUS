from datetime import datetime

from langchain_core.runnables import Runnable, chain

from api import get_chat_completions
from retriever import get_retriever


@chain
def get_retrieval_chain(state: dict) -> Runnable:
    retriever = get_retriever()
    log = state["log"]
    query = log[-1]["content"]
    context = "".join([f"{document.page_content}\n\n" for document in retriever.invoke(query)])
    system_prompt = f"Answer any use questions based solely on the context below:\n\n{context}"
    output = get_chat_completions([
        *log,
        {"role": "system", "content": system_prompt}
    ])
    log.append({
        "type": "retrieval",
        "role": "assistant",
        "content": output,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return {"log": log}
