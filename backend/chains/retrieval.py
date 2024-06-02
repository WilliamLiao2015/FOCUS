import os
import json

from langchain_core.runnables import Runnable, chain

from api import get_chat_completions
from utils import get_time
from retriever import get_retriever


@chain
def get_retrieval_chain(state: dict) -> Runnable:
    retriever = get_retriever()
    log = state["log"]
    query = log[-1]["content"]

    context = "".join([f"{document.page_content}\n\n" for document in retriever.invoke(query)])
    history = ""

    for filename in os.listdir("./logs"):
        if filename.startswith("output"):
            with open(f"./logs/{filename}", "r", encoding="utf-8") as fp:
                past_log = json.load(fp)["log"]
                summary = past_log[-1]
                history += f"{summary['time']}: {summary['content']}\n"

    system_context = f"Answer any questions based on the context below:\n\n{context}"
    system_history = f"You may also use the following past history to answer user's new questions:\n\n{history}"

    output = get_chat_completions([
        *log,
        {"role": "system", "content": system_context},
        {"role": "system", "content": system_history},
        {"role": "system", "content": f"When the context is different from the given history, you should respect what user said in the past, and try your best to give a new answer.\n\nCurrent time is: {get_time()}"}
    ])
    log.append({
        "type": "retrieval",
        "role": "assistant",
        "content": output,
        "time": get_time()
    })

    return {"log": log}
