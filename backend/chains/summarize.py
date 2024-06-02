from langchain_core.runnables import Runnable, chain

from api import get_chat_completions
from utils import get_time


@chain
def summarize_chain(state: dict) -> Runnable:
    log = state["log"]
    conversation = [f"{message['role']}: {message['content']}" for message in log]
    system_prompt = f"Summarize the following conversation, specifying what do the user and assistant say, respectively."
    output = get_chat_completions([
        *log,
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(conversation)}
    ])
    log.append({
        "type": "summarize",
        "role": "assistant",
        "content": output,
        "time": get_time()
    })
    return {"log": log}
