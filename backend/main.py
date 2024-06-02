import json

from chains import get_chain
from utils import get_time


if __name__ == "__main__":
    chain = get_chain()
    results = chain.invoke({
        "log": [
            {
                "type": "query",
                "role": "user",
                "content": "What is FOCUS?", # How old is Alison Hawk?
                "time": get_time()
            }
        ]
    })

    with open(f"./logs/output {get_time().replace(':', '-')}.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(results["log"][-1]["content"])
