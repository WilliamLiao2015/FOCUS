import json

from chains import chain
from utils import get_time


if __name__ == "__main__":
    results = chain.invoke({
        "log": [
            {
                "type": "query",
                "role": "user",
                "content": "What is FOCUS in document?",
                "time": get_time()
            }
        ]
    })

    with open(f"./logs/output {get_time()}.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(results["log"][-1]["content"])
