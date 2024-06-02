import json

from datetime import datetime

from chains import chain


if __name__ == "__main__":
    results = chain.invoke({
        "log": [
            {
                "type": "query",
                "role": "user",
                "content": "What is FOCUS in document?",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ]
    })

    with open(f"./logs/output {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(results["log"][-1]["content"])
