import json

from datetime import datetime

from chains import chain


if __name__ == "__main__":
    results = chain.invoke({
        "messages": [
            {"role": "user","content": "What is FOCUS in document?"}
        ],
        "log": []
    })

    with open(f"./logs/output {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(results["messages"][-1]["content"])
