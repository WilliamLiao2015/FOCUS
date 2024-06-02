from chains import get_retrieval_chain


if __name__ == "__main__":
    input = {"input": "What is FOCUS in document?"}
    result = get_retrieval_chain(input)
    print(result["input"])
