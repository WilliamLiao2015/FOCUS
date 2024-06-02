from chains import chain
from langchain_core.output_parsers import StrOutputParser


if __name__ == "__main__":
    chain = chain | StrOutputParser()
    print(chain.invoke({"input": "What is FOCUS?"}))
