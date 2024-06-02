from langchain import hub
from langchain_core.runnables import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAI

from retriever import get_retriever


retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
llm = OpenAI(base_url="http://localhost:1234/v1")


def get_stuff_documents_chain() -> Runnable:
    return create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)

def get_retrieval_chain() -> Runnable:
    return create_retrieval_chain(retriever=get_retriever(), combine_docs_chain=get_stuff_documents_chain())
