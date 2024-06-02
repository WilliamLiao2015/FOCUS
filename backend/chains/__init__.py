from chains.retrieval import get_retrieval_chain
from chains.summarize import summarize_chain


def get_chain():
    return get_retrieval_chain | summarize_chain
