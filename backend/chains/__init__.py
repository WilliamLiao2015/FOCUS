from chains.retrieval import get_retrieval_chain
from chains.summarize import summarize_chain


chain = get_retrieval_chain | summarize_chain
