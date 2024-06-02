import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
from typing import List


class MyEmbeddings:
    def __init__(self, model=None, state_dict_path=None):
        if not model: model = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model, trust_remote_code=True)
        if state_dict_path: self.model.load_state_dict(state_dict_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    
    def embed_query(self, query: str) -> List[float]:
            return self.model.encode([query])


def get_retriever(doc_directory="./test_directory", model_name=None, state_dict_path=None):
    all_splits = []

    for filename in os.listdir(doc_directory):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(doc_directory, filename))
            PDF_data = loader.load()
            all_splits.extend(text_splitter.split_documents(PDF_data))
        else:
            loader = TextLoader(os.path.join(doc_directory, filename), encoding="utf-8")
            text_data = loader.load()
            all_splits.extend(text_splitter.split_documents(text_data))
    if not model_name and not state_dict_path:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {"device": "cpu"}
        embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    else:
        embedding = MyEmbeddings(model_name, state_dict_path)

    vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding)
    retriever = vectordb.as_retriever()
    return retriever


if __name__ == "__main__":
    print("\n".join([document.page_content for document in get_retriever().invoke("What is FOCUS in document?")]))
