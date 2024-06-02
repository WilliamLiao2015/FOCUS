import glob
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
from typing import List

sys.path.extend("../")
from model import SentenceTransformerModel

model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {"device": "cuda"}
default_embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


class MyEmbeddings:
    def __init__(self, ckpt):
        self.model = SentenceTransformerModel.load_from_checkpoint(ckpt)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model(t).tolist() for t in texts]
    
    def embed_query(self, query: str) -> List[float]:
            return self.model([query])


def get_retriever(doc_directory="./data", ckpt=None, use_saved=True):
    all_splits = []

    for filename in glob.iglob(doc_directory + "/**/*", recursive=True):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filename)
            PDF_data = loader.load()
            all_splits.extend(text_splitter.split_documents(PDF_data))
        elif filename.endswith(".txt") or filename.endswith(".md"):
            loader = TextLoader(filename, encoding="utf-8")
            text_data = loader.load()
            all_splits.extend(text_splitter.split_documents(text_data))
    if ckpt: embedding_model = MyEmbeddings(ckpt)
    else: embedding_model = default_embedding_model

    if use_saved:
        try: vectordb = Chroma(persist_directory="./data/vectordb", embedding_function=embedding_model)
        except: vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding_model, persist_directory="./data/vectordb")
    else: vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding_model, persist_directory="./data/vectordb")
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    return retriever


if __name__ == "__main__":
    print("\n".join([document.page_content for document in get_retriever().invoke("What is FOCUS?")]))
