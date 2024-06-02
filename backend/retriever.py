import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def get_retriever(doc_directory="./test_directory"):
    all_splits = []

    for pdf_file in os.listdir(doc_directory):
        loader = PyMuPDFLoader(os.path.join(doc_directory, pdf_file))
        PDF_data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_splits.extend(text_splitter.split_documents(PDF_data))

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    persist_directory = "db"
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)
    retriever = vectordb.as_retriever()
    return retriever


if __name__ == "__main__":
    print(get_retriever())
