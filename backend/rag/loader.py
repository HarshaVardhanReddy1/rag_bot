from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

def load_documents(path):
    documents = []

    for file in path.iterdir():
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
            documents.extend(loader.load())

    return documents


def get_retriever():
     
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )

    if os.path.exists("chroma_db"):
        print("Loading existing DB...")
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
    else:
        print("Creating new DB...")

        path = Path("docs")
        documents = load_documents(path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


 