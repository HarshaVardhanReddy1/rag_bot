import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path("docs")
VECTOR_DB_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def sanitize_text(value: str):
    if not value:
        return ""

    # Some PDFs extract invalid surrogate code points that fail during UTF-8 encoding.
    cleaned = value.encode("utf-8", errors="ignore").decode("utf-8")
    return cleaned.replace("\x00", "")


def sanitize_documents(documents: list[Document]):
    sanitized = []

    for document in documents:
        sanitized.append(
            Document(
                page_content=sanitize_text(document.page_content),
                metadata=dict(document.metadata),
            )
        )

    return sanitized


def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
    )


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )


def is_supported_document(file_path: Path):
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_document(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    documents = sanitize_documents(loader.load())

    for document in documents:
        document.metadata["source"] = str(file_path)
        document.metadata["file_name"] = file_path.name

    return documents


def load_documents(path: Path):
    documents = []

    if not path.exists():
        return documents

    for file_path in path.iterdir():
        if file_path.is_file() and is_supported_document(file_path):
            documents.extend(load_document(file_path))

    return documents


def get_vectorstore():
    embeddings = get_embeddings()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )


def split_documents(documents):
    if not documents:
        return []

    splitter = get_text_splitter()
    return sanitize_documents(splitter.split_documents(documents))


def seed_vectorstore_from_docs():
    vectorstore = get_vectorstore()

    if os.path.exists(VECTOR_DB_DIR) and vectorstore._collection.count() > 0:
        return vectorstore

    documents = load_documents(DOCS_DIR)
    chunks = split_documents(documents)

    if chunks:
        vectorstore.add_documents(chunks)

    return vectorstore


def ingest_file(file_path: Path, uploaded_by=None):
    documents = load_document(file_path)

    for document in documents:
        if uploaded_by:
            document.metadata["uploaded_by"] = str(uploaded_by)

    chunks = split_documents(documents)
    vectorstore = get_vectorstore()

    if chunks:
        vectorstore.add_documents(chunks)

    return {
        "documents_added": len(documents),
        "chunks_added": len(chunks),
        "file_name": file_path.name,
        "source": str(file_path),
    }


def get_retriever():
    vectorstore = seed_vectorstore_from_docs()

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
