from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.rag.config import CHUNK_OVERLAP, CHUNK_SIZE, SUPPORTED_EXTENSIONS


# Remove invalid bytes and null characters so downstream tools receive clean text.
def clean_document_text(value: str) -> str:
    if not value:
        return ""

    cleaned = value.encode("utf-8", errors="ignore").decode("utf-8")
    return cleaned.replace("\x00", "")


# Rebuild documents with cleaned text while keeping their metadata intact.
def clean_documents(documents: list[Document]) -> list[Document]:
    return [
        Document(
            page_content=clean_document_text(doc.page_content),
            metadata=dict(doc.metadata),
        )
        for doc in documents
    ]


# Create the text splitter used to break uploaded files into retrieval-sized chunks.
def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


# Load one supported file into a single normalized document for ingestion.
def load_file_as_document(file_path: Path) -> Document:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    docs = clean_documents(loader.load())
    return Document(
        page_content="\n".join(d.page_content for d in docs),
        metadata={"source": str(file_path), "file_name": file_path.name},
    )


# Split one document into smaller chunks so Pinecone retrieval stays precise.
def split_document_into_chunks(document: Document) -> list[Document]:
    return create_text_splitter().split_documents([document])
