import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.rag.documents import load_file_as_document, split_document_into_chunks
from backend.rag.retriever import create_hybrid_retriever


def _build_upload_identity(file_path: Path, uploaded_by: Any) -> str:
    raw_identity = f"{uploaded_by}:{file_path.resolve()}:{file_path.stat().st_size}"
    return hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()[:16]


def _build_chunk_id(upload_identity: str, chunk_index: int) -> str:
    return f"{upload_identity}:chunk:{chunk_index}"


# Copy retrieval scores into a predictable metadata field for later API responses.
def enrich_retrieved_documents(documents: list[Document]) -> list[Document]:
    for document in documents:
        score = document.metadata.get("relevance_score")
        if score is None:
            score = document.metadata.get("score")

        document.metadata["relevance_score"] = (
            float(score) if isinstance(score, (int, float)) else None
        )

        chunk_index = document.metadata.get("chunk_index")
        document.metadata["chunk_index"] = (
            int(chunk_index) if isinstance(chunk_index, (int, float, str)) else None
        )

    return documents


# Read an uploaded file, chunk it, and store it in the vector index for one user.
def ingest_uploaded_file(file_path: Path, uploaded_by: Any = None) -> dict[str, str]:
    if uploaded_by is None:
        raise ValueError("uploaded_by is required to index user-specific documents.")

    document = load_file_as_document(file_path)
    document.metadata["uploaded_by"] = str(uploaded_by)

    chunks = split_document_into_chunks(document)
    upload_identity = _build_upload_identity(file_path, uploaded_by)
    chunk_ids = [_build_chunk_id(upload_identity, index) for index, _ in enumerate(chunks)]
    retriever = create_hybrid_retriever()
    retriever.add_texts(
        [chunk.page_content for chunk in chunks],
        ids=chunk_ids,
        metadatas=[
            {
                "chunk_id": chunk_ids[index],
                "chunk_index": index,
                "file_name": chunk.metadata.get("file_name"),
                "source": chunk.metadata.get("source"),
                "user_id": str(uploaded_by),
            }
            for index, chunk in enumerate(chunks)
        ],
    )

    return {
        "file_name": file_path.name,
        "source": str(file_path),
    }


# Fetch the most relevant indexed chunks for a user's question.
def retrieve_relevant_docs(query: str, user_id: str) -> list[Document]:
    retriever = create_hybrid_retriever()
    documents = retriever.invoke(
        query,
        filter={"user_id": str(user_id)},
    )
    return enrich_retrieved_documents(documents)
