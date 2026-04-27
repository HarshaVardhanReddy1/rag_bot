from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.rag.documents import load_file_as_document, split_document_into_chunks
from backend.rag.retriever import create_hybrid_retriever


# Copy retrieval scores into a predictable metadata field for later API responses.
def enrich_retrieved_documents(documents: list[Document]) -> list[Document]:
    enriched_documents: list[Document] = []

    for document in documents:
        metadata = dict(document.metadata)
        score = metadata.get("relevance_score")
        if score is None:
            score = metadata.get("score")

        metadata["relevance_score"] = (
            float(score) if isinstance(score, (int, float)) else None
        )

        enriched_documents.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )

    return enriched_documents


# Read an uploaded file, chunk it, and store it in the vector index for one user.
def ingest_uploaded_file(file_path: Path, uploaded_by: Any = None) -> dict[str, str]:
    document = load_file_as_document(file_path)
    if uploaded_by:
        document.metadata["uploaded_by"] = str(uploaded_by)

    chunks = split_document_into_chunks(document)
    retriever = create_hybrid_retriever()
    retriever.add_texts(
        [chunk.page_content for chunk in chunks],
        metadatas=[
            {
                "file_name": chunk.metadata.get("file_name"),
                "source": chunk.metadata.get("source"),
                "user_id": str(uploaded_by),
            }
            for chunk in chunks
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
