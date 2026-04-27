import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder

from backend.rag.pinecone import index

load_dotenv()

DOCS_DIR = Path("docs")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
HYBRID_TOP_K = 5
HYBRID_ALPHA = 0.5
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100


embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_API_KEY"),
)

_default_sparse_encoder: BM25Encoder | None = None


def clean_document_text(value: str) -> str:
    if not value:
        return ""

    cleaned = value.encode("utf-8", errors="ignore").decode("utf-8")
    return cleaned.replace("\x00", "")


def clean_documents(documents: list[Document]) -> list[Document]:
    return [
        Document(
            page_content=clean_document_text(doc.page_content),
            metadata=dict(doc.metadata),
        )
        for doc in documents
    ]


def _enrich_retrieved_documents(documents: list[Document]) -> list[Document]:
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


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


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


def split_document_into_chunks(document: Document) -> list[Document]:
    return create_text_splitter().split_documents([document])


def get_sparse_encoder() -> BM25Encoder:
    global _default_sparse_encoder

    if _default_sparse_encoder is None:
        _default_sparse_encoder = BM25Encoder.default()

    return _default_sparse_encoder


def create_hybrid_retriever() -> PineconeHybridSearchRetriever:
    return PineconeHybridSearchRetriever(
        embeddings=embedding_model,
        sparse_encoder=get_sparse_encoder(),
        index=index,
        alpha=HYBRID_ALPHA,
        top_k=HYBRID_TOP_K,
    )


def ingest_uploaded_file(file_path: Path, uploaded_by=None):
    doc = load_file_as_document(file_path)

    if uploaded_by:
        doc.metadata["uploaded_by"] = str(uploaded_by)

    chunks = split_document_into_chunks(doc)
    texts = [chunk.page_content for chunk in chunks]
    retriever = create_hybrid_retriever()

    retriever.add_texts(
        texts,
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


def retrieve_relevant_docs(query: str, user_id: str):
    retriever = create_hybrid_retriever()
    documents = retriever.invoke(
        query,
        filter={"user_id": str(user_id)},
    )
    return _enrich_retrieved_documents(documents)
