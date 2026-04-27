import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

DOCS_DIR = Path("docs")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
HYBRID_TOP_K = 5
HYBRID_ALPHA = 0.5
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
EMBEDDING_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"


# Build the shared embedding model once so every retriever uses the same setup.
def create_embedding_model() -> HuggingFaceEndpointEmbeddings:
    return HuggingFaceEndpointEmbeddings(
        repo_id=EMBEDDING_REPO_ID,
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
    )


embedding_model = create_embedding_model()
