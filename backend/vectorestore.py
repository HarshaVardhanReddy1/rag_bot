import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

# =========================
# CONFIG
# =========================
FOLDERS = ["docs", "uploads/docs"]
CHROMA_DB_PATH = "chroma_db"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

 

# =========================
# STEP 1: DELETE OLD DB
# =========================
if os.path.exists(CHROMA_DB_PATH):
    shutil.rmtree(CHROMA_DB_PATH)
    print("Old Chroma DB deleted")

# =========================
# STEP 2: LOAD ALL PDFs
# =========================
documents = []

for folder in FOLDERS:
    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder, file)
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file_path

            documents.extend(docs)

print(f"Loaded {len(documents)} pages from all PDFs")

# =========================
# STEP 3: SPLIT INTO CHUNKS
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# =========================
# STEP 4: CREATE EMBEDDINGS
# =========================
embedding = HuggingFaceEndpointEmbeddings(
     repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
)

# =========================
# STEP 5: CREATE CHROMA DB
# =========================
vectorstore =  Chroma(
        collection_name="rag_documents",
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding,
    )
 

print("✅ Chroma DB created successfully!")