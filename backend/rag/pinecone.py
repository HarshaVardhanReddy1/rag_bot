import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "rag-hybrid-index")

if not api_key:
    raise ValueError("Missing Pinecone API key.")

pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

description = pc.describe_index(index_name)
if description.metric != "dotproduct":
    raise ValueError(f"Pinecone index '{index_name}' must use metric='dotproduct'.")

if description.dimension != 384:
    raise ValueError(f"Pinecone index '{index_name}' must use dimension=384.")

while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

index = pc.Index(index_name)
