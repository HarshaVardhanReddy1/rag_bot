from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from backend.settings import settings
from dotenv import load_dotenv
import os
load_dotenv()

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        temperature=0.3
    )
    return ChatHuggingFace(llm=llm)