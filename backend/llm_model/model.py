import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

DEFAULT_LLM_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TEMPERATURE = 0.3


def get_llm() -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id=DEFAULT_LLM_REPO_ID,
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        temperature=DEFAULT_TEMPERATURE,
    )
    return ChatHuggingFace(llm=endpoint)
