from langchain_core.messages import HumanMessage
from backend.rag.loader import get_retriever
from dotenv import load_dotenv
from backend.llm_model.model import get_llm
import os

load_dotenv()


 


def get_prompt(context,query):
    return HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
You are a helpful AI assistant.

Rules:
- provide the answer based on the provided context

Context:
{context}

Question:
{query}

Answer:
"""
                }
            ]
        )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def get_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
     

    def rag_pipeline(question: str):
        try:
            docs = retriever.invoke(question)

            # ✅ Handle no results
            if not docs:
                return {
                    "answer": "No relevant information found.",
                    "sources": []
                }

            context = format_docs(docs)

            final_prompt = get_prompt(context,question)

            response = llm.invoke([final_prompt])

            return {
                "answer": response.content,
                "sources": [doc.metadata for doc in docs]
            }

        except Exception as e:
            return {
                "answer": "Something went wrong while processing your request.",
                "error": str(e)
            }

    return rag_pipeline