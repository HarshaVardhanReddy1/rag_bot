from typing import Any

from dotenv import load_dotenv
from langsmith import traceable

from backend.llm_model.model import get_llm
from backend.rag.prompts import NO_DATA_FOUND_ANSWER, build_rag_prompt, format_docs
from backend.rag.services import retrieve_relevant_docs

load_dotenv()

GENERIC_ERROR_ANSWER = "Something went wrong while processing your request."


# Retrieve matching documents for one question and user.
@traceable(name="rag_document_retrieval")
def retrieve_documents(question: str, user_id: str):
    return retrieve_relevant_docs(question, user_id)


# Send either a string prompt or message prompt to the chat model.
@traceable(name="rag_llm_generation")
def generate_llm_response(llm, final_prompt):
    if isinstance(final_prompt, str):
        return llm.invoke(final_prompt)
    return llm.invoke([final_prompt])


# Shape the final API payload with answer text and document source details.
@traceable(name="rag_post_process")
def post_process_response(response, documents) -> dict[str, Any]:
    source_data = [
        {
            "source": document.metadata.get("source"),
            "file_name": document.metadata.get("file_name"),
            "relevance_score": document.metadata.get("relevance_score"),
            "content": document.page_content,
        }
        for document in documents
    ]

    return {
        "answer": response.content,
        "sources": list(
            dict.fromkeys(
                source["source"] for source in source_data if source.get("source")
            )
        ),
        "source_data": source_data,
    }


# Return the standard empty response when nothing relevant is retrieved.
def build_empty_result() -> dict[str, Any]:
    return {
        "answer": NO_DATA_FOUND_ANSWER,
        "sources": [],
        "source_data": [],
    }


# Return a safe error payload instead of letting request handling fail hard.
def build_error_result(error: Exception) -> dict[str, Any]:
    return {
        "answer": GENERIC_ERROR_ANSWER,
        "sources": [],
        "source_data": [],
        "error": str(error),
    }


# Assemble the full RAG flow so callers can treat it like one service function.
def get_rag_chain():
    llm = get_llm()

    @traceable(name="rag_pipeline")
    def rag_pipeline(
        question: str,
        user_id: str,
        history_text: str = "",
    ) -> dict[str, Any]:
        try:
            documents = retrieve_documents(question, user_id)
            if not documents:
                return build_empty_result()

            context = format_docs(documents)
            prompt = build_rag_prompt(context, question, history_text)
            response = generate_llm_response(llm, prompt)
            return post_process_response(response, documents)
        except Exception as exc:
            return build_error_result(exc)

    return rag_pipeline
