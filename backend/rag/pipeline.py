from typing import Any

from langsmith import traceable

from backend.llm_model.model import get_llm
from backend.rag.prompts import NO_DATA_FOUND_ANSWER, build_rag_prompt, format_docs
from backend.rag.services import retrieve_relevant_docs

GENERIC_ERROR_ANSWER = "Something went wrong while processing your request."


# Send either a string prompt or message prompt to the chat model.
@traceable(name="rag_llm_generation")
def generate_llm_response(llm, final_prompt):
    if isinstance(final_prompt, str):
        return llm.invoke(final_prompt)
    return llm.invoke([final_prompt])


# Shape the final API payload with answer text and document source details.
@traceable(name="rag_post_process")
def post_process_response(response, documents) -> dict[str, Any]:
    source_data = []
    sources: list[str] = []
    seen_sources: set[str] = set()

    for document in documents:
        source = document.metadata.get("source")
        source_data.append(
            {
                "chunk_id": document.metadata.get("chunk_id"),
                "chunk_index": document.metadata.get("chunk_index"),
                "source": source,
                "file_name": document.metadata.get("file_name"),
                "relevance_score": document.metadata.get("relevance_score"),
                "content": document.page_content,
            }
        )
        if source and source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)

    return {
        "answer": response.content,
        "sources": sources,
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
            documents = retrieve_relevant_docs(question, user_id)
            if not documents:
                return build_empty_result()

            context = format_docs(documents)
            prompt = build_rag_prompt(context, question, history_text)
            response = generate_llm_response(llm, prompt)
            return post_process_response(response, documents)
        except Exception as exc:
            return build_error_result(exc)

    return rag_pipeline
