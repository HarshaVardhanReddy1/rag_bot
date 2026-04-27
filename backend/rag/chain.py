from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import traceable

from backend.llm_model.model import get_llm
from backend.rag.loader import retrieve_relevant_docs

load_dotenv()

NO_DATA_FOUND_ANSWER = "No data found."
GENERIC_ERROR_ANSWER = "Something went wrong while processing your request."


@traceable(name="rag_prompt_creation")
def get_prompt(context: str, query: str, history_text: str = "") -> HumanMessage:
    history_block = ""
    if history_text:
        history_block = f"""
Conversation history:
{history_text}

Use conversation history only to understand follow-up questions or references like
"it", "that", or "the previous answer". Do not use conversation history as factual
evidence. The document context below is the only source of truth.
"""

    return HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""
You are a precise RAG assistant that answers from uploaded documents.

Primary rule:
Answer the user's question using only the document context provided below.

Answering rules:
- Start with the direct answer. Then add only the details needed to support it.
- Combine information across chunks when they clearly discuss the same topic.
- Preserve exact project names, methods, results, numbers, dates, tools, and technical terms.
- If the user asks "what is abstract", "abstract", or asks for another section title, treat it as a request to summarize or provide that section from the uploaded document.
- Do not use outside knowledge.
- If the document has no relevant info, reply exactly: {NO_DATA_FOUND_ANSWER}

Style:
- Keep answers concise and specific.

Retrieved document context:
{context}
{history_block}

User question:
{query}

Answer:
""",
            }
        ]
    )


@traceable(name="rag_context_formatting")
def format_docs(documents) -> str:
    formatted_docs = []
    for index, document in enumerate(documents, start=1):
        file_name = document.metadata.get("file_name") or "unknown source"
        formatted_docs.append(f"[Source {index}: {file_name}]\n{document.page_content}")
    return "\n\n".join(formatted_docs)


@traceable(name="rag_document_retrieval")
def retrieve_documents(question: str, user_id: str):
    return retrieve_relevant_docs(question, user_id)


@traceable(name="rag_llm_generation")
def generate_llm_response(llm, final_prompt):
    if isinstance(final_prompt, str):
        return llm.invoke(final_prompt)
    return llm.invoke([final_prompt])


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


def _build_empty_result() -> dict[str, Any]:
    return {
        "answer": NO_DATA_FOUND_ANSWER,
        "sources": [],
        "source_data": [],
    }


def _build_error_result(error: Exception) -> dict[str, Any]:
    return {
        "answer": GENERIC_ERROR_ANSWER,
        "sources": [],
        "source_data": [],
        "error": str(error),
    }


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
                return _build_empty_result()

            context = format_docs(documents)
            prompt = get_prompt(context, question, history_text)
            response = generate_llm_response(llm, prompt)
            return post_process_response(response, documents)

        except Exception as exc:
            return _build_error_result(exc)

    return rag_pipeline
