from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from time import perf_counter
from datetime import datetime, timezone
from typing import Any

from backend.llm_model.model import get_llm
from backend.rag.logging_utils import log_execution_event, log_execution_time
from backend.rag.loader import retrieve_relevant_docs

load_dotenv()


# ---------------- PROMPT ----------------

@log_execution_time("prompt_creation")
def get_prompt(context, query, history_text=""):
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
- If the document has no relevant info, reply exactly: No data found.

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


# ---------------- FORMAT DOCS ----------------

@log_execution_time("context_formatting")
def format_docs(docs):
    formatted_docs = []

    for index, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name") or "unknown source"
        formatted_docs.append(
            f"[Source {index}: {file_name}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


# ---------------- RETRIEVAL WRAPPER ----------------

@log_execution_time("retrieve_documents")
def retrieve_documents(question: str, user_id: str):
    return retrieve_relevant_docs(question, user_id)


# ---------------- LLM ----------------

@log_execution_time("llm_generation")
def generate_llm_response(llm, final_prompt):
    if isinstance(final_prompt, str):
        return llm.invoke(final_prompt)

    return llm.invoke([final_prompt])


# ---------------- POST PROCESS ----------------

@log_execution_time("post_process_response")
def post_process_response(response, docs):
    source_data = [
        {
            "source": doc.metadata.get("source"),
            "file_name": doc.metadata.get("file_name"),
            "relevance_score": doc.metadata.get("relevance_score", None),
            "retrieval_reason": doc.metadata.get("retrieval_reason", None),
            "content": doc.page_content,
        }
        for doc in docs
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


# ---------------- RAG PIPELINE ----------------

def get_rag_chain():
    llm = get_llm()

    def rag_pipeline(question: str, user_id: str, history_text: str = "") -> dict[str, Any]:
        start_time = datetime.now(timezone.utc).isoformat()
        start_counter = perf_counter()

        try:
            docs = retrieve_documents(question, user_id)

            if not docs:
                result = {
                    "answer": "No data found.",  # ✅ consistent with prompt
                    "sources": [],
                    "source_data": [],
                }
                end_time = datetime.now(timezone.utc).isoformat()
                duration_ms = (perf_counter() - start_counter) * 1000
                log_execution_event("rag_pipeline", start_time, end_time, duration_ms, "success")
                return result

            context = format_docs(docs)
            final_prompt = get_prompt(context, question, history_text)
            response = generate_llm_response(llm, final_prompt)
            result = post_process_response(response, docs)

            end_time = datetime.now(timezone.utc).isoformat()
            duration_ms = (perf_counter() - start_counter) * 1000
            log_execution_event("rag_pipeline", start_time, end_time, duration_ms, "success")

            return result

        except Exception as e:
            end_time = datetime.now(timezone.utc).isoformat()
            duration_ms = (perf_counter() - start_counter) * 1000

            log_execution_event(
                "rag_pipeline",
                start_time,
                end_time,
                duration_ms,
                "exception",
                str(e),
            )

            return {
                "answer": "Something went wrong while processing your request.",
                "sources": [],
                "source_data": [],
                "error": str(e),
            }

    return rag_pipeline
