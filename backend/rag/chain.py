from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from time import perf_counter
from datetime import datetime, timezone

from backend.llm_model.model import get_llm
from backend.rag.logging_utils import log_execution_event, log_execution_time
from backend.rag.loader import retrieve_relevant_documents

load_dotenv()


@log_execution_time("prompt_creation")
def get_prompt(context, query, history_text=""):
    history_block = ""
    if history_text:
        history_block = f"\nConversation History:\n{history_text}\n"

    return HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""
You are a helpful AI assistant.

Rules:
- Answer only if the provided context contains the actual answer to the question.
- Do not use outside knowledge.
- Do not guess or infer an answer that is not directly supported by the context.
- If the context does not contain the answer, reply exactly: No data found.

Context:
{context}
{history_block}

Question:
{query}

Answer:
""",
            }
        ]
    )


@log_execution_time("context_formatting")
def format_docs(docs):
    result = "\n\n".join([doc.page_content for doc in docs])
    return result


@log_execution_time("retrieve_documents")
def retrieve_documents(question: str):
    return retrieve_relevant_documents(question)


@log_execution_time("llm_generation")
def generate_llm_response(llm, final_prompt):
    if isinstance(final_prompt, str):
        return llm.invoke(final_prompt)

    return llm.invoke([final_prompt])


@log_execution_time("post_process_response")
def post_process_response(response, docs):
    source_data = [
        {
            "source": doc.metadata.get("source"),
            "file_name": doc.metadata.get("file_name"),
            "page": doc.metadata.get("page"),
            "relevance_score": doc.metadata.get("relevance_score"),
            "retrieval_reason": doc.metadata.get("retrieval_reason"),
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


def get_rag_chain():
    llm = get_llm()

    def rag_pipeline(question: str, history_text: str = ""):
        start_time = datetime.now(timezone.utc).isoformat()
        start_counter = perf_counter()

        try:
            docs = retrieve_documents(question)

            if not docs:
                result = {
                    "answer": "No relevant information found.",
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
