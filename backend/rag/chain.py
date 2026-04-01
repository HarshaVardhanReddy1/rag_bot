from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from time import perf_counter
from datetime import datetime, timezone

from backend.llm_model.model import get_llm
from backend.rag.logging_utils import log_execution_event, log_execution_time
from backend.rag.loader import get_retriever

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
- provide the answer based on the provided context

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
    retriever = get_retriever()
    return retriever.invoke(question)


@log_execution_time("rerank_documents")
def rerank_documents(question: str, docs):
    # Placeholder for future reranking logic. Logging this step now keeps
    # latency tracing stable when a dedicated reranker is introduced later.
    return docs


@log_execution_time("llm_generation")
def generate_llm_response(llm, final_prompt):
    return llm.invoke([final_prompt])


@log_execution_time("post_process_response")
def post_process_response(response, docs):
    return {
        "answer": response.content,
        "sources": [doc.metadata.get("source") for doc in docs],
    }


def get_rag_chain():
    llm = get_llm()

    def rag_pipeline(question: str, history_text: str = ""):
        start_time = datetime.now(timezone.utc).isoformat()
        start_counter = perf_counter()

        try:
            docs = retrieve_documents(question)
            docs = rerank_documents(question, docs)

            if not docs:
                result = {
                    "answer": "No relevant information found.",
                    "sources": [],
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
                "error": str(e),
            }

    return rag_pipeline
