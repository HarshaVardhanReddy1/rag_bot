from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from backend.llm_model.model import get_llm
from backend.rag.loader import get_retriever

load_dotenv()


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


def format_docs(docs):
    result ="\n\n".join([doc.page_content for doc in docs])
    print(result)
    return result


def get_rag_chain():
    llm = get_llm()

    def rag_pipeline(question: str, history_text: str = ""):
        try:
            retriever = get_retriever()
            docs = retriever.invoke(question)

            if not docs:
                return {
                    "answer": "No relevant information found.",
                    "sources": [],
                }

            context = format_docs(docs)
            final_prompt = get_prompt(context, question, history_text)
            response = llm.invoke([final_prompt])

            return {
                "answer": response.content,
                "sources": [doc.metadata.get("source") for doc in docs],
            }

        except Exception as e:
            return {
                "answer": "Something went wrong while processing your request.",
                "error": str(e),
            }

    return rag_pipeline
