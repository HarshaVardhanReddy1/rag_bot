from langchain_core.messages import HumanMessage
from langsmith import traceable

NO_DATA_FOUND_ANSWER = "No data found."


# Build the final prompt that constrains the model to the retrieved documents.
@traceable(name="rag_prompt_creation")
def build_rag_prompt(
    context: str,
    query: str,
    history_text: str = "",
) -> HumanMessage:
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


# Convert retrieved documents into one readable context block for the model.
@traceable(name="rag_context_formatting")
def format_docs(documents) -> str:
    formatted_docs = []
    for index, document in enumerate(documents, start=1):
        file_name = document.metadata.get("file_name") or "unknown source"
        formatted_docs.append(f"[Source {index}: {file_name}]\n{document.page_content}")
    return "\n\n".join(formatted_docs)
