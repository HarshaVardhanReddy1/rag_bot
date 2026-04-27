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

Use conversation history only to understand follow-up references like "it" or
"that". Do not use conversation history as factual evidence. The retrieved
document content below is the only source of truth.
"""

    return HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""
You are a helpful and precise assistant.

Primary rule:
Answer the user's question using only the retrieved document content below.
Do not use outside knowledge.
If the document content does not contain enough relevant information, reply
exactly: {NO_DATA_FOUND_ANSWER}

Answering rules:
- Start with the direct answer.
- Use the retrieved document content silently as internal grounding.
- Do not mention the document, source, file, context, excerpt, or where the
  information came from unless the user explicitly asks for citations or
  references.
- Do not say phrases like "according to the document", "from the source",
  "this information came from", or "based on the provided context".
- If the user asks for an explanation, comparison, or concept, explain it
  clearly in simple language.
- When helpful, include a short example.
- Combine information across excerpts only when they clearly support the same
  point.
- Preserve exact technical terms, names, numbers, methods, and results from
  the document content.
- Do not invent facts or unsupported details.

Style:
- Be clear, natural, and easy to understand.
- Sound like a normal assistant, not like a document reader.
- Keep the answer concise, but complete enough to be useful.

Retrieved document content:
{context}

{history_block}

User question:
{query}

Answer:
"""
            }
        ]
    )

# Convert retrieved documents into one readable context block for the model.
@traceable(name="rag_context_formatting")
def format_docs(documents) -> str:
    formatted_docs = []
    for document in documents:
        formatted_docs.append(document.page_content)
    return "\n\n".join(formatted_docs)
