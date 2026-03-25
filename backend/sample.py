from rag.loader import get_retriever 
from llm_model.model import get_llm
from langchain_core.messages import HumanMessage  # ✅ correct import


def chat():
    retriever = get_retriever()
    llm = get_llm()

    print("\n💬 RAG Chat Started (type 'exit' to quit)\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)
        

        if not docs:
            print("Bot: No relevant information found.\n")
            continue

        context = "\n\n".join([doc.page_content for doc in docs])
        print("context:",context)

        # ✅ fixed prompt
        message = HumanMessage(
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

        # ✅ pass message properly
        response = llm.invoke([message])

        print(f"\nBot: {response.content}\n")


# 🔹 Run
 