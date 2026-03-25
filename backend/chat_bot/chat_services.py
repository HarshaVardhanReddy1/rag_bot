from backend.rag.chain import get_rag_chain

rag_chain = get_rag_chain()

def get_chat_response(query: str):
    return rag_chain(query)