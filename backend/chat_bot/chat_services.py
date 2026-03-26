from backend.chat_bot.chat_summarization import summarize_if_needed
from backend.rag.chain import get_rag_chain
from backend.chat_bot.chat_utils import send_message, _get_owned_chat, _build_history_text
from backend.database import chats_collection, messages_collection
from datetime import datetime, timezone

rag_chain = get_rag_chain()

def get_chat_response(query: str,chat_id: str,user: dict):
    chat_doc = _get_owned_chat(chat_id, user)
    chat_id_obj = chat_doc["_id"]
    summary = chat_doc.get("summary", "")

    history_text = _build_history_text(chat_id_obj, summary)
    send_message("user", query, chat_id_obj)
    result = rag_chain(query, history_text)
    send_message("assistant", result["answer"], chat_id_obj)
    summarize_if_needed(chat_id_obj)
    chats_collection.update_one(
        {"_id": chat_id_obj},
        {"$set": {"updated_at": datetime.now(timezone.utc)}},
    )

    return result





def generate_new_chat(data, user):
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": str(user["_id"]),
        "title": data.title.strip(),
        "summary": "",
        "created_at": now,
        "updated_at": now,
    }

    result = chats_collection.insert_one(doc)

    return {
        "chat_id": str(result.inserted_id),
        "title": doc["title"],
    }


def get_chat_list(user):
    chats = chats_collection.find(
        {"user_id": str(user["_id"])},
        {"title": 1, "summary": 1, "updated_at": 1},
    ).sort("updated_at", -1)

    return [
        {
            "chat_id": str(chat["_id"]),
            "title": chat.get("title"),
            "summary": chat.get("summary"),
            "updated_at": chat.get("updated_at"),
        }
        for chat in chats
    ]


def get_chat_messages(chat_id: str, user: dict):
    chat_doc = _get_owned_chat(chat_id, user)

    messages = messages_collection.find({"chat_id": chat_doc["_id"]}).sort(
        "created_at", 1
    )

    return {
        "messages": [
            {
                "role": message["role"],
                "content": message["content"],
                "image_url": message.get("image_url"),
                "created_at": message["created_at"],
            }
            for message in messages
        ]
    }
