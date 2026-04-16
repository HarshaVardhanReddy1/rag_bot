from datetime import datetime, timezone

from bson import ObjectId
from fastapi import HTTPException, status

from backend.database import chats_collection, messages_collection


def send_message(role, content, chat_id, *, sources=None, source_data=None):
    chat_id_obj = chat_id if isinstance(chat_id, ObjectId) else ObjectId(chat_id)

    doc = {
        "chat_id": chat_id_obj,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc),
    }

    if sources is not None:
        doc["sources"] = sources
    if source_data is not None:
        doc["source_data"] = source_data

    messages_collection.insert_one(doc)

    return doc


def _get_owned_chat(chat_id: str, user: dict):
    try:
        chat_id_obj = ObjectId(chat_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat id",
        ) from exc

    chat_doc = chats_collection.find_one(
        {"_id": chat_id_obj, "user_id": str(user["_id"])}
    )

    if not chat_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    return chat_doc


def _build_history_text(chat_id_obj: ObjectId, summary: str) -> str:
    messages = messages_collection.find({"chat_id": chat_id_obj}).sort(
        "created_at", -1
    ).limit(10)

    history_text = ""

    if summary:
        history_text += f"Older Conversation Summary:\n{summary}\n\n"

    for msg in list(messages)[::-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    return history_text
