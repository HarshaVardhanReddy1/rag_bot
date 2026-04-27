from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from fastapi import HTTPException, status

from backend.database import chats_collection, messages_collection


def send_message(
    role: str,
    content: str,
    chat_id: str | ObjectId,
    *,
    sources: list[str] | None = None,
    source_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chat_object_id = chat_id if isinstance(chat_id, ObjectId) else ObjectId(chat_id)

    message_doc = {
        "chat_id": chat_object_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc),
    }
    if sources is not None:
        message_doc["sources"] = sources
    if source_data is not None:
        message_doc["source_data"] = source_data

    messages_collection.insert_one(message_doc)
    return message_doc


def _get_owned_chat(chat_id: str, user: dict[str, Any]) -> dict[str, Any]:
    try:
        chat_object_id = ObjectId(chat_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat id",
        ) from exc

    chat_doc = chats_collection.find_one(
        {"_id": chat_object_id, "user_id": str(user["_id"])}
    )
    if not chat_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    return chat_doc


def _build_history_text(chat_id: ObjectId, summary: str) -> str:
    recent_messages = list(
        messages_collection.find({"chat_id": chat_id})
        .sort("created_at", -1)
        .limit(10)
    )[::-1]

    history_lines: list[str] = []
    if summary:
        history_lines.append(f"Older Conversation Summary:\n{summary}")

    for message in recent_messages:
        role = "User" if message["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {message['content']}")

    return "\n".join(history_lines)
