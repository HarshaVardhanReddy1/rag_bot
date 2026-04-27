from datetime import datetime, timezone

from bson import ObjectId

from backend.database import chats_collection, messages_collection
from backend.llm_model.model import get_llm

SUMMARY_BATCH_SIZE = 10


def summarize_if_needed(chat_id: str | ObjectId) -> None:
    chat_object_id = chat_id if isinstance(chat_id, ObjectId) else ObjectId(chat_id)

    total_messages = messages_collection.count_documents({"chat_id": chat_object_id})
    if total_messages % SUMMARY_BATCH_SIZE != 0:
        return

    messages = messages_collection.find({"chat_id": chat_object_id}).sort(
        "created_at",
        1,
    ).skip(total_messages - SUMMARY_BATCH_SIZE).limit(SUMMARY_BATCH_SIZE)

    chat_doc = chats_collection.find_one(
        {"_id": chat_object_id},
        {"summary": 1, "_id": 0},
    )
    previous_summary = chat_doc.get("summary") if chat_doc else ""

    conversation_lines: list[str] = []
    if previous_summary:
        conversation_lines.append(f"Previous summary: {previous_summary}")

    for message in messages:
        conversation_lines.append(f"{message['role']}: {message['content']}")

    prompt = f"""
You are an AI assistant.

Summarize the following conversation clearly and concisely.

Conversation:
{'\n'.join(conversation_lines)}

Instructions:

* Capture the main topic of the conversation
* Include key questions asked by the user
* Include important responses or outcomes
* Avoid unnecessary details or repetition
* Keep the summary short and easy to understand (3-5 sentences)
* Do not add any new information

Provide only the summary.
"""

    summary = get_llm().invoke(prompt).content.strip()
    chats_collection.update_one(
        {"_id": chat_object_id},
        {
            "$set": {
                "summary": summary,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
