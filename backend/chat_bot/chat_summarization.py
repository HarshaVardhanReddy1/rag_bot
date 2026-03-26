from bson import ObjectId
from datetime import datetime, timezone

from backend.database import messages_collection, chats_collection
from backend.llm_model.model import get_llm


def summarize_if_needed(chat_id: str | ObjectId):
    chat_id = chat_id if isinstance(chat_id, ObjectId) else ObjectId(chat_id)

    total_messages = messages_collection.count_documents({
        "chat_id": chat_id
    })

    # summarize every 10 messages
    if total_messages % 10 != 0:
        return

    start = total_messages - 9

    messages = messages_collection.find(
        {"chat_id": chat_id}
    ).sort("created_at", 1).skip(start - 1).limit(10)

    doc = chats_collection.find_one(
        {"_id": chat_id},
        {"summary": 1, "_id": 0}
    )

    summary = doc.get("summary") if doc else ""

    chat_text = ""

    if summary:
        chat_text += f"Previous summary: {summary}\n"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        chat_text += f"{role}: {content}\n"

    prompt = f"""
You are an AI assistant.

Summarize the following conversation clearly and concisely.

Conversation:
{chat_text}

Instructions:

* Capture the main topic of the conversation
* Include key questions asked by the user
* Include important responses or outcomes
* Avoid unnecessary details or repetition
* Keep the summary short and easy to understand (3–5 sentences)
* Do not add any new information

Provide only the summary.

"""
    llm = get_llm()

    summary = llm.invoke(prompt).content.strip()

    chats_collection.update_one(
        {"_id": chat_id},
        {
            "$set": {
                "summary": summary,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )
