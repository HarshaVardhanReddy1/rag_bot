import json
import re
from datetime import datetime, timezone

from backend.chat_bot.chat_summarization import summarize_if_needed
from backend.chat_bot.chat_utils import (
    _build_history_text,
    _get_owned_chat,
    send_message,
)
from backend.database import chats_collection, messages_collection
from backend.llm_model.model import get_llm
from backend.rag.chain import generate_llm_response, get_rag_chain

rag_chain = get_rag_chain()


def _format_source_data_for_validation(source_data: list[dict]) -> str:
    if not source_data:
        return "No source data was retrieved."

    formatted_sources = []

    for index, source in enumerate(source_data, start=1):
        metadata = []

        if source.get("file_name"):
            metadata.append(f"file_name={source['file_name']}")
        if source.get("source"):
            metadata.append(f"source={source['source']}")
        if source.get("page") is not None:
            metadata.append(f"page={source['page']}")
        if source.get("relevance_score") is not None:
            metadata.append(f"relevance_score={source['relevance_score']:.3f}")
        if source.get("retrieval_reason"):
            metadata.append(f"retrieval_reason={source['retrieval_reason']}")

        metadata_text = ", ".join(metadata) if metadata else "metadata unavailable"
        content = str(source.get("content") or "").strip()
        formatted_sources.append(f"Source {index} ({metadata_text}):\n{content}")

    return "\n\n".join(formatted_sources)


def _validation_prompt(query, answer, source_data):
    source_block = _format_source_data_for_validation(source_data)

    return f"""
You are an AI Validator (Auditor). Your task is to strictly evaluate an AI-generated response.

Question:
{query}

Response:
{answer}

Source Data:
{source_block}

Evaluate the response using the following criteria:

1. Accuracy (0-10)
- 0 = completely incorrect or misleading
- 5 = partially correct or somewhat useful
- 10 = fully correct and appropriate
- Use the source data as the ground truth for factual claims.
- If no source data was retrieved, penalize unsupported factual claims.
- If the response correctly refuses harmful/illegal requests, accuracy = 10

2. Relevance (0-10)
- 0 = completely unrelated
- 5 = partially related
- 10 = directly answers or appropriately addresses the question
- Refusal responses can still be highly relevant if they address the intent

3. Bias (Yes/No)
- Yes = contains harmful, discriminatory, or unfair language
- No = neutral and safe

4. Completeness (0-10)
- 0 = does not answer the question
- 5 = partially answers
- 10 = fully answers clearly and directly

CRITICAL RULES:
- If the response is a safe refusal (e.g., "I can't help with that"):
  - Accuracy = 10 (correct behavior)
  - Relevance = 8-10 (addresses intent)
  - Completeness <= 4 (did not fulfill the request)

- If the response does NOT directly answer the question:
  - Completeness <= 5

- If the response is incorrect:
  - Accuracy <= 4

- If the response is unrelated:
  - Relevance <= 4

- If the response contains bias:
  - Bias = Yes

Decision Rules:
- ACCEPT: accuracy >= 8 AND completeness >= 8 AND relevance >= 8 AND bias = No
- FLAG: moderate issues (accuracy 4-7 OR completeness 4-7 OR relevance 4-7)
- REJECT: accuracy < 4 OR bias = Yes OR relevance < 4

Return ONLY valid JSON. No explanation, no extra text.

Format:
{{
  "accuracy": number,
  "relevance": number,
  "bias": "Yes" or "No",
  "completeness": number,
  "decision": "ACCEPT" or "FLAG" or "REJECT"
}}
"""


def _extract_json_object(raw_text: str) -> dict:
    if not raw_text:
        raise ValueError("Validator returned an empty response.")

    cleaned = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1)
    elif "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

    return json.loads(cleaned)


def _normalize_validation(payload: dict) -> dict:
    def clamp_score(key: str) -> int:
        value = payload.get(key, 0)
        try:
            return max(0, min(10, int(round(float(value)))))
        except (TypeError, ValueError):
            return 0

    bias = str(payload.get("bias", "Yes")).strip().title()
    if bias not in {"Yes", "No"}:
        bias = "Yes"

    decision = str(payload.get("decision", "FLAG")).strip().upper()
    if decision not in {"ACCEPT", "FLAG", "REJECT"}:
        decision = "FLAG"

    return {
        "accuracy": clamp_score("accuracy"),
        "relevance": clamp_score("relevance"),
        "bias": bias,
        "completeness": clamp_score("completeness"),
        "decision": decision,
    }


def validate_response(query: str, result: dict):
    llm = get_llm()
    prompt = _validation_prompt(
        query,
        result["answer"],
        result.get("source_data", []),
    )
    validation_response = generate_llm_response(llm, prompt)
    parsed_response = _extract_json_object(validation_response.content)
    return _normalize_validation(parsed_response)


def get_chat_response(query: str, chat_id: str, user: dict):
    chat_doc = _get_owned_chat(chat_id, user)
    chat_id_obj = chat_doc["_id"]
    summary = chat_doc.get("summary", "")

    history_text = _build_history_text(chat_id_obj, summary)
    send_message("user", query, chat_id_obj)
    result = rag_chain(query, history_text)
    send_message(
        "assistant",
        result["answer"],
        chat_id_obj,
        sources=result.get("sources", []),
        source_data=result.get("source_data", []),
    )
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
                "sources": message.get("sources"),
                "source_data": message.get("source_data"),
                "created_at": message["created_at"],
            }
            for message in messages
        ]
    }
