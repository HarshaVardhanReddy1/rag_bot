import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from backend.chat_bot.chat_summarization import summarize_if_needed
from backend.chat_bot.chat_utils import (
    _build_history_text,
    _get_owned_chat,
    send_message,
)
from backend.database import chats_collection, messages_collection
from backend.llm_model.model import get_llm
from backend.rag.chain import generate_llm_response, get_rag_chain

logger = logging.getLogger(__name__)

DEFAULT_CHAT_TITLE = "New Chat"
MAX_VALIDATION_SOURCE_CHARS = 12000
DEFAULT_VALIDATION_RESPONSE = {
    "accuracy": 0,
    "relevance": 0,
    "bias": "No",
    "completeness": 0,
    "decision": "FLAG",
}

rag_chain = get_rag_chain()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _sanitize_sources(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    seen: set[str] = set()
    cleaned_sources: list[str] = []

    for item in value:
        source = _safe_text(item)
        if not source or source in seen:
            continue
        seen.add(source)
        cleaned_sources.append(source)

    return cleaned_sources


def _sanitize_source_data(source_data: Any) -> list[dict[str, Any]]:
    if not isinstance(source_data, list):
        return []

    cleaned_sources: list[dict[str, Any]] = []

    for item in source_data:
        if not isinstance(item, dict):
            continue

        cleaned_source = {
            "source": _safe_text(item.get("source")) or None,
            "file_name": _safe_text(item.get("file_name")) or None,
            "relevance_score": item.get("relevance_score"),
            "retrieval_reason": _safe_text(item.get("retrieval_reason")) or None,
            "content": _safe_text(item.get("content")),
        }
        cleaned_sources.append(cleaned_source)

    return cleaned_sources


def _normalize_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        logger.warning("RAG pipeline returned unexpected result type: %s", type(result))
        return {
            "answer": "Something went wrong while processing your request.",
            "sources": [],
            "source_data": [],
        }

    answer = _safe_text(result.get("answer"))
    normalized = {
        "answer": answer or "No data found.",
        "sources": _sanitize_sources(result.get("sources")),
        "source_data": _sanitize_source_data(result.get("source_data")),
    }

    if result.get("error"):
        normalized["error"] = _safe_text(result.get("error"))

    return normalized


def _format_source_data_for_validation(source_data: list[dict[str, Any]]) -> str:
    if not source_data:
        return "No source data was retrieved."

    formatted_sources: list[str] = []
    total_chars = 0

    for index, source in enumerate(source_data, start=1):
        metadata = []

        file_name = _safe_text(source.get("file_name"))
        source_path = _safe_text(source.get("source"))
        retrieval_reason = _safe_text(source.get("retrieval_reason"))
        relevance_score = source.get("relevance_score")

        if file_name:
            metadata.append(f"file_name={file_name}")
        if source_path:
            metadata.append(f"source={source_path}")
        if isinstance(relevance_score, (int, float)):
            metadata.append(f"relevance_score={relevance_score:.3f}")
        if retrieval_reason:
            metadata.append(f"retrieval_reason={retrieval_reason}")

        metadata_text = ", ".join(metadata) if metadata else "metadata unavailable"
        content = _safe_text(source.get("content"))
        formatted_source = f"Source {index} ({metadata_text}):\n{content}"

        if total_chars + len(formatted_source) > MAX_VALIDATION_SOURCE_CHARS:
            remaining_chars = MAX_VALIDATION_SOURCE_CHARS - total_chars
            if remaining_chars > 0:
                formatted_sources.append(formatted_source[:remaining_chars].rstrip())
            break

        formatted_sources.append(formatted_source)
        total_chars += len(formatted_source)

    return "\n\n".join(formatted_sources)


def _validation_prompt(query: str, answer: str, source_data: list[dict[str, Any]]) -> str:
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
  - Accuracy = 10
  - Relevance = 8-10
  - Completeness <= 4

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


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        raise ValueError("Validator returned an empty response.")

    cleaned = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1)
    elif "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Validator response must be a JSON object.")

    return payload


def _derive_decision(
    *,
    accuracy: int,
    relevance: int,
    completeness: int,
    bias: str,
) -> str:
    if accuracy < 4 or relevance < 4 or bias == "Yes":
        return "REJECT"
    if accuracy >= 8 and relevance >= 8 and completeness >= 8:
        return "ACCEPT"
    return "FLAG"


def _normalize_validation(payload: dict[str, Any]) -> dict[str, Any]:
    def clamp_score(key: str) -> int:
        value = payload.get(key, 0)
        try:
            return max(0, min(10, int(round(float(value)))))
        except (TypeError, ValueError):
            return 0

    accuracy = clamp_score("accuracy")
    relevance = clamp_score("relevance")
    completeness = clamp_score("completeness")

    bias = str(payload.get("bias", "No")).strip().title()
    if bias not in {"Yes", "No"}:
        bias = "No"

    decision = str(payload.get("decision", "")).strip().upper()
    derived_decision = _derive_decision(
        accuracy=accuracy,
        relevance=relevance,
        completeness=completeness,
        bias=bias,
    )
    if decision not in {"ACCEPT", "FLAG", "REJECT"}:
        decision = derived_decision
    elif decision != derived_decision:
        logger.warning(
            "Validator decision %s does not match derived decision %s; using derived value.",
            decision,
            derived_decision,
        )
        decision = derived_decision

    return {
        "accuracy": accuracy,
        "relevance": relevance,
        "bias": bias,
        "completeness": completeness,
        "decision": decision,
    }


def validate_response(query: str, result: dict[str, Any]) -> dict[str, Any]:
    normalized_result = _normalize_result(result)
    prompt = _validation_prompt(
        _safe_text(query),
        normalized_result["answer"],
        normalized_result["source_data"],
    )

    try:
        llm = get_llm()
        validation_response = generate_llm_response(llm, prompt)
        raw_content = getattr(validation_response, "content", validation_response)
        parsed_response = _extract_json_object(_safe_text(raw_content))
        return _normalize_validation(parsed_response)
    except Exception:
        logger.exception("Failed to validate response for query: %s", _safe_text(query))
        return dict(DEFAULT_VALIDATION_RESPONSE)


def get_chat_response(query: str, chat_id: str, user: dict[str, Any]) -> dict[str, Any]:
    normalized_query = _safe_text(query)
    if not normalized_query:
        raise ValueError("Query must not be empty.")

    chat_doc = _get_owned_chat(chat_id, user)
    chat_id_obj = chat_doc["_id"]
    summary = _safe_text(chat_doc.get("summary"))

    history_text = _build_history_text(chat_id_obj, summary)
    send_message("user", normalized_query, chat_id_obj)

    result = _normalize_result(rag_chain(normalized_query, str(user["_id"]), history_text))

    send_message(
        "assistant",
        result["answer"],
        chat_id_obj,
        sources=result["sources"],
        source_data=result["source_data"],
    )

    try:
        summarize_if_needed(chat_id_obj)
    except Exception:
        logger.exception("Failed to summarize chat %s", chat_id)

    chats_collection.update_one(
        {"_id": chat_id_obj},
        {"$set": {"updated_at": datetime.now(timezone.utc)}},
    )

    return result


def generate_new_chat(data: Any, user: dict[str, Any]) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    raw_title = _safe_text(getattr(data, "title", ""))
    title = raw_title or DEFAULT_CHAT_TITLE

    doc = {
        "user_id": str(user["_id"]),
        "title": title,
        "summary": "",
        "created_at": now,
        "updated_at": now,
    }

    result = chats_collection.insert_one(doc)

    return {
        "chat_id": str(result.inserted_id),
        "title": title,
    }


def get_chat_list(user: dict[str, Any]) -> list[dict[str, Any]]:
    chats = chats_collection.find(
        {"user_id": str(user["_id"])},
        {"title": 1, "summary": 1, "updated_at": 1},
    ).sort("updated_at", -1)

    return [
        {
            "chat_id": str(chat["_id"]),
            "title": _safe_text(chat.get("title")),
            "summary": _safe_text(chat.get("summary")),
            "updated_at": chat.get("updated_at"),
        }
        for chat in chats
    ]


def get_chat_messages(chat_id: str, user: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    chat_doc = _get_owned_chat(chat_id, user)
    messages = messages_collection.find(
        {"chat_id": chat_doc["_id"]}
    ).sort("created_at", 1)

    return {
        "messages": [
            {
                "role": _safe_text(message.get("role")),
                "content": _safe_text(message.get("content")),
                "image_url": message.get("image_url"),
                "sources": _sanitize_sources(message.get("sources")),
                "source_data": _sanitize_source_data(message.get("source_data")),
                "created_at": message.get("created_at"),
            }
            for message in messages
        ]
    }
