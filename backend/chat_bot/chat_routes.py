from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from backend.database import audit_logs_collection

from backend.chat_bot.chat_services import (
    generate_new_chat,
    get_chat_list,
    get_chat_messages,
    get_chat_response,
    validate_response,
)
from backend.chat_bot.chat_utils import _get_owned_chat, send_message
from backend.chat_bot.schemas import NewChatRequest
from backend.database import chats_collection
from backend.dependencies import get_current_user
from backend.rag.loader import DOCS_DIR, SUPPORTED_EXTENSIONS, ingest_uploaded_file

router = APIRouter()
UPLOADS_DIR = DOCS_DIR / "uploads"
logger = logging.getLogger(__name__)
AUDIT_MODEL_VERSION = "v1.0"


async def _save_and_ingest_upload(file: UploadFile, user_id: str):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOADS_DIR / f"{uuid4().hex}_{Path(file.filename).name}"

    try:
        saved_path.write_bytes(await file.read())
        return ingest_uploaded_file(saved_path, uploaded_by=user_id)
    except Exception:
        if saved_path.exists():
            saved_path.unlink()
        raise
    finally:
        await file.close()


def _record_audit_log(*, user: dict, chat_id: str, query: str, result: dict):
    try:
        metrics = _build_audit_metrics(validate_response(query, result))
        audit_logs_collection.insert_one(_build_audit_log(
            user=user,
            chat_id=chat_id,
            query=query,
            result=result,
            metrics=metrics,
        ))
        return metrics

    except Exception:
        logger.exception("Failed to validate or store audit log for chat %s", chat_id)
        return None


def _safe_text(value) -> str:
    return str(value or "").strip()


def _build_retrieved_chunks(source_data) -> list[dict]:
    if not isinstance(source_data, list):
        return []

    chunks = []
    for item in source_data:
        if not isinstance(item, dict):
            continue

        chunks.append(
            {
                "file_name": _safe_text(item.get("file_name")) or None,
                "source": _safe_text(item.get("source")) or None,
                "content": _safe_text(item.get("content")),
                "relevance_score": item.get("relevance_score"),
                "retrieval_reason": _safe_text(item.get("retrieval_reason")) or None,
            }
        )

    return chunks


def _build_audit_metrics(validation: dict) -> dict:
    confidence = round(
        (validation["accuracy"] + validation["relevance"] + validation["completeness"]) / 3,
        2,
    )
    return {
        "accuracy": validation["accuracy"],
        "bias": validation["bias"],
        "completeness": validation["completeness"],
        "relevance": validation["relevance"],
        "confidence": confidence,
        "decision": validation["decision"],
    }


def _build_audit_log(
    *,
    user: dict,
    chat_id: str,
    query: str,
    result: dict,
    metrics: dict,
) -> dict:
    sources = [_safe_text(source) for source in result.get("sources", []) if _safe_text(source)]
    retrieved_chunks = _build_retrieved_chunks(result.get("source_data", []))
    has_error = bool(_safe_text(result.get("error")))
    created_at = datetime.now(timezone.utc)

    return {
        "user_id": str(user["_id"]),
        "chat_id": chat_id,
        "query": _safe_text(query),
        "answer": _safe_text(result.get("answer")),
        "sources": sources,
        "source_count": len(sources),
        "chunk_count": len(retrieved_chunks),
        "retrieved_chunks": retrieved_chunks,
        "has_error": has_error,
        "error": _safe_text(result.get("error")) or None,
        **metrics,
        "model_version": AUDIT_MODEL_VERSION,
        "created_at": created_at,
    }


'''  File part:
    This is chat route, user can send query and file. if file is uploaded,
      we call save and ingest_uploaded_file, means first we create a file with the file name and uuid,
      then write the bytes inside that file, then we call ingest_uploaded_file.
    There we load the document means converting the long document into chunks then
    converts them into embeddings then add them into vectorstore.

    Query part:
    we call get_chat_response

'''
@router.post("/chat")
async def chat(
    chat_id: str = Form(...),
    query: str = Form(""),
    file: UploadFile | None = File(None),
    user: dict = Depends(get_current_user),
):
    query = query.strip()
    chat_doc = _get_owned_chat(chat_id, user)
    chat_id_obj = chat_doc["_id"]
    upload_result = (
        await _save_and_ingest_upload(file, user["_id"])
        if file and file.filename
        else None
    )

    if not query and not upload_result:
        raise HTTPException(
            status_code=400,
            detail="Write a message or attach a document before sending.",
        )

    if not query:
        send_message(
            "user",
            f"Uploaded document: {upload_result['file_name']}",
            chat_id_obj,
        )
        answer = "Document uploaded and added to the knowledge base."
        send_message(
            "assistant",
            answer,
            chat_id_obj,
            sources=[upload_result["source"]],
            source_data=[],
        )
        chats_collection.update_one(
            {"_id": chat_id_obj},
            {"$set": {"updated_at": datetime.now(timezone.utc)}},
        )
        return {
            "answer": answer,
            "sources": [upload_result["source"]],
            "source_data": [],
            "upload": upload_result,
        }

    result = get_chat_response(query, chat_id, user)
    result["retrieved_chunks"] = _build_retrieved_chunks(result.get("source_data", []))
    metrics = _record_audit_log(user=user, chat_id=chat_id, query=query, result=result)
    if metrics:
        result["metrics"] = metrics

    if upload_result:
        result["upload"] = upload_result
    return result


@router.post("/newChat")
async def new_chat(data: NewChatRequest, user: dict = Depends(get_current_user)):
    return generate_new_chat(data, user)


@router.get("/chatList")
async def list_chat_items(user: dict = Depends(get_current_user)):
    return get_chat_list(user)


@router.get("/chat/{chat_id}/messages")
async def list_chat_messages(chat_id: str, user: dict = Depends(get_current_user)):
    return get_chat_messages(chat_id, user)


@router.post("/logout")
def logout():
    return {"message": "Logged out successfully"}
