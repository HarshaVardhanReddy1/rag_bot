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
from backend.rag.loader import DOCS_DIR, SUPPORTED_EXTENSIONS, ingest_file

router = APIRouter()
UPLOADS_DIR = DOCS_DIR / "uploads"
logger = logging.getLogger(__name__)


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
        return ingest_file(saved_path, uploaded_by=user_id)
    except Exception:
        if saved_path.exists():
            saved_path.unlink()
        raise
    finally:
        await file.close()


def _record_audit_log(*, user: dict, chat_id: str, query: str, result: dict):
    try:
        response = validate_response(query, result)
        confidence = (
            response["accuracy"] + response["relevance"] + response["completeness"]
        ) / 3
        data = {
                "user_id": str(user["_id"]),
                "chat_id": chat_id,
                "query": query,
                "answer": result.get("answer"),
                "accuracy": response["accuracy"],
                "bias": response["bias"],
                "completeness": response["completeness"],
                "relevance": response["relevance"],
                "confidence": confidence,
                "decision": response["decision"],
                "model_version": "v1.0",
                "created_at": datetime.now(timezone.utc),
            }
        audit_logs_collection.insert_one(data)
        
    except Exception:
        logger.exception("Failed to validate or store audit log for chat %s", chat_id)


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
        send_message("user", f"Uploaded document: {upload_result['file_name']}", chat_id_obj)
        answer = "Document uploaded and added to the knowledge base."
        send_message("assistant", answer, chat_id_obj)
        chats_collection.update_one(
            {"_id": chat_id_obj},
            {"$set": {"updated_at": datetime.now(timezone.utc)}},
        )
        return {
            "answer": answer,
            "sources": [upload_result["source"]],
            "upload": upload_result,
        }

    result = get_chat_response(query, chat_id, user)
    _record_audit_log(user=user, chat_id=chat_id, query=query, result=result)

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
