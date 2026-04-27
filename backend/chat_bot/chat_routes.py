from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.chat_bot.chat_services import (
    generate_new_chat,
    get_chat_list,
    get_chat_messages,
    get_chat_response,
)
from backend.chat_bot.chat_utils import _get_owned_chat, send_message
from backend.chat_bot.schemas import NewChatRequest
from backend.database import chats_collection
from backend.dependencies import get_current_user
from backend.rag import DOCS_DIR, SUPPORTED_EXTENSIONS, ingest_uploaded_file

router = APIRouter()
UPLOADS_DIR = DOCS_DIR / "uploads"


# Save an uploaded file locally and ingest it into the user's RAG index.
async def _save_and_ingest_upload(file: UploadFile, user_id: str) -> dict:
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


# Return the chat payload used when the request only uploads a document.
def _build_upload_only_response(chat_id_obj, upload_result: dict) -> dict:
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


@router.post("/chat")
# Handle a chat request that may include both a message and a document upload.
async def chat(
    chat_id: str = Form(...),
    query: str = Form(""),
    file: UploadFile | None = File(None),
    user: dict = Depends(get_current_user),
):
    normalized_query = query.strip()
    chat_doc = _get_owned_chat(chat_id, user)
    chat_id_obj = chat_doc["_id"]

    upload_result = (
        await _save_and_ingest_upload(file, user["_id"])
        if file and file.filename
        else None
    )

    if not normalized_query and not upload_result:
        raise HTTPException(
            status_code=400,
            detail="Write a message or attach a document before sending.",
        )

    if not normalized_query:
        return _build_upload_only_response(chat_id_obj, upload_result)

    result = get_chat_response(normalized_query, chat_id, user)
    result["retrieved_chunks"] = result.get("source_data", [])

    if upload_result:
        result["upload"] = upload_result
    return result


@router.post("/newChat")
# Create a new empty chat session for the current user.
async def new_chat(data: NewChatRequest, user: dict = Depends(get_current_user)):
    return generate_new_chat(data, user)


@router.get("/chatList")
# List the user's chats in most recently updated order.
async def list_chat_items(user: dict = Depends(get_current_user)):
    return get_chat_list(user)


@router.get("/chat/{chat_id}/messages")
# Return the full message history for one chat the user owns.
async def list_chat_messages(chat_id: str, user: dict = Depends(get_current_user)):
    return get_chat_messages(chat_id, user)


@router.post("/logout")
# Keep a simple logout endpoint for the frontend flow.
def logout():
    return {"message": "Logged out successfully"}
