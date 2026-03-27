from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.chat_bot.chat_services import generate_new_chat, get_chat_list, get_chat_messages
from backend.chat_bot.chat_services import get_chat_response
from backend.chat_bot.chat_utils import _get_owned_chat, send_message
from backend.chat_bot.schemas import NewChatRequest
from backend.database import chats_collection
from backend.dependencies import get_current_user
from backend.rag.loader import DOCS_DIR, SUPPORTED_EXTENSIONS, ingest_file

router = APIRouter()
UPLOADS_DIR = DOCS_DIR / "uploads"


@router.post("/chat")
async def chat(
    chat_id: str = Form(...),
    query: str = Form(""),
    file: UploadFile | None = File(None),
    user: dict = Depends(get_current_user),
):
    try:
        query = query.strip()
        upload_result = None

        if file and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                )

            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = Path(file.filename).name
            saved_path = UPLOADS_DIR / f"{uuid4().hex}_{safe_name}"

            try:
                file_bytes = await file.read()
                saved_path.write_bytes(file_bytes)
                upload_result = ingest_file(saved_path, uploaded_by=user["_id"])
            except Exception:
                if saved_path.exists():
                    saved_path.unlink()
                raise
            finally:
                await file.close()

        if not query and not upload_result:
            raise HTTPException(
                status_code=400,
                detail="Write a message or attach a document before sending.",
            )

        if not query:
            chat_doc = _get_owned_chat(chat_id, user)
            chat_id_obj = chat_doc["_id"]
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
        if upload_result:
            result["upload"] = upload_result
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
