from fastapi import APIRouter, HTTPException,Depends
from pydantic import BaseModel
from backend.chat_bot.chat_services import get_chat_response
from backend.chat_bot.chat_services import generate_new_chat, get_chat_list, get_chat_messages
from backend.dependencies import get_current_user
from backend.chat_bot.schemas import NewChatRequest
router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    chat_id: str


@router.post("/chat")
def chat(request: ChatRequest,user: dict = Depends(get_current_user)):
    try:
        result = get_chat_response(request.query,request.chat_id,user)
        return result
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
