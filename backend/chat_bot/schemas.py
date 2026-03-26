from pydantic import BaseModel, Field


class NewChatRequest(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=100)
