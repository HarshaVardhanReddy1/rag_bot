from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status

from backend.authentication.schemas import RegisterSchema
from backend.authentication.utils import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from backend.database import users_collection


def _build_token_payload(user: dict[str, Any]) -> dict[str, str]:
    return {
        "user_id": str(user["_id"]),
        "role": str(user.get("role", "user")),
    }


def register_user(data: RegisterSchema) -> dict[str, str]:
    existing_user = users_collection.find_one({"email": data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user is already registered with this email",
        )

    user_doc = {
        "name": data.name,
        "email": data.email,
        "password_hash": hash_password(data.password),
        "role": "user",
        "created_at": datetime.now(timezone.utc),
    }
    result = users_collection.insert_one(user_doc)

    return {
        "message": "user registered successfully",
        "user_id": str(result.inserted_id),
    }


def user_login(data: Any) -> dict[str, str]:
    user = users_collection.find_one({"email": data.username})
    if not user or not verify_password(data.password, user.get("password_hash")):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="invalid credentials",
        )

    token_payload = _build_token_payload(user)
    return {
        "message": "Login successful",
        "access_token": create_access_token(token_payload),
        "refresh_token": create_refresh_token(token_payload),
        "token_type": "bearer",
    }
