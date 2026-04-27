from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from backend.authentication.schemas import RegisterSchema
from backend.authentication.services import register_user, user_login

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(data: RegisterSchema):
    return register_user(data)


@router.post("/login")
async def login(data: OAuth2PasswordRequestForm = Depends()):
    return user_login(data)
