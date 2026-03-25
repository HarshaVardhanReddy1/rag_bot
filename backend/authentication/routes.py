from fastapi import APIRouter, status,Depends

from backend.authentication.schemas import RegisterSchema, LoginSchema
from backend.authentication.services import register_user, user_login
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(data: RegisterSchema):
    return register_user(data)


@router.post("/login")
async def login(data:OAuth2PasswordRequestForm = Depends() ):
    return user_login(data)