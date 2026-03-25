from pydantic import EmailStr, BaseModel, Field


class RegisterSchema(BaseModel):
    name: str = Field(min_length=3, max_length=20)
    email: EmailStr
    password: str = Field(min_length=8, max_length=20)


class LoginSchema(BaseModel):
    email: EmailStr
    password: str