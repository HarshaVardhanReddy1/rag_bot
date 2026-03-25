from fastapi import HTTPException
from datetime import datetime,timezone
from backend.authentication.schemas import RegisterSchema, LoginSchema
from backend.authentication.utils import hash_password, verify_password, create_access_token, create_refresh_token
from backend.database import users_collection


def register_user(data:RegisterSchema):
    # check email if it is in the user table or not
    user = users_collection.find_one({"email": data.email})
    if user:
        raise HTTPException(status_code=400, detail="user is already registered with this email")
    # hash the password
    hashed_password = hash_password(data.password)
    # create doc to save
    user_doc = {
        "name":data.name,
        "email":data.email,
        "password_hash":hashed_password,
        "role":"user",
        "created_at":datetime.now(timezone.utc)
    }
    result = users_collection.insert_one(user_doc)
    return ({
        "message":"user registered successfully",
        "user_id":str(result.inserted_id)
    })

def user_login(data):
    # check email if it is in the user table or not
    user = users_collection.find_one({"email": data.username})
    if not user:
        raise HTTPException(status_code=404, detail="invalid credentials")
    password_check = verify_password(data.password,user.get("password_hash"))
    if not password_check:
        raise HTTPException(status_code=404, detail="invalid credentials")
    access_token = create_access_token({"user_id": str(user["_id"]),"role":user.get("role")})
    refresh_token = create_refresh_token({"user_id": str(user["_id"]),"role":user.get("role")})

    return {
        "message": "Login successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
