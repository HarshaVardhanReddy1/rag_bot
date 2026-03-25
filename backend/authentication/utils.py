from datetime import timezone, datetime, timedelta
from passlib.context import CryptContext
from backend.settings import settings
from jose import jwt,JWTError


pwd_context = CryptContext(schemes=["bcrypt"],deprecated = "auto")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

def hash_password(password:str):
    return pwd_context.hash(password)
def verify_password(plain_password:str,hashed_password:str):
    return pwd_context.verify(plain_password,hashed_password)

def create_access_token(data:dict):
    payload = data.copy()
    expire = datetime.now(timezone.utc)+timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload["exp"] = expire
    token = jwt.encode(payload,settings.JWT_SECRET_KEY,algorithm=ALGORITHM)
    return token

def create_refresh_token(data:dict):
    payload = data.copy()
    expire = datetime.now(timezone.utc)+timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload['exp'] = expire
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=ALGORITHM)

    return token


def decode_access_token(token: str):

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload

    except JWTError:
        return None
