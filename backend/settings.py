import os
from dotenv import load_dotenv

load_dotenv()

class Settings():
    def __init__(self):
        self.HF_API_KEY = os.getenv("HF_API_KEY")
        self.MONGO_URL = os.getenv("MONGO_URL")
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")


settings = Settings()