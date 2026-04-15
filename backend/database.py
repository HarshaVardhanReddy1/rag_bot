from pymongo import MongoClient
from backend.settings import settings

client = MongoClient(settings.MONGO_URL or "mongodb://localhost:27017/")
db = client["rag_bot_db"]

users_collection = db["users"]
chats_collection = db["chats"]
messages_collection = db["messages"]
audit_logs_collection = db["audit_logs"]
