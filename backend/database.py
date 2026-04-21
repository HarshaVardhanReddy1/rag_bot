from pymongo import DESCENDING, MongoClient
from backend.settings import settings

client = MongoClient(settings.MONGO_URL or "mongodb://localhost:27017/")
db = client["rag_bot_db"]

users_collection = db["users"]
chats_collection = db["chats"]
messages_collection = db["messages"]
audit_logs_collection = db["audit_logs"]

chats_collection.create_index([("user_id", 1), ("updated_at", DESCENDING)])
messages_collection.create_index([("chat_id", 1), ("created_at", 1)])
audit_logs_collection.create_index([("chat_id", 1), ("created_at", DESCENDING)])
audit_logs_collection.create_index([("user_id", 1), ("created_at", DESCENDING)])
audit_logs_collection.create_index([("decision", 1), ("created_at", DESCENDING)])
