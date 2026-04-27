from pymongo import DESCENDING, MongoClient

from backend.settings import settings

DEFAULT_MONGO_URL = "mongodb://localhost:27017/"
DATABASE_NAME = "rag_bot_db"

client = MongoClient(settings.MONGO_URL or DEFAULT_MONGO_URL)
db = client[DATABASE_NAME]

users_collection = db["users"]
chats_collection = db["chats"]
messages_collection = db["messages"]
audit_logs_collection = db["audit_logs"]

chats_collection.create_index([("user_id", 1), ("updated_at", DESCENDING)])
messages_collection.create_index([("chat_id", 1), ("created_at", 1)])
audit_logs_collection.create_index([("chat_id", 1), ("created_at", DESCENDING)])
audit_logs_collection.create_index([("user_id", 1), ("created_at", DESCENDING)])
audit_logs_collection.create_index([("decision", 1), ("created_at", DESCENDING)])
