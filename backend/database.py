from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["rag_bot_db"]

users_collection = db["users"]
chats_collection = db["chats"]
messages_collection = db["messages"]