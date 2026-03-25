from fastapi import FastAPI
from backend.authentication.routes import router as auth_router
from backend.chat_bot.chat_routes import router as chat_router

app  = FastAPI(title="rag_bot")


app.include_router(chat_router)
app.include_router(auth_router)

@app.get("/")
def print_hello():
    return "hello"