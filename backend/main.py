from fastapi import FastAPI

from backend.authentication.routes import router as auth_router
from backend.chat_bot.chat_routes import router as chat_router


def create_app() -> FastAPI:
    application = FastAPI(title="rag_bot")
    application.include_router(chat_router)
    application.include_router(auth_router)
    return application


app = create_app()


@app.get("/")
def health_check() -> str:
    return "hello"
