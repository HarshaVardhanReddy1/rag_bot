import logging
from pathlib import Path

 
 

from fastapi import FastAPI
from backend.authentication.routes import router as auth_router
from backend.chat_bot.chat_routes import router as chat_router

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "rag_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logging.getLogger("backend.rag.pipeline").setLevel(logging.INFO)

app  = FastAPI(title="rag_bot")

 

app.include_router(chat_router)
app.include_router(auth_router)

@app.get("/")
def print_hello():
    return "hello"
