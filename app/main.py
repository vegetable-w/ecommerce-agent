from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router

app = FastAPI(title="ECカスタマーサポート", version="0.1.0")
app.include_router(chat_router)
app.include_router(extract_router)

# チャット画面。API と同一オリジンで配信するので CORS 設定は不要。
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
