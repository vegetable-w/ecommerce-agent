from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin import router as admin_router
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
from app.api.rageval import router as rageval_router

app = FastAPI(title="ECカスタマーサポート", version="0.1.0")
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)
app.include_router(rageval_router)

# チャット画面。API と同一オリジンで配信するので CORS 設定は不要。
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# 画面のパス → static 配下のファイル名。ファイルの有無に関わらずルートは登録する。
# 「ファイルがあるときだけ @app.get する」形にすると、HTML を置くまで /kb が 404 に
# なるのは同じでも、OpenAPI にも経路が現れず、フロントを書く側から見て
# 「まだ実装されていない」のか「名前を間違えた」のか区別が付かない。
_PAGES = {"/kb": "kb.html", "/admin": "admin.html", "/rag-eval": "rageval.html"}

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    def _make_page(filename: str):
        async def page() -> FileResponse:
            path = STATIC_DIR / filename
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"画面 {filename} はまだ配置されていません")
            return FileResponse(path)

        return page

    for _route, _filename in _PAGES.items():
        app.add_api_route(_route, _make_page(_filename), include_in_schema=False)
