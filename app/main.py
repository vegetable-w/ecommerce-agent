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

# 画面のパス → static 配下のファイル名。middleware がこれを見るので、
# ルート登録より前に定義しておく。
_PAGES = {"/kb": "kb.html", "/admin": "admin.html", "/rag-eval": "rageval.html"}

app = FastAPI(title="ECカスタマーサポート", version="0.1.0")


@app.middleware("http")
async def _no_stale_frontend(request, call_next):
    """画面と静的ファイルに Cache-Control: no-cache を付ける。

    FileResponse も StaticFiles も ETag と Last-Modified は返すが Cache-Control を
    付けない。すると browser は**発見的キャッシュ**に落ちて、サーバへ確認しないまま
    手元の写しを再利用することがある。実測: static/kb.html にボタンを足しても
    画面に出てこなかった(サーバは新しい内容を返せる状態だった)。

    no-cache は「キャッシュするな」ではなく「使う前に必ず確認しろ」なので、
    中身が変わっていなければ 304 で返り、転送量はほぼ増えない。
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path in _PAGES or path == "/":
        response.headers["Cache-Control"] = "no-cache"
    return response
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)
app.include_router(rageval_router)

# チャット画面。API と同一オリジンで配信するので CORS 設定は不要。
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ファイルの有無に関わらずルートは登録する。「ファイルがあるときだけ @app.get する」
# 形にすると、HTML を置くまで /kb が 404 になるのは同じでも、OpenAPI にも経路が
# 現れず、フロントを書く側から見て「まだ実装されていない」のか「名前を間違えた」
# のか区別が付かない。
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
