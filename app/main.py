from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.actions import router as actions_router
from app.api.admin import router as admin_router
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.extract import router as extract_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
from app.api.rageval import router as rageval_router
from app.core import logging_setup
from app.graph import runtime

# 画面のパス → static 配下のファイル名。middleware がこれを見るので、
# ルート登録より前に定義しておく。
_PAGES = {"/kb": "kb.html", "/admin": "admin.html", "/rag-eval": "rageval.html"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """graph の checkpointer を起動時に 1 つ開き、終了時に閉じる。

    checkpointer をリクエストごとに開かないことは runtime の前提そのもので
    (app/graph/runtime.py の冒頭)、その「1 つだけ」を保証できる場所がここしかない。

    **init_graph が落ちたらアプリを起動させない(degrade して立ち上げない)。**
    graph は /api/chat と /api/agent の両方にとって唯一の実装なので、開けないまま
    起動すると「プロセスは健康、チャットは毎回 500」というサーバになる。/kb や
    /admin は graph 無しでも動くため管理画面だけ生かす手もあるが、それは監視から
    見て正常に見えてしまうのが困る。init_graph が落ちる原因(checkpointer の sqlite を
    開けない)はリクエストを受けても直らない環境側の問題なので、起動時に大きな音で
    倒れる方が復旧が早い。

    閉じる側を finally に置くのは、起動後にアプリ本体で例外が起きた場合でも
    sqlite のハンドルを手放すため。ここを抜けずに落ちると、ファイルロックが
    残ったままプロセスだけ消える。

    ログの設定をここで行うのは、**プロセスとして立ち上がるときにだけ**設定したいため。
    import 時に行うと、router を 1 つ読み込んだだけのテストまでログファイルを開く。
    """
    # graph より先に呼ぶ。ここから後ろで起きることを記録できるようにする。
    # これを呼ばないとアプリ自身のログは 1 行も出ない(logging_setup の冒頭)。
    logging_setup.configure()
    await runtime.init_graph()
    try:
        yield
    finally:
        await runtime.close_graph()


app = FastAPI(title="ECカスタマーサポート", version="0.1.0", lifespan=lifespan)


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
app.include_router(actions_router)
app.include_router(conversations_router)

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
