"""管理ダッシュボード (/admin) の HTTP 出口。

カードは 1 モジュール 1 枚で、**カードごとに try を閉じる**。MySQL が落ちていても
Milvus のカードは出るし、Milvus が落ちていても設定のカードは出る。

まとめて 1 つの try で包むと、依存先が 1 つ落ちただけで画面全体が 500 になる。
管理画面が真っ白になるのは、まさに何かが落ちている最中である。そのときに
「どれが落ちているか」を出せなくなる作りは、この画面の存在意義を否定する。
"""

import asyncio
import logging
from collections.abc import Callable

from fastapi import APIRouter

from app.config import settings
from app.core import jobs
from app.db import repository
from app.kb import documents, milvus_client, sources

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


async def _card(key: str, title: str, loader: Callable) -> dict:
    """1 枚のカードを作る。loader が何を投げても、このカードだけが失敗として返る。"""
    try:
        stats = await loader()
        return {"key": key, "title": title, "ok": True, "stats": stats, "error": None}
    except Exception as exc:
        logger.exception("管理カードの取得に失敗 key=%s", key)
        # 例外メッセージ自体は出さない。SQLAlchemy の OperationalError は接続 URL を
        # 本文に含み、そこにはパスワードが入っている。型名だけなら秘密を持たない
        return {
            "key": key, "title": title, "ok": False, "stats": None,
            "error": f"取得できません({type(exc).__name__})",
        }


async def _conversations() -> dict:
    return await repository.conversation_stats()


async def _knowledge() -> dict:
    stats = await repository.knowledge_stats()
    return {**stats, "recent": [
        {"id": c.id, "questions": c.questions, "vectorize_status": c.vectorize_status}
        for c in await repository.list_recent_chunks(5)
    ]}


async def _staging() -> dict:
    return await repository.staging_stats()


def _milvus_sync() -> dict:
    client = milvus_client.get_client()
    exists = client.has_collection(milvus_client.COLLECTION)
    return {
        "uri": settings.milvus_uri,
        "collection": milvus_client.COLLECTION,
        "exists": exists,
        "count": milvus_client.count(client) if exists else 0,
        "dim": milvus_client.DIM,
    }


async def _milvus() -> dict:
    # 同期クライアントなので別スレッドへ。落ちている相手だと接続タイムアウトぶん
    # 待たされ、イベントループごと止まる
    return await asyncio.to_thread(_milvus_sync)


async def _sources() -> dict:
    files = []
    for name, ctype in sources.SOURCE_TYPES.items():
        path = sources.source_path(name)
        if not path.is_file():
            files.append({"name": name, "content_type": ctype, "exists": False, "chunks": None})
            continue
        cs = documents.build_chunks(path.read_text(encoding="utf-8"), content_type=ctype)
        files.append({"name": name, "content_type": ctype, "exists": True, "chunks": len(cs)})
    return {"files": files, "total_chunks": sum(f["chunks"] or 0 for f in files)}


async def _jobs() -> dict:
    all_status = jobs.status_all()
    try:
        make = jobs.resolve_make()
        make_error = None
    except jobs.MakeNotFound as exc:
        make, make_error = None, str(exc)
    return {
        "make": make, "make_error": make_error,
        "running": [j["name"] for j in all_status if j["running"]],
        "jobs": all_status,
    }


async def _config() -> dict:
    # 秘密は 1 つも入れない。API キーはもちろん、DATABASE_URL も
    # 資格情報を含むので出さない
    return {
        "chat_model": settings.chat_model,
        "embed_model": settings.embed_model,
        "extract_method": settings.extract_method,
        "token_budget": settings.token_budget,
        "request_timeout": settings.request_timeout,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_min_score": settings.retrieval_min_score,
        "milvus_uri": settings.milvus_uri,
    }


@router.get("/overview")
async def overview() -> dict:
    cards = [
        await _card("conversations", "会話・チケット", _conversations),
        await _card("knowledge", "ナレッジ (MySQL)", _knowledge),
        await _card("vectors", "ベクトル索引 (Milvus)", _milvus),
        await _card("staging", "会話からの抽出", _staging),
        await _card("sources", "ナレッジ元資料", _sources),
        await _card("jobs", "ジョブ", _jobs),
        await _card("config", "設定", _config),
    ]
    return {"cards": cards, "degraded": [c["key"] for c in cards if not c["ok"]]}
