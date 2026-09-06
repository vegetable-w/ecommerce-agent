"""ナレッジ登録画面 (/kb) の HTTP 出口。

この層は薄いままにしておくこと。特に次の 2 つを守る:

- **分割の実装はここに書かない。** プレビューも取り込みも app.kb.documents.build_chunks を
  通す。プレビュー用にもう 1 つ分割器を置くと、「画面で見た 31 件」と「登録された 29 件」の
  ように静かにずれ、しかもどちらが正しいのか誰にも分からなくなる。
- **ベクトル化も自前で書かない。** app.kb.dualwrite.vectorize_pending を呼ぶ。あちらには
  「上流が入力より少ないベクトルを返したらバッチ全体を pending のまま残す」という
  実測で見つけた番人が入っている。ここで embed → upsert を書き直すとその守りを失う。
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.core import jobs, retrieval
from app.db import repository
from app.kb import chunking, documents, dualwrite, milvus_client, sources
from app.kb.documents import Chunk
from app.schemas.kb import (
    ChunkPreview,
    IngestRequest,
    IngestResponse,
    PreviewRequest,
    PreviewResponse,
    SearchRequest,
    SearchResponse,
    StagingResponse,
    VectorizeResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb", tags=["kb"])


# ---------------------------------------------------------------------------
# 入力の解決(プレビューと取り込みで共有)
# ---------------------------------------------------------------------------


def _resolve_input(req: PreviewRequest) -> tuple[str, str, str | None]:
    """(Markdown 本文, content_type, 資料名) を返す。不正な入力はすべて 400。

    source が指定された場合は sources.source_path() に名前を検査させる。あちらは
    登録済み一覧との完全一致だけを通すので、`../` を含む名前はファイルを開く前に
    ValueError になる。ここで自前のパス連結や正規化を書き足さないこと(検査を
    すり抜ける経路が増えるだけになる)。
    """
    if req.source is not None:
        try:
            path = sources.source_path(req.source)
        except ValueError:
            # 受け取った名前はエラー文へ echo しない。代わりに有効な名前を示す
            raise HTTPException(
                status_code=400,
                detail="資料一覧にない資料名です。指定できるのは: "
                + " / ".join(sources.SOURCE_TYPES),
            )
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"資料ファイルが見つかりません: {req.source}")
        return path.read_text(encoding="utf-8"), sources.SOURCE_TYPES[req.source], req.source

    text = req.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text か source のどちらかを指定してください")
    ctype = req.content_type
    if ctype not in sources.CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="content_type が不正です。指定できるのは: " + " / ".join(sources.CONTENT_TYPES),
        )
    return text, ctype, None


def _mark_duplicates(chunks: list[Chunk], known: set[str] | None) -> list[bool | None]:
    """各 chunk が既存と重複するかを並び順で判定する。

    指紋は questions と answer の両方から作る(repository.chunk_fingerprint)。
    質問文だけで突き合わせると、大きな表が割れてできた複数 chunk は見出しを共有する
    ため、2 枚目以降が「重複」として捨てられる。

    known が None のときは判定不能。False(重複でない)で埋めてはいけない。
    """
    if known is None:
        return [None] * len(chunks)
    seen = set(known)
    out: list[bool | None] = []
    for c in chunks:
        fp = repository.chunk_fingerprint(c.questions, c.answer)
        out.append(fp in seen)
        seen.add(fp)  # 同じリクエスト内の完全な重複も 2 件目以降を重複とみなす
    return out


def _to_preview(chunks: list[Chunk], flags: list[bool | None]) -> list[ChunkPreview]:
    return [
        ChunkPreview(
            index=i,
            category=c.category,
            section_path=c.section_path,
            questions=c.questions,
            answer=c.answer,
            chars=len(c.answer),
            is_key_clause=bool(c.is_key_clause),
            is_table=chunking.is_table_block(c.answer),
            is_duplicate=flags[i],
        )
        for i, c in enumerate(chunks)
    ]


async def _known_fingerprints() -> set[str] | None:
    """既存 chunk の指紋。MySQL を読めなければ None(=「判定できない」)を返す。"""
    try:
        return await repository.list_chunk_fingerprints()
    except Exception:
        logger.exception("既存 chunk の指紋を取得できない")
        return None


# ---------------------------------------------------------------------------
# Milvus 側の読み取り
# ---------------------------------------------------------------------------


def _milvus_count_sync() -> int:
    """collection が無ければ 0。**ここで作らない**。

    ensure_collection を呼ぶと読み取りのはずの画面表示が collection を新規作成する
    副作用を持つ。作るのは書き込み経路(ベクトル化)の責任。
    """
    client = milvus_client.get_client()
    if not client.has_collection(milvus_client.COLLECTION):
        return 0
    return milvus_client.count(client)


async def _milvus_count() -> int:
    # pymilvus は同期クライアント。落ちている相手だと接続タイムアウトぶん待つので、
    # イベントループを塞がないように別スレッドへ逃がす
    return await asyncio.to_thread(_milvus_count_sync)


# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------


@router.get("/overview")
async def overview() -> dict:
    """/kb の画面全体。依存先ごとに try を分け、1 つ落ちても残りは出す。"""
    knowledge, knowledge_error = None, None
    try:
        knowledge = await repository.knowledge_stats()
    except Exception as exc:
        logger.exception("knowledge_stats に失敗")
        knowledge_error = f"MySQL から取得できません({type(exc).__name__})"

    milvus_count, milvus_error = None, None
    try:
        milvus_count = await _milvus_count()
    except Exception as exc:
        logger.exception("Milvus の件数取得に失敗")
        milvus_error = f"Milvus から取得できません({type(exc).__name__})"

    # 片側でも読めなければ None(取得不可)。False にしてはいけない。
    # False は「両方読めて数が食い違っている」という別の事実であり、対処も違う
    # (再ベクトル化すべきか、まず依存先を起こすべきか)。
    consistent = None
    if knowledge is not None and milvus_count is not None:
        consistent = knowledge["done"] == milvus_count

    staging, staging_error = None, None
    try:
        staging = await repository.staging_stats()
    except Exception as exc:
        logger.exception("staging_stats に失敗")
        staging_error = f"MySQL から取得できません({type(exc).__name__})"

    recent = []
    try:
        recent = [
            {
                "id": c.id, "questions": c.questions, "section_path": c.section_path,
                "content_type": c.content_type, "is_key_clause": bool(c.is_key_clause),
                "vectorize_status": c.vectorize_status, "chars": len(c.answer),
            }
            for c in await repository.list_recent_chunks(20)
        ]
    except Exception:
        logger.exception("直近 chunk の取得に失敗")

    return {
        "knowledge": knowledge,
        "knowledge_error": knowledge_error,
        "milvus": {"count": milvus_count, "collection": milvus_client.COLLECTION,
                   "uri": settings.milvus_uri},
        "milvus_error": milvus_error,
        "consistent": consistent,
        "staging": staging,
        "staging_error": staging_error,
        "sources": _source_inventory(),
        "recent": recent,
        "jobs": jobs.status_all(),
    }


def _source_inventory() -> list[dict]:
    """登録済み資料ごとの chunk 数。資料 1 つが壊れていても他は出す。"""
    out = []
    for name, ctype in sources.SOURCE_TYPES.items():
        entry = {"name": name, "content_type": ctype, "exists": False,
                 "chunks": None, "key_clauses": None, "chars": None, "error": None}
        try:
            path = sources.source_path(name)
            if path.is_file():
                md = path.read_text(encoding="utf-8")
                cs = documents.build_chunks(md, content_type=ctype)
                entry |= {"exists": True, "chunks": len(cs),
                          "key_clauses": sum(c.is_key_clause for c in cs), "chars": len(md)}
        except Exception as exc:
            logger.exception("資料の読み取りに失敗 name=%s", name)
            entry["error"] = f"読み取れません({type(exc).__name__})"
        out.append(entry)
    return out


@router.post("/preview", response_model=PreviewResponse)
async def preview(req: PreviewRequest) -> PreviewResponse:
    """分割結果を見るだけ。MySQL にも Milvus にも一切書かない。"""
    md, ctype, source = _resolve_input(req)
    chunks = documents.build_chunks(md, content_type=ctype)
    known = await _known_fingerprints()
    flags = _mark_duplicates(chunks, known)
    items = _to_preview(chunks, flags)
    return PreviewResponse(
        source=source,
        content_type=ctype,
        total=len(items),
        key_clauses=sum(1 for c in items if c.is_key_clause),
        tables=sum(1 for c in items if c.is_table),
        duplicates=None if known is None else sum(1 for f in flags if f),
        duplicate_check="ok" if known is not None else "unavailable",
        chunks=items,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """MySQL(正) → Milvus(索引) の順で書く。順序を入れ替えないこと。

    ベクトル化に失敗しても MySQL 側は**巻き戻さない**。本文は正の側に入っており、
    status=pending のまま残るので、再実行(make kb-vectorize / POST /api/kb/vectorize)が
    残りだけを拾う。ロールバックするとその再開点を自ら捨てることになる。
    """
    md, ctype, source = _resolve_input(req)
    chunks = documents.build_chunks(md, content_type=ctype)

    try:
        known = await repository.list_chunk_fingerprints()
        flags = _mark_duplicates(chunks, known)
        fresh = [c for c, dup in zip(chunks, flags) if not dup]
        ids = await dualwrite.write_pending(fresh)
    except SQLAlchemyError as exc:
        logger.exception("ナレッジの保存に失敗")
        raise HTTPException(
            status_code=503,
            detail="データベースを一時的に利用できません。しばらくしてからもう一度お試しください",
        ) from exc

    skipped = len(chunks) - len(ids)
    total = None
    try:
        total = (await repository.knowledge_stats())["total"]
    except Exception:
        logger.exception("登録後の件数取得に失敗")

    if not req.vectorize:
        return IngestResponse(
            source=source, content_type=ctype, inserted=len(ids), skipped=skipped, ids=ids,
            knowledge_total=total, vectorized=None,
            message=f"{len(ids)} 件を pending として保存しました(重複 {skipped} 件は登録していません)",
        )

    try:
        client = await asyncio.to_thread(milvus_client.get_client)
        await asyncio.to_thread(milvus_client.ensure_collection, client)
        vectorized = await dualwrite.vectorize_pending(client)
    except Exception as exc:
        logger.exception("ベクトル化に失敗 inserted=%d", len(ids))
        # ロールバックしない。502 で「保存済み・再実行で補完可能」と伝えるだけにする
        raise HTTPException(
            status_code=502,
            detail=f"{len(ids)} 件は保存済み(pending)。もう一度ベクトル化すれば補完できます",
        ) from exc

    return IngestResponse(
        source=source, content_type=ctype, inserted=len(ids), skipped=skipped, ids=ids,
        knowledge_total=total, vectorized=vectorized,
        message=f"{len(ids)} 件を保存し、{vectorized} 件をベクトル化しました",
    )


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize() -> VectorizeResponse:
    """pending をまとめてベクトル化する(冪等)。"""
    try:
        client = await asyncio.to_thread(milvus_client.get_client)
        await asyncio.to_thread(milvus_client.ensure_collection, client)
        n = await dualwrite.vectorize_pending(client)
    except Exception as exc:
        logger.exception("ベクトル化に失敗")
        raise HTTPException(
            status_code=502,
            detail="ベクトル化に失敗しました。対象は pending のまま残っているので、再実行で補完できます",
        ) from exc

    count = pending = None
    try:
        count = await _milvus_count()
    except Exception:
        logger.exception("Milvus の件数取得に失敗")
    try:
        pending = await repository.count_chunks_by_status("pending")
    except Exception:
        logger.exception("pending 件数の取得に失敗")
    return VectorizeResponse(
        vectorized=n, milvus_count=count, pending=pending,
        message=f"{n} 件をベクトル化しました",
    )


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """登録した知識が実際に引けるかを確かめる検索テスト。"""
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query を指定してください")
    top_k = req.top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if req.min_score is None else req.min_score
    try:
        hits = await retrieval.search_knowledge(query, top_k=top_k, min_score=min_score)
    except Exception as exc:
        logger.exception("検索に失敗")
        raise HTTPException(
            status_code=502,
            detail="検索に失敗しました。埋め込み上流か Milvus を確認してください",
        ) from exc
    return SearchResponse(query=query, top_k=top_k, min_score=min_score, hits=hits)


@router.get("/staging", response_model=StagingResponse)
async def staging(
    status: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=500)
) -> StagingResponse:
    try:
        rows = await repository.list_staging(status=status, limit=limit)
    except SQLAlchemyError as exc:
        logger.exception("staging の取得に失敗")
        raise HTTPException(
            status_code=503,
            detail="データベースを一時的に利用できません。しばらくしてからもう一度お試しください",
        ) from exc
    stats = None
    try:
        stats = await repository.staging_stats()
    except Exception:
        logger.exception("staging_stats に失敗")
    return StagingResponse(
        rows=[
            {
                "id": r.id, "batch_no": r.batch_no, "source_ref": r.source_ref,
                "question": r.question, "answer": r.answer, "status": r.status,
            }
            for r in rows
        ],
        stats=stats,
    )
