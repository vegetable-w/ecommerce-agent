"""査読画面の API。データフライホイールの最後の一段(09 章 spec §8.1)。

低信頼プール → 正規化と重複排除 → **人の判断** → ナレッジベースへ書き戻し、の
「人の判断」から先を受け持つ。承認された答えは 03 章の取り込み経路
(dualwrite.write_pending → vectorize_pending)をそのまま通す。ここに別の
書き込み経路を作らないこと。作ると、重複排除も近傍リンクもベクトル化の再実行も
片方だけ直る形になり、ナレッジベースの状態が経路ごとに食い違う。

**書き込みが済んでから status を動かす。順序を入れ替えないこと。** repository の
update_review_status は `pending` の行しか動かせない(同時に開いた 2 人の判断が
後勝ちで消えるのを防ぐため)。先に `approved` にしてから書き戻すと、書き戻しが
落ちた行は「承認済みなのにナレッジベースには無い」まま固定され、二度と承認できない
= やり直す手段が無くなる。逆順なら、落ちても pending のまま残るので押し直せる。
(ナレッジ側に chunk だけが残るが、あちらは status=pending として再実行で回収される
設計で、承認できない行が残るより害が小さい。)
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.core import labels
from app.db import repository
from app.kb import documents, dualwrite, milvus_client
from app.kb.documents import Chunk
from app.schemas.review import (
    ApproveRequest,
    ApproveResponse,
    RejectResponse,
    ReviewDetailOut,
    ReviewListResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# 査読を通ったナレッジの出所を示す固定値。app/kb/mining.py が会話由来の chunk に
# category="過去の会話" / section_path="mined" を付けるのと同じ扱いで、
# 「どの経路で入った知識か」を後から見分けられるようにする。
# section_path に質問文を入れない: あの列は文書内の位置(VARCHAR(512))であって
# 質問の置き場ではなく、512 文字の normalized_question を継ぐと DataError になる。
_CATEGORY = "フライホイール"
_SECTION_PATH = "flywheel"
# 承認された答えは正規化済みの質問と 1 対 1 の Q&A なので faq として入れる。
_CONTENT_TYPE = "faq"

_DB_DOWN = "データベースを一時的に利用できません。しばらくしてからもう一度お試しください"


async def _write_chunks(chunks: list[Chunk]) -> list[int]:
    """03 章の取り込み経路(MySQL 側)。テストはこの縫い目を差し替える。"""
    return await dualwrite.write_pending(chunks)


async def _vectorize() -> int:
    """03 章の取り込み経路(Milvus 側)。

    client の用意を asyncio.to_thread へ逃がすのは app/api/kb.py の
    /api/kb/vectorize と同じ作法。pymilvus は同期の gRPC なので、
    接続の確立(と collection の検査・ロード)は event loop の上でやらない。

    dualwrite 側は触っていない。あの中の upsert / flush は同期呼び出しのままで、
    server process から呼ぶと event loop を塞ぐが、それは /api/kb/vectorize と
    retrieval.search_knowledge が既にやっていることで、この経路が持ち込む問題では
    ない(しかも承認 1 件で流れる chunk は 1 件)。塞ぐのを本当に直すなら
    Milvus を触る全経路をまとめて直す話になるので、査読 API の都合で
    片方だけ別の作法へ寄せない。
    """
    client = await asyncio.to_thread(milvus_client.get_client)
    await asyncio.to_thread(milvus_client.ensure_collection, client)
    return await dualwrite.vectorize_pending(client)


def _item_out(r) -> dict:
    return {
        "id": r.id,
        "normalized_question": r.normalized_question,
        "ai_suggested_answer": r.ai_suggested_answer,
        "occurrence_count": r.occurrence_count,
        "review_status": r.review_status,
        "status_label": labels.label(labels.REVIEW_STATUS, r.review_status),
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


async def _detail_or_404(review_id: int):
    try:
        detail = await repository.get_review_detail(review_id)
    except SQLAlchemyError as exc:
        # 例外そのものは log にだけ残す。detail へ載せると接続文字列や SQL が
        # 画面まで届く(app/api/actions.py と同じ規約)。
        logger.exception("ナレッジの穴を読めなかった review=%s", review_id)
        raise HTTPException(status_code=503, detail=_DB_DOWN) from exc
    if detail is None:
        raise HTTPException(status_code=404, detail="そのナレッジの穴は見つかりません")
    return detail


# /queue は /{review_id} より**前に**置く。後ろに置くと "queue" が review_id として
# 解釈され、一覧の取得が 422 になる(経路の宣言順がそのまま照合順になる)。
@router.get("/api/review/queue", response_model=ReviewListResponse)
async def review_queue(status: str | None = None) -> dict:
    """査読待ちの一覧。occurrence_count 降順(よく来る穴が上)。status 省略で全件。"""
    try:
        rows = await repository.list_review_queue(status)
    except SQLAlchemyError as exc:
        logger.exception("査読キューを読めなかった status=%s", status)
        raise HTTPException(status_code=503, detail=_DB_DOWN) from exc
    return {"items": [_item_out(r) for r in rows]}


@router.get("/api/review/{review_id}", response_model=ReviewDetailOut)
async def review_detail(review_id: int) -> dict:
    """1 件の詳細。**生の質問と、そのときの検索の写しを必ず付ける。**

    正規化後の 1 行だけを見て承認すると、正規化が意味を削っていた場合
    (注文番号や条件が落ちた場合)に気づけない。写しは「ナレッジが無いのか、
    有るのに引けていないのか」を査読者が見分けるための材料。
    """
    item, raws = await _detail_or_404(review_id)
    out = _item_out(item)
    out["approved_answer"] = item.approved_answer
    out["raws"] = [
        {
            "raw_question": r.raw_question,
            "source": r.source,
            "reason": r.reason,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "retrieved_chunks": r.retrieved_chunks,
        }
        for r in raws
    ]
    return out


@router.post("/api/review/{review_id}/approve", response_model=ApproveResponse)
async def approve(review_id: int, req: ApproveRequest) -> dict:
    """承認して、答えをナレッジベースへ書き戻す。

    ナレッジベースへ入るのは **査読者が確定させた答え**(req.approved_answer)。
    モデルの参考回答(ai_suggested_answer)ではない。参考回答をそのまま書くなら
    人の査読を挟む意味が無い。
    """
    item, _ = await _detail_or_404(review_id)
    if item.review_status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"この項目は既に「{labels.label(labels.REVIEW_STATUS, item.review_status)}」です",
        )

    chunk = Chunk(
        category=_CATEGORY,
        questions=item.normalized_question,
        answer=req.approved_answer,
        section_path=_SECTION_PATH,
        content_type=_CONTENT_TYPE,
        # 見出しに相当するのは正規化済みの質問。documents.py と同じ判定器を使う
        # (重要条項の語彙を 2 箇所に持たない)。
        is_key_clause=documents._is_key(item.normalized_question),
    )
    try:
        chunk_ids = await _write_chunks([chunk])
        await _vectorize()
    except Exception as exc:
        # 例外の型を絞らない。ここから先は MySQL・埋め込み上流・Milvus の 3 者が
        # 絡むので、型を並べると並べ忘れた 1 つが 500 になって画面へ traceback が出る。
        logger.exception(
            "ナレッジベースへの書き戻しに失敗 review=%s(status は pending のまま。押し直せる)",
            review_id,
        )
        raise HTTPException(
            status_code=502,
            detail="ナレッジベースへの書き戻しに失敗しました。判定は未確定のままなので、もう一度承認できます",
        ) from exc

    try:
        moved = await repository.update_review_status(
            review_id, "approved", approved_answer=req.approved_answer
        )
    except SQLAlchemyError as exc:
        logger.exception("承認の記録に失敗 review=%s", review_id)
        raise HTTPException(status_code=503, detail=_DB_DOWN) from exc
    if not moved:
        # 詳細を読んでからここへ来るまでの間に、別の査読者が判定を確定させた。
        # 書き戻し自体は済んでいるので、それを隠さずに伝える(黙って 200 を返すと、
        # 画面には自分の答えが入ったように見えるが、保存されているのは相手の判定)。
        logger.warning("承認を記録できなかった review=%s(別の査読者が先に確定させた)", review_id)
        raise HTTPException(
            status_code=409,
            detail="別の担当者が先に判定したため、この承認は記録されませんでした(ナレッジベースへの書き戻しは完了しています)",
        )

    logger.info(
        "承認 review=%s → knowledge_chunks %s(ベクトル化まで完了。次の質問から検索で当たる)",
        review_id, chunk_ids,
    )
    return {"ok": True, "chunk_ids": chunk_ids}


@router.post("/api/review/{review_id}/reject", response_model=RejectResponse)
async def reject(review_id: int) -> dict:
    """却下する。**ナレッジベースには何も書かない。**

    却下は最終判断で、復活の経路は用意しない(spec §4: 同じ質問が再び来ても
    occurrence_count が進むだけで status は戻らない)。
    """
    try:
        moved = await repository.update_review_status(review_id, "rejected")
    except SQLAlchemyError as exc:
        logger.exception("却下の記録に失敗 review=%s", review_id)
        raise HTTPException(status_code=503, detail=_DB_DOWN) from exc
    if not moved:
        # 動かせなかった理由は 2 つある(行が無い / 既に判定済み)。UPDATE の
        # rowcount だけでは区別できないので、ここで初めて読みに行く。
        item, _ = await _detail_or_404(review_id)
        raise HTTPException(
            status_code=409,
            detail=f"この項目は既に「{labels.label(labels.REVIEW_STATUS, item.review_status)}」です",
        )
    logger.info("却下 review=%s(ナレッジベースへは書き戻さない)", review_id)
    return {"ok": True}
