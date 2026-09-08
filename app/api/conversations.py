"""会話の一覧と履歴を覗く read-only エンドポイント(07 章)。

長い会話でも文脈が保たれていることを人が確かめるには、会話を切り替えられること
(一覧)と、切り替えた先の中身を読み直せること(履歴)の 2 つが要る。

**このモジュールに書き込みの入口を足さないこと。** ここは受け入れ検証のための
覗き窓で、会話を消したり状態を変えたりできるようにすると、画面の事故がそのまま
DB に残る。会話へ書き込む経路は /api/chat と /api/actions/* だけ。
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.core import labels
from app.db import repository
from app.schemas.conversations import (
    ConversationItem,
    ConversationListResponse,
    MessageItem,
    MessageListResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["conversations"])

_LIST_DOWN_MSG = "会話一覧を一時的に取得できません。しばらくしてからもう一度お試しください"
_HISTORY_DOWN_MSG = "会話履歴を一時的に取得できません。しばらくしてからもう一度お試しください"


@router.get("/api/conversations", response_model=ConversationListResponse)
async def list_conversations(
    user_id: str = Query(min_length=1, description="この利用者の会話だけを返す"),
    limit: int = Query(default=50, ge=1, le=200, description="返す最大件数(新しい順)"),
) -> ConversationListResponse:
    """会話の一覧(新しい順)。画面左の切り替え欄がこれで描かれる。

    user_id は必須にする。既定値を持たせると、指定を忘れたときに全利用者の会話が
    並び、他人の問い合わせ内容が preview として出る。

    limit に上限を付けるのは、1 リクエストで全履歴を引かせないため。
    """
    try:
        rows = await repository.list_conversations(user_id, limit)
    except SQLAlchemyError as exc:
        # 例外そのものはログにだけ残す。detail へ載せると接続文字列や SQL が画面まで届く。
        logger.exception("会話一覧の取得に失敗 user=%s", user_id)
        raise HTTPException(status_code=503, detail=_LIST_DOWN_MSG) from exc
    return ConversationListResponse(items=[
        ConversationItem(
            id=r["id"],
            # 表示名は labels からのみ引く(app/api/actions.py と同じ規約)。
            status=labels.label(labels.CONVERSATION_STATUS, r["status"]),
            preview=r["preview"],
            has_summary=r["has_summary"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ])


@router.get("/api/conversations/{conversation_id}/messages",
            response_model=MessageListResponse)
async def list_conversation_messages(conversation_id: int) -> MessageListResponse:
    """1 つの会話の発話。画面が履歴を描き直すために読む。

    tool の行は返さない(repository.list_dialog_messages が人とサポートの発話だけを
    返す)。生の JSON は画面に出す物ではない。

    会話の有無を先に確かめるのは、存在しない会話に対して「発話が 0 件の会話」を
    返さないため。空の一覧と「そんな会話は無い」は画面で言うべきことが違う。
    確認自体が落ちたら 404 とは言い切らない(在る会話に「見つかりません」と
    答えてしまう)ので、SQLAlchemyError はそのまま 503 へ落とす。
    """
    try:
        conv = await repository.get_conversation(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="会話が見つかりません")
        rows = await repository.list_dialog_messages(conversation_id)
    except SQLAlchemyError as exc:
        logger.exception("会話履歴の取得に失敗 conv=%s", conversation_id)
        raise HTTPException(status_code=503, detail=_HISTORY_DOWN_MSG) from exc
    return MessageListResponse(items=[
        MessageItem(role=m.role, content=m.content, created_at=m.created_at)
        for m in rows
        # 本文の無い行は返さない。log_node は回答が空のターンを content=NULL で
        # 残すので、そのまま返すと空の吹き出しが履歴に毎ターン並ぶ。
        if (m.content or "").strip()
    ])
