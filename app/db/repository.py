"""データアクセス層。tools (Task 7-8) と orchestrator (Task 11) から呼ばれる。

重要: `app.db.base.async_session` は module attribute 経由でしか参照しないこと
(`import app.db.base as db` して `db.async_session()` の形で使う)。トップレベルで
`from app.db.base import async_session` のように束縛すると、tests/conftest.py の
`monkeypatch.setattr("app.db.base.async_session", ...)` (db_session_factory fixture)
がこのモジュール内の参照には反映されず、テストが本番の support データベースを
静かに触ってしまう。
"""

from datetime import datetime

from sqlalchemy import func, select, text

import app.db.base as db
from app.db.models import (
    Conversation,
    Faq,
    KnowledgeChunk,
    Message,
    QaExtractionStaging,
    Ticket,
)
from app.kb.dedup import normalize_question

_TICKET_SEQ = 0


def _gen_ticket_no() -> str:
    global _TICKET_SEQ
    _TICKET_SEQ += 1
    return f"T{datetime.now():%Y%m%d%H%M%S}{_TICKET_SEQ:03d}"


async def create_conversation(user_id: str) -> int:
    async with db.async_session() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        await s.commit()
        return conv.id


async def get_conversation(conversation_id: int) -> Conversation | None:
    async with db.async_session() as s:
        return await s.get(Conversation, conversation_id)


async def append_message(
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with db.async_session() as s:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        s.add(msg)
        await s.commit()
        return msg.id


async def list_messages(conversation_id: int) -> list[Message]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())


async def search_faq(keyword: str) -> list[Faq]:
    # keyword は LLM が渡す自由文字列。素朴な部分一致検索(言い回しの違いを拾えない
    # 意味的な限界)はこの章では意図的にそのままにしているが、LIKE のメタ文字
    # '%' '_' を未エスケープで埋め込むと、それらの記号を含まない無関係な行にまで
    # 誤ヒットする(例: "50%オフ" が "50Xオフセール中です" にヒットしてしまう)。
    # これは意図した挙動ではなくバグなので、メタ文字はエスケープする。
    # エスケープ文字にはバックスラッシュではなく '!' を使う: バックスラッシュは
    # MySQL がデフォルト(NO_BACKSLASH_ESCAPES 未設定)で文字列リテラル内でも
    # エスケープ文字として扱うため、`LIKE ... ESCAPE '\\'` のような SQL リテラルを
    # 書こうとすると二重のエスケープ処理がぶつかって構文エラーになる(実機の
    # MySQL で確認済み)。'!' はそのような特別な意味を持たないため、この種の
    # 混乱が起きない。
    escaped = keyword.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    async with db.async_session() as s:
        result = await s.execute(
            select(Faq).where(Faq.question.like(f"%{escaped}%", escape="!"))
        )
        return list(result.scalars())


async def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
    ticket_no = _gen_ticket_no()
    async with db.async_session() as s:
        s.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "escalated"
        await s.commit()
    return ticket_no


async def insert_knowledge_chunk(
    category: str, questions: str, answer: str,
    section_path: str | None = None, content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session() as s:
        row = KnowledgeChunk(
            category=category, questions=questions, answer=answer,
            section_path=section_path, content_type=content_type,
            is_key_clause=is_key_clause,
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"
            await s.commit()


async def set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id
            await s.commit()


async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())


async def list_all_questions() -> list[str]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions))
        return list(result.scalars())


async def insert_staging(batch_no: str, source_ref: str | None, question: str, answer: str) -> int:
    async with db.async_session() as s:
        row = QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())


async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(QaExtractionStaging, i)
            if row is not None:
                row.status = status
        await s.commit()


async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    async with db.async_session() as s:
        conv_ids = list((await s.execute(select(Conversation.id).order_by(Conversation.id))).scalars())
        out = []
        for cid in conv_ids:
            msgs = list((await s.execute(
                select(Message).where(Message.conversation_id == cid).order_by(Message.id)
            )).scalars())
            out.append((cid, msgs))
        return out


# ---------------------------------------------------------------------------
# /kb 管理画面の読み取り側集計 (Task 17)
# ---------------------------------------------------------------------------


async def knowledge_stats() -> dict:
    """knowledge_chunks の件数まとめ。1 クエリで total / pending / done / 重要条項を出す。

    ステータス別に COUNT を撃ち直すと、その間に別の書き込みが入って total != pending + done
    という「どの瞬間にも存在しなかった数字」を画面に出しうる。集計は 1 文にまとめる。
    """
    async with db.async_session() as s:
        row = (await s.execute(
            select(
                func.count().label("total"),
                func.sum(
                    func.if_(KnowledgeChunk.vectorize_status == "pending", 1, 0)
                ).label("pending"),
                func.sum(
                    func.if_(KnowledgeChunk.vectorize_status == "done", 1, 0)
                ).label("done"),
                func.sum(KnowledgeChunk.is_key_clause).label("key_clauses"),
            ).select_from(KnowledgeChunk)
        )).one()
        # 0 件のとき SUM は NULL を返す。int(None) は落ちるので 0 に畳む
        return {
            "total": int(row.total),
            "pending": int(row.pending or 0),
            "done": int(row.done or 0),
            "key_clauses": int(row.key_clauses or 0),
        }


async def conversation_stats() -> dict:
    """02 章までのテーブルの件数。/api/admin/overview の「会話」カード用。"""
    async with db.async_session() as s:
        out = {}
        for key, model in (("conversations", Conversation), ("messages", Message),
                           ("tickets", Ticket), ("faq", Faq)):
            out[key] = int(
                (await s.execute(select(func.count()).select_from(model))).scalar_one()
            )
        return out


async def list_recent_chunks(limit: int = 20) -> list[KnowledgeChunk]:
    """最近登録した chunk を id 降順で返す。/kb の「直近の登録」欄用。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit)
        )
        return list(result.scalars())


def chunk_fingerprint(questions: str, answer: str) -> str:
    """重複判定の指紋。questions と answer の**両方**を正規化して連結する。

    質問文だけで突き合わせてはいけない。大きな表は split_table_rows で複数 chunk に
    割れるが、各断片は同じ見出し(= questions)を共有する(実測: after-sales-manual.md の
    「よくある問い合わせの対応時間」が 2 chunk に割れ、questions と section_path は同一で
    answer だけが違う)。質問一致で消すと、表の 2 枚目以降が丸ごと登録されない。
    """
    return normalize_question(questions) + "|" + normalize_question(answer)


async def list_chunk_fingerprints() -> set[str]:
    async with db.async_session() as s:
        rows = await s.execute(
            select(KnowledgeChunk.questions, KnowledgeChunk.answer)
        )
        return {chunk_fingerprint(q, a) for q, a in rows}


async def staging_stats() -> dict:
    """qa_extraction_staging のステータス別件数 + バッチ数。"""
    async with db.async_session() as s:
        rows = (await s.execute(
            select(QaExtractionStaging.status, func.count())
            .group_by(QaExtractionStaging.status)
        )).all()
        batches = int((await s.execute(
            select(func.count(func.distinct(QaExtractionStaging.batch_no)))
        )).scalar_one())
    # 該当行が 0 件のステータスも 0 として必ず出す。キーが欠けると画面側で
    # 「0 件」と「集計に失敗」を区別できなくなる
    out = {k: 0 for k in ("extracted", "kept", "discarded")}
    for status, n in rows:
        out[status] = int(n)
    out["total"] = sum(out[k] for k in ("extracted", "kept", "discarded"))
    out["batches"] = batches
    return out


async def list_staging(status: str | None = None, limit: int = 100) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        stmt = select(QaExtractionStaging).order_by(QaExtractionStaging.id.desc()).limit(limit)
        if status:
            stmt = stmt.where(QaExtractionStaging.status == status)
        return list((await s.execute(stmt)).scalars())


async def clear_knowledge() -> None:
    """03 章の 2 テーブルを空にする(kb-reset ジョブ用)。通常の API 経路からは呼ばない。

    DELETE ではなく TRUNCATE を使う: AUTO_INCREMENT も戻るため、Milvus の
    collection を作り直した直後に id の採番が 1 から揃う(Milvus 側の主キーは
    knowledge_chunks.id なので、ここがずれると再構築後の id が飛ぶ)。
    knowledge_chunks は prev/next の自己参照 FK を持ち、そのままでは TRUNCATE できないので
    FOREIGN_KEY_CHECKS を落とす(tests/conftest.py の後片付けと同じ手順)。
    """
    async with db.async_session() as s:
        await s.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        await s.execute(text("TRUNCATE TABLE qa_extraction_staging"))
        await s.execute(text("TRUNCATE TABLE knowledge_chunks"))
        await s.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        await s.commit()
