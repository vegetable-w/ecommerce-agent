"""データアクセス層。tools (Task 7-8) と orchestrator (Task 11) から呼ばれる。

重要: `app.db.base.async_session` は module attribute 経由でしか参照しないこと
(`import app.db.base as db` して `db.async_session()` の形で使う)。トップレベルで
`from app.db.base import async_session` のように束縛すると、tests/conftest.py の
`monkeypatch.setattr("app.db.base.async_session", ...)` (db_session_factory fixture)
がこのモジュール内の参照には反映されず、テストが本番の support データベースを
静かに触ってしまう。
"""

import logging
from datetime import datetime

from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError

import app.db.base as db
from app.db.models import (
    Conversation,
    ConversationSummary,
    EvalRun,
    FaithCase,
    Faq,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    ReviewQueue,
    Ticket,
    ToolAuditLog,
)
from app.kb.dedup import normalize_question

logger = logging.getLogger(__name__)

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


async def list_pending_chunks(chunk_ids: list[int] | None = None) -> list[KnowledgeChunk]:
    """ベクトル化待ちの chunk。既定は DB 全体の pending。

    chunk_ids を渡すと、その id に限って pending を返す(09 章)。レビューでの承認は
    自分が書いた 1 件だけをベクトル化したいが、取り込み(make kb-vectorize)は
    全 pending を拾わなければならない。既定を None のままにしてあるのは、
    後者の呼び出し元(03/04 章の script と job)の振る舞いを変えないため。
    空の list は「対象なし」であって全件ではない(id で絞る側が 0 件になったときに
    黙って全件へ広がると、承認 1 件で数百件の埋め込みが走る)。
    """
    async with db.async_session() as s:
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        if chunk_ids is not None:
            stmt = stmt.where(KnowledgeChunk.id.in_(chunk_ids))
        result = await s.execute(stmt)
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


async def list_all_chunks() -> list[KnowledgeChunk]:
    """全 chunk を id 順で返す。md との差分を取る repatch から使う。"""
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk).order_by(KnowledgeChunk.id))
        return list(result.scalars())


async def update_knowledge_chunk(
    chunk_id: int, category: str, questions: str, answer: str,
    section_path: str | None, content_type: str | None, is_key_clause: int,
) -> bool:
    """本文を書き換え、status を pending に戻す。id は変えない。

    id を保つのは、Milvus の PK が chunk id そのものだから。同じ id で upsert すれば
    古いベクトルがその場で置き換わり、消し忘れた古い行が検索に残ることがない
    (新しい id を振ると、古い行を別途消さない限り二重に当たる)。
    """
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is None:
            return False
        row.category = category
        row.questions = questions
        row.answer = answer
        row.section_path = section_path
        row.content_type = content_type
        row.is_key_clause = is_key_clause
        row.vectorize_status = "pending"
        await s.commit()
        return True


async def delete_knowledge_chunk(chunk_id: int) -> bool:
    """1 行だけ消す。md から節が消えたときに使う。

    前後リンクはこの関数では触らない。呼ぶ側が文書単位で張り直す
    (1 件ずつ繋ぎ直すと、消した行を指したままの中間状態が残る)。
    """
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is None:
            return False
        await s.delete(row)
        await s.commit()
        return True


async def list_chunk_sections() -> list[tuple[str, str]]:
    """全 chunk の (section_path, answer)。評価セットの検証(読み取り専用)から使う。

    正となるナレッジは Milvus ではなく MySQL 側なので、こちらを読む。Milvus が
    落ちていても評価セットの妥当性は確かめられる、という切り分けのため。
    """
    async with db.async_session() as s:
        rows = await s.execute(
            select(KnowledgeChunk.section_path, KnowledgeChunk.answer)
            .order_by(KnowledgeChunk.id)
        )
        return [(path or "", answer or "") for path, answer in rows]


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


async def find_chunk_id_by_fingerprint(questions: str, answer: str) -> int | None:
    """同じ指紋の chunk が既にあればその id。無ければ None(09 章)。

    判定そのものは chunk_fingerprint に任せる(03 章の取り込みと同じ規則)。
    id まで返すのは、レビューでの承認をやり直したときに「書かない」だけでは足りないため。
    一度目で MySQL への書き込みが済んでベクトル化だけが落ちた行は pending のまま
    残っており、二度目の承認はその**既にある行**をベクトル化しなければ、承認済みなのに
    検索へ出てこない知識ができる。集合(list_chunk_fingerprints)では拾えない。
    """
    target = chunk_fingerprint(questions, answer)
    async with db.async_session() as s:
        rows = await s.execute(
            select(KnowledgeChunk.id, KnowledgeChunk.questions, KnowledgeChunk.answer)
            .order_by(KnowledgeChunk.id)
        )
        for cid, q, a in rows:
            if chunk_fingerprint(q, a) == target:
                return cid
    return None


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


async def _insert_low_confidence(
    conversation_id: int | None,
    raw_question: str,
    source: str,
    reason: str | None,
    retrieved_chunks: list | None,
) -> int:
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
            retrieved_chunks=retrieved_chunks,
        )
        s.add(row)
        await s.commit()
        return row.id


async def insert_low_confidence(
    conversation_id: int | None,
    raw_question: str,
    source: str,
    reason: str | None,
    retrieved_chunks: list | None = None,
) -> int:
    """回答を断った質問を低信頼プールへ積み、その id を返す。

    source は DDL の ENUM('retrieval_low_conf','self_check','user_feedback') に従う。

    retrieved_chunks は投入時点の検索結果の写し(09 章)。既定 None で末尾に足して
    あるのは、既存の呼び出し元(04 章の回答拒否パスと自己点検パス)を 4 引数のまま
    通すため。検索を通らない経路は渡さなくてよい。

    conversation_id は conversations への FK であり、存在しない id を渡すと
    IntegrityError になる。ただし呼び出し元(04 章の回答拒否パス)は「根拠が足りないので
    正直に断る」という正常系の途中でここを呼ぶ。ここで例外を投げると、穏当な回答拒否が
    そのまま 500 に化けてしまう。プールにとっての本体は質問文であり、会話への紐付けは
    後から辿るための手掛かりに過ぎないので、紐付けだけを捨てて質問文を残す。
    """
    try:
        return await _insert_low_confidence(
            conversation_id, raw_question, source, reason, retrieved_chunks
        )
    except IntegrityError:
        if conversation_id is None:
            raise
        logger.warning(
            "conversation_id=%s が見つからないため、会話に紐づけずにプールへ積む",
            conversation_id,
        )
        return await _insert_low_confidence(
            None, raw_question, source, reason, retrieved_chunks
        )


async def low_confidence_exists(conversation_id: int, raw_question: str) -> bool:
    """その会話で同じ質問が既にプールへ入っているか(09 章)。

    source は見ない。プールの 1 行は「直すべき質問」1 件であって「起きた出来事」の
    記録ではないので、同じ会話の同じ質問が別の source で 2 行になると、
    flywheel がそれを 1 つの穴へまとめて occurrence_count を 2 にする
    = レビューの優先度が 1 ターンで二重に重み付けされる。
    """
    async with db.async_session() as s:
        found = await s.execute(
            select(LowConfidenceQuestion.id)
            .where(LowConfidenceQuestion.conversation_id == conversation_id,
                   LowConfidenceQuestion.raw_question == raw_question)
            .limit(1)
        )
        return found.first() is not None


# ---------------------------------------------------------------------------
# 幻覚ケース台帳 (Task 19)
# ---------------------------------------------------------------------------

# DDL の ENUM と同じ並び。画面のタブもこの順で出す。
FAITH_STATUSES = ("unresolved", "resolved", "no_action_needed")
# 「人が一度目を通して片を付けた」状態。再び幻覚と判定されたら unresolved へ戻す対象で
# あり、対処メモ(resolution)が必須になる状態でもある。
FAITH_CLOSED_STATUSES = ("resolved", "no_action_needed")
# resolution は VARCHAR(300)。ここで弾かないと MySQL の DataError がそのまま 500 になる
FAITH_RESOLUTION_MAX = 300


def _faith_now() -> datetime:
    """台帳に入れる現在時刻。秒未満は落とす。

    時刻を Python 側で作るのは、first_seen_at と last_seen_at の片方だけ
    server_default(CURRENT_TIMESTAMP)に任せると、アプリと MySQL の時計や
    タイムゾーンがずれたときに並び順が壊れるため。

    秒未満を落とすのは、対象の列が DATETIME(小数秒の精度なし)で、MySQL が
    保存時に **四捨五入** するから。13:14:40.77 を渡すと DB には 13:14:41 が入り、
    書き込み直後に返した行(Python 側の値)と、次に読み直した行の時刻が食い違う。
    """
    return datetime.now().replace(microsecond=0)


async def upsert_faith_case(
    eval_id: str,
    *,
    bucket: str,
    query: str,
    answer: str,
    reason: str,
    strategy: str = "hybrid_rerank",
    citations: list | None = None,
    judge_model: str | None = None,
) -> dict:
    """幻覚と判定された問いを台帳へ積む。eval_id ごとに 1 行。

    eval_id 以外をキーワード専用にしているのは取り違え防止のため。query / answer /
    reason はどれも日本語の長い文字列で、位置引数で並べると入れ替わっても型では気づけず、
    台帳が静かに嘘をつく。

    既存行があれば **今回の実行の内容で上書き**する。台帳の 1 行が指すのは
    「最後に幻覚と判定されたときの姿」であって、初回の姿ではない。判定を見直す人が
    見たいのは最新の回答と最新の根拠だからである(初回の姿を残したいなら行を増やす
    設計になるが、それでは「同じ問いが何度も幻覚になっている」が数えられない)。

    戻り値は呼び出し元(評価スクリプト)が新規 / 再発 / 継続中を区別するための dict:

        {"id", "eval_id", "status", "seen_count", "created", "recurred", "previous_status"}

    - created=True                      … 初めて幻覚と判定された
    - created=False かつ recurred=True  … 対処済みだったものが再発した
    - created=False かつ recurred=False … 未対処のまま続いている

    同一 eval_id への同時 upsert は想定していない(1 回の実行の中で eval_id は一意で、
    評価が二重に走ることもない)。万一衝突すれば uk_eval_id の IntegrityError が
    呼び出し元へ上がる。台帳への書き込み失敗はログ 1 行に留めてレポートには影響させない、
    というのが呼び出し側の約束なので、ここで握り潰して二重計上を作らない。
    """
    now = _faith_now()
    async with db.async_session() as s:
        row = (
            await s.execute(select(FaithCase).where(FaithCase.eval_id == eval_id))
        ).scalar_one_or_none()

        if row is None:
            row = FaithCase(
                eval_id=eval_id, bucket=bucket, query=query, answer=answer, reason=reason,
                strategy=strategy, citations=citations, judge_model=judge_model,
                status="unresolved", seen_count=1, first_seen_at=now, last_seen_at=now,
            )
            s.add(row)
            await s.commit()
            return {
                "id": row.id, "eval_id": eval_id, "status": "unresolved", "seen_count": 1,
                "created": True, "recurred": False, "previous_status": None,
            }

        previous_status = row.status
        row.bucket = bucket
        row.query = query
        row.answer = answer
        row.reason = reason
        row.strategy = strategy
        row.citations = citations
        row.judge_model = judge_model
        row.seen_count = row.seen_count + 1
        row.last_seen_at = now

        recurred = previous_status in FAITH_CLOSED_STATUSES
        if recurred:
            # 対処済みのはずが再び幻覚になった。未対処へ戻して作業一覧に再浮上させる。
            # これを落とすと「解決済み」の顔をしたまま再発が埋もれ、台帳を見ても
            # 何も起きていないように見える。
            row.status = "unresolved"
            # resolved_at はクリアしない(DDL のコメントの通り、いつ一度片が付いた
            # ことになっていたかが再発の判断材料になる)。
            # resolution も**残す**。前回どう直したつもりだったかは、再発したときに
            # 最初に読みたい情報であり、消すとその手掛かりを自ら捨てることになる。
            # 人手で unresolved へ戻す set_faith_case_status とは扱いが違う:
            # あちらは人が「その対処は間違いだった」と言っているので消す。
        await s.commit()
        return {
            "id": row.id, "eval_id": eval_id, "status": row.status,
            "seen_count": row.seen_count, "created": False, "recurred": recurred,
            "previous_status": previous_status,
        }


async def faith_case_status_map() -> dict[str, str]:
    """台帳全体の eval_id -> status。読み取り専用。

    評価スクリプトが「今回の実行で幻覚と判定された問いが、台帳ではどう扱われているか」を
    引くために使う。ケースごとに 1 件ずつ問い合わせると、幻覚が増えた実行ほど問い合わせが
    増えて評価の最後だけが遅くなるので、まとめて 1 回で読む。台帳の行数は多くても
    評価セットの問題数(300)程度で、値も status の短い文字列だけなので全件を載せてよい。
    """
    async with db.async_session() as s:
        rows = (await s.execute(select(FaithCase.eval_id, FaithCase.status))).all()
    return {eval_id: status for eval_id, status in rows}


async def list_faith_cases(
    status: str | None = None, page: int = 1, size: int = 20
) -> dict:
    """台帳の一覧 1 ページぶん。

    並び順は「未対処 → last_seen_at の新しい順 → id の大きい順」。未対処を先頭に置くのは
    この画面が作業一覧だから。last_seen_at は DATETIME(秒精度)なので同じ秒に積まれた
    行では引き分けになる。その場合に順序が実行ごとに揺れてページ送りで行が重複したり
    消えたりしないよう、id で決着を付ける(id の大きい方が後に積まれた行)。

    counts は **フィルタにもページにも依存しない全体の件数**。画面のタブに出す数字で
    あり、「未対処 3 件」タブを開いた状態で resolved タブが 0 件に見えてはいけない。
    total の方はフィルタ適用後の件数(ページ送りの母数)。
    """
    page = max(1, int(page))
    size = max(1, int(size))
    async with db.async_session() as s:
        rows = (
            await s.execute(
                select(FaithCase.status, func.count()).group_by(FaithCase.status)
            )
        ).all()
        # 0 件のステータスも 0 として必ず出す。キーが欠けると画面側で
        # 「0 件」と「集計に失敗」を区別できなくなる(staging_stats と同じ方針)
        counts = {k: 0 for k in FAITH_STATUSES}
        for st, n in rows:
            counts[st] = int(n)
        counts["total"] = sum(counts[k] for k in FAITH_STATUSES)

        stmt = select(FaithCase)
        total_stmt = select(func.count()).select_from(FaithCase)
        if status:
            stmt = stmt.where(FaithCase.status == status)
            total_stmt = total_stmt.where(FaithCase.status == status)
        total = int((await s.execute(total_stmt)).scalar_one())

        stmt = stmt.order_by(
            func.if_(FaithCase.status == "unresolved", 0, 1),
            FaithCase.last_seen_at.desc(),
            FaithCase.id.desc(),
        ).offset((page - 1) * size).limit(size)
        items = list((await s.execute(stmt)).scalars())

    return {
        "rows": items,
        "total": total,
        "page": page,
        "size": size,
        # 0 件のときは 0 ページ。画面で「1 / 0」と出したくなければ表示側で丸めること
        "pages": (total + size - 1) // size,
        "status": status,
        "counts": counts,
    }


async def set_faith_case_status(
    case_id: int, status: str, resolution: str | None = None
) -> FaithCase | None:
    """台帳の状態を人手で変える。存在しない id なら None(呼び出し元が 404 にする)。

    入力の不正は ValueError で返す。検査をここ 1 か所に置くことで、API からでも
    スクリプトからでも同じ規則が効く(API 側はこれを 400 に写す)。
    """
    if status not in FAITH_STATUSES:
        raise ValueError("status は " + " / ".join(FAITH_STATUSES) + " のいずれかです")

    # strip() は全角スペース(U+3000)も落とす。日本語の入力では素で打てる文字なので、
    # strip(" ") のように半角だけにすると「　」1 文字の対処メモが通ってしまう。
    note = (resolution or "").strip()
    if status in FAITH_CLOSED_STATUSES:
        if not note:
            raise ValueError(
                "対処メモを入力してください（どう直したか / なぜ直さなくてよいか）"
            )
        if len(note) > FAITH_RESOLUTION_MAX:
            raise ValueError(f"対処メモは {FAITH_RESOLUTION_MAX} 文字以内で入力してください")

    async with db.async_session() as s:
        row = await s.get(FaithCase, case_id)
        if row is None:
            return None
        row.status = status
        if status in FAITH_CLOSED_STATUSES:
            row.resolution = note
            row.resolved_at = _faith_now()
        else:
            # 人が「やはり未対処」と言った以上、その対処メモはもう有効ではないので消す。
            # 再発(upsert_faith_case)で unresolved へ戻る場合とは扱いが違う。
            row.resolution = None
            row.resolved_at = None
        await s.commit()
        return row


# ---------------------------------------------------------------------------
# 07 会話の要約(スライディングウィンドウの境界)
# ---------------------------------------------------------------------------


async def count_messages_after(conversation_id: int, after_id: int | None) -> int:
    """前回の要約以降に増えた message の件数。要約を起動するかの判定材料。

    after_id が空なら「まだ一度も要約していない」なので全件を数える。
    """
    async with db.async_session() as s:
        q = (select(func.count()).select_from(Message)
             .where(Message.conversation_id == conversation_id))
        if after_id:
            q = q.where(Message.id > after_id)
        return int((await s.execute(q)).scalar_one())


async def list_dialog_messages(conversation_id: int) -> list[Message]:
    """人とサポートの発話だけを id 昇順で返す。要約に渡す素材。

    tool の行を外すのは、生の JSON を要約させても事実が増えず、
    かえって注文番号のような数字が紛れ込んで誤要約の元になるため。
    """
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id,
                   Message.role.in_(("user", "assistant")))
            .order_by(Message.id)
        )
        return list(result.scalars())


async def append_summary_fragment(conversation_id: int, from_msg_id: int,
                                  upto_msg_id: int, content: str) -> int:
    """要約の断片を 1 行**追記**し、振った seq を返す。既存の行は書き換えない。

    seq は会話ごとの通し番号(1 始まり)で、既存の最大 + 1 を採る。同じ会話の要約は
    同時に走らせない(app/core/summarizer.py が走行中の会話を弾く)ので、
    読んで書くまでの間に別の断片が割り込むことはない。

    conversations.summary の方は上書きされていくが、こちらは各区間を圧縮した時点の
    姿がそのまま残る。ある事実がどの区間で落ちたかは、この並びを追って初めて分かる。
    """
    async with db.async_session() as s:
        current = (await s.execute(
            select(func.max(ConversationSummary.seq))
            .where(ConversationSummary.conversation_id == conversation_id)
        )).scalar()
        seq = int(current or 0) + 1
        s.add(ConversationSummary(
            conversation_id=conversation_id, seq=seq,
            from_msg_id=from_msg_id, upto_msg_id=upto_msg_id, content=content,
        ))
        await s.commit()
        return seq


async def update_conversation_summary(conversation_id: int, summary: str,
                                      upto_msg_id: int) -> None:
    """要約と、その要約が覆う範囲を**一緒に**書く。

    別々に書くと、要約は新しいのに境界が古い(同じ内容が窓にも要約にも出る)、
    あるいは境界だけ進んで要約が古い(その間の発話がどこにも残らない)という
    ずれ方をする。後者は取り返しがつかない。

    会話が消えていても落とさない。要約は非同期で走るので、書き戻す頃には
    会話が消えていることがありうる。
    """
    async with db.async_session() as s:
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.summary = summary
            conv.summary_upto_msg_id = upto_msg_id
            await s.commit()


# ---------------------------------------------------------------------------
# 07 会話の一覧(受け入れ検証の sidebar)
# ---------------------------------------------------------------------------

# 一覧の 1 行に収まる長さ。全文を送ると sidebar が最初の質問の長文で埋まる。
_PREVIEW_CHARS = 40


async def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    """その利用者の会話一覧(新しい順)。最初の質問の抜粋と、要約済みかの印を添える。

    画面で会話を切り替えるためだけの読み取り。ORM の行ではなく dict を返すのは、
    preview と has_summary が**組み立てた値**で、Conversation の列に対応しないため。

    preview は「最初の」ユーザー発話。最後の発話にすると一覧が毎ターン書き換わり、
    どの会話だったかを目で追えなくなる。要約本文は載せない(数百文字あるので、
    一覧の応答が要約で埋まる)。載せるのは有無の印だけ。

    **会話ごとに問い合わせを往復しない。** 1 件ずつ最初の発話を引くと 50 件で
    51 クエリになる。会話をまとめて引いてから、その id 群に対する最初の user 行を
    1 回で引く(合計 2 クエリ。会話の件数が増えても増えない)。

    status は DB の英語識別子のまま返す。日本語への変換は API 層の仕事で、
    ここで labels を引くと日本語の表示名がデータ層から漏れ出す。
    """
    async with db.async_session() as s:
        convs = list((await s.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.id.desc())
            .limit(limit)
        )).scalars())
        if not convs:
            return []

        # 会話ごとの「最初の user 行」の id を 1 回で求め、その本文だけを引く。
        first_ids = (
            select(func.min(Message.id))
            .where(Message.conversation_id.in_([c.id for c in convs]),
                   Message.role == "user")
            .group_by(Message.conversation_id)
        )
        previews = {
            cid: (content or "")[:_PREVIEW_CHARS]
            for cid, content in (await s.execute(
                select(Message.conversation_id, Message.content)
                .where(Message.id.in_(first_ids))
            )).all()
        }

    return [
        {
            "id": c.id,
            "status": c.status,
            # まだ発話の無い会話も一覧に出す(preview は空)。落とすと、作った直後の
            # 会話が sidebar から消え、いま居る場所が一覧に無いことになる。
            "preview": previews.get(c.id, ""),
            "has_summary": bool(c.summary),
            "updated_at": c.updated_at,
        }
        for c in convs
    ]


# ---------------------------------------------------------------------------
# tool 監査 (08 Task 1)
# ---------------------------------------------------------------------------


async def insert_tool_audit(
    conversation_id: int | None,
    tool_call_id: str | None,
    tool_name: str,
    tool_source: str,
    mcp_server: str | None,
    arguments: dict | None,
    result_summary: str | None,
    status: str,
    error_message: str | None,
    retry_count: int,
    duration_ms: int | None,
) -> None:
    """tool 呼び出しの監査記録を 1 件保存する。

    status は DDL の ENUM('success','failed','timeout','validation_blocked',
    'permission_denied') に従う英語識別子。日本語は app/core/labels.py の
    TOOL_AUDIT_STATUS でのみ対応付ける。

    例外はここで握り潰さず、そのまま呼び出し元(execution engine)へ返す。
    監査の書き込みが失敗しても tool の実行自体は止めてはいけないが、その判断は
    engine 側の責務で、データ層が黙って失敗を飲み込むと「記録が無い」ことに
    誰も気づけなくなる。
    """
    async with db.async_session() as s:
        s.add(
            ToolAuditLog(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_source=tool_source,
                mcp_server=mcp_server,
                arguments=arguments,
                result_summary=result_summary,
                status=status,
                error_message=error_message,
                retry_count=retry_count,
                duration_ms=duration_ms,
            )
        )
        await s.commit()


# ---------------------------------------------------------------------------
# データフライホイール (09 Task 1)
#
# 低信頼プール(生の質問) → 正規化と重複排除 → review_queue(人がレビューする単位)
# → 承認された答えを 03 章の取り込み経路でナレッジベースへ戻す、という閉ループの
# データ層。ここは pipeline と API の両方から呼ばれる。
# ---------------------------------------------------------------------------


async def fetch_unmatched_low_conf(limit: int) -> list[LowConfidenceQuestion]:
    """まだ review_queue へまとめられていない低信頼質問を、古い順に取る。

    id 昇順にするのは、pipeline を何度回しても同じ質問から処理され、
    limit で切ったときの積み残しが「新しい方」に揃うようにするため。
    降順にすると、古い質問が limit の外に残り続けて永久に処理されない。
    """
    async with db.async_session() as s:
        stmt = (
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(LowConfidenceQuestion.id.asc())
            .limit(limit)
        )
        return list((await s.execute(stmt)).scalars())


async def list_review_candidates(limit: int = 200) -> list[dict]:
    """重複排除の突き合わせ相手。id と正規化済みの質問文だけを返す。

    行全体ではなく 2 列に絞るのは、この結果が候補の一覧としてモデルの prompt に
    載るため。answer 本文まで積むと候補数に比例して無駄に token を食う。

    updated_at 降順にするのは、直近に触れた穴ほど今の質問と重なりやすいため。
    limit で切られるのは古い穴の側になる。
    """
    async with db.async_session() as s:
        stmt = (
            select(ReviewQueue.id, ReviewQueue.normalized_question)
            .order_by(ReviewQueue.updated_at.desc())
            .limit(limit)
        )
        rows = (await s.execute(stmt)).all()
    return [{"id": r.id, "normalized_question": r.normalized_question} for r in rows]


async def insert_review_item(normalized_question: str, ai_suggested_answer: str | None) -> int:
    """新しいナレッジの穴を 1 件レビュー待ちへ積み、その id を返す。"""
    async with db.async_session() as s:
        row = ReviewQueue(
            normalized_question=normalized_question,
            ai_suggested_answer=ai_suggested_answer,
        )
        s.add(row)
        await s.commit()
        return row.id


async def increment_occurrence(review_id: int) -> None:
    """既存の穴に同じ質問が再び来たときに出現回数を 1 進める。

    Python 側で読んでから +1 して書き戻すのではなく、SQL の
    `occurrence_count = occurrence_count + 1` で加算する。pipeline は複数の質問を
    まとめて処理するので、読み書きに分けると同じ穴への 2 件が同じ値を読んで
    片方の加算が消える。
    """
    async with db.async_session() as s:
        await s.execute(
            update(ReviewQueue)
            .where(ReviewQueue.id == review_id)
            .values(occurrence_count=ReviewQueue.occurrence_count + 1)
        )
        await s.commit()


async def set_matched_review(lcq_id: int, review_id: int) -> None:
    """低信頼質問を、まとめ先のナレッジの穴へ紐づける。

    この列が埋まった時点でその質問は fetch_unmatched_low_conf から外れる。
    つまりこの更新が pipeline の「処理済み」の印そのものになるので、
    occurrence_count を進めたあとに必ず呼ぶこと。
    """
    async with db.async_session() as s:
        await s.execute(
            update(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.id == lcq_id)
            .values(matched_review_id=review_id)
        )
        await s.commit()


async def list_review_queue(status: str | None) -> list[ReviewQueue]:
    """レビュー待ちの一覧。occurrence_count 降順(よく来る穴が上)。

    status=None は絞り込みなしの全件。

    第 2 キーに id を足すのは list_eval_runs と同じ理由。穴のほとんどは 1 件のままなので
    同点が普通で、第 1 キーだけでは並びが実行ごとに変わり、レビューキューを開き直すたびに
    順番が入れ替わる(どこまで見たかが分からなくなる)。
    """
    async with db.async_session() as s:
        stmt = select(ReviewQueue).order_by(
            ReviewQueue.occurrence_count.desc(), ReviewQueue.id.desc()
        )
        if status:
            stmt = stmt.where(ReviewQueue.review_status == status)
        return list((await s.execute(stmt)).scalars())


async def get_review_detail(
    review_id: int,
) -> tuple[ReviewQueue, list[LowConfidenceQuestion]] | None:
    """1 件のナレッジの穴と、そこへまとめられた生の質問たち。

    生の質問を一緒に返すのは、正規化後の 1 行だけを見て承認すると、
    正規化が意味を削っていた場合(例えば注文番号や条件が落ちた場合)に
    気づけないため。レビューする人は元の言い回しを読めなければならない。

    穴そのものが無ければ None。生の質問が 0 件なのは異常ではない
    (紐づけ前や、pipeline が穴だけ先に作った直後)。
    """
    async with db.async_session() as s:
        item = await s.get(ReviewQueue, review_id)
        if item is None:
            return None
        raws = list(
            (
                await s.execute(
                    select(LowConfidenceQuestion)
                    .where(LowConfidenceQuestion.matched_review_id == review_id)
                    .order_by(LowConfidenceQuestion.id.asc())
                )
            ).scalars()
        )
        return item, raws


async def update_review_status(
    review_id: int, status: str, approved_answer: str | None = None
) -> bool:
    """レビュー結果を確定する。**pending の行だけ**動かせる。更新できたら True。

    「読んで pending か確かめてから書く」形にはしない。2 人が同時に同じ行を開いて
    片方が承認、片方が却下を押すと、どちらも読み取り時点では pending なので両方が
    成功してしまい、後勝ちで片方の判断が黙って消える。承認は 03 章の取り込み経路を
    通してナレッジベースへ書き戻る操作なので、消えたことに誰も気づけない。

    条件を UPDATE の WHERE に畳み込めば、状態の確認と更新が 1 文の中で起きる。
    行ロックを取れた方だけが rowcount=1 を得て、もう片方は 0 を受け取り、
    自分の操作が通らなかったことを呼び出し元へ返せる。
    """
    async with db.async_session() as s:
        result = await s.execute(
            update(ReviewQueue)
            .where(ReviewQueue.id == review_id, ReviewQueue.review_status == "pending")
            .values(review_status=status, approved_answer=approved_answer)
        )
        await s.commit()
        return result.rowcount == 1


async def insert_eval_run(triggered_by: str, dataset_size: int, metrics: dict) -> int:
    """評価の 1 回分を記録し、その id を返す。"""
    async with db.async_session() as s:
        row = EvalRun(
            triggered_by=triggered_by,
            dataset_size=dataset_size,
            metrics=metrics,
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_eval_runs(limit: int = 10) -> list[EvalRun]:
    """新しい順に評価の実行結果を返す。トレンドの描画元。

    created_at は DATETIME(秒精度)で、同じ秒に 2 件入ることがある。
    第 2 キーに id を足しておかないと同秒の 2 件の並びが実行ごとに変わり、
    トレンドの端が揺れる。
    """
    async with db.async_session() as s:
        stmt = (
            select(EvalRun)
            .order_by(EvalRun.created_at.desc(), EvalRun.id.desc())
            .limit(limit)
        )
        return list((await s.execute(stmt)).scalars())
