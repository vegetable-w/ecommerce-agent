"""データアクセス層。tools (Task 7-8) と orchestrator (Task 11) から呼ばれる。

重要: `app.db.base.async_session` は module attribute 経由でしか参照しないこと
(`import app.db.base as db` して `db.async_session()` の形で使う)。トップレベルで
`from app.db.base import async_session` のように束縛すると、tests/conftest.py の
`monkeypatch.setattr("app.db.base.async_session", ...)` (db_session_factory fixture)
がこのモジュール内の参照には反映されず、テストが本番の support データベースを
静かに触ってしまう。
"""

from datetime import datetime

from sqlalchemy import select

import app.db.base as db
from app.db.models import Conversation, Faq, Message, Ticket

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
