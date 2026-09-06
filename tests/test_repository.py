"""app.db.repository のテスト。

_test_engine はセッションスコープのイベントループ上で作成される(tests/conftest.py 参照)。
このモジュールのテストも同じループで動かさないと asyncmy のコネクションが別ループに
紐付いたままになり "attached to a different loop" の RuntimeError になる。
"""

import pytest

from app.db import repository as repo
from app.db.models import Conversation, Faq, Ticket

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_create_and_get_conversation(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv is not None and conv.user_id == "u1"
    assert await repo.get_conversation(999999) is None


async def test_append_and_list_messages_in_order(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="注文 1001 は今どこですか")
    await repo.append_message(
        cid,
        "assistant",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
    )
    await repo.append_message(cid, "tool", content='{"status":"輸送中"}', tool_call_id="c1")
    msgs = await repo.list_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[2].tool_call_id == "c1"


async def test_search_faq_like_hit_and_miss(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="返品ポリシー", answer="7日間の返品に対応", category="アフターサービス"))
        await s.commit()
    assert len(await repo.search_faq("返品")) == 1  # ヒット
    assert await repo.search_faq("靴") == []  # 検索漏れ


async def test_create_ticket_writes_and_flips_conversation_status(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    no = await repo.create_ticket(cid, "返品したい", "after_sales")
    assert no.startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, no)
        assert t.ticket_type == "after_sales" and t.status == "pending"
        conv = await s.get(Conversation, cid)
        assert conv.status == "escalated"


async def test_repository_uses_module_attribute_not_bound_reference(db_session_factory, db_clean):
    """repository が `app.db.base.async_session` を module attribute 経由で参照していることを保証する。

    もし repository.py が `from app.db.base import async_session` のようにトップレベルで
    束縛してしまうと、conftest.py の `monkeypatch.setattr("app.db.base.async_session", ...)`
    (db_session_factory fixture 参照)が repository 側の参照には反映されず、テストが
    本番の support データベースを静かに触ってしまう。これは最も危険な失敗モードなので、
    2つの角度から確認する:
      1. repository モジュールが `async_session` という名前をトップレベルに持たないこと
         (`import app.db.base as db` の形でしか参照していないことの構造的な証拠)。
      2. monkeypatch されたテスト用 factory を経由して書き込んだ行が、実際に
         db_session_factory (= テストDB) 側から見えること(=本番DBに書かれていないことの
         間接証拠。少なくとも monkeypatch が効いていることの直接証拠)。
    """
    assert not hasattr(repo, "async_session")

    cid = await repo.create_conversation("u1")
    async with db_session_factory() as s:
        conv = await s.get(Conversation, cid)
        assert conv is not None and conv.user_id == "u1"
