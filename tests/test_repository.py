"""app.db.repository のテスト。

_test_engine はセッションスコープのイベントループ上で作成される(tests/conftest.py 参照)。
このモジュールのテストも同じループで動かさないと asyncmy のコネクションが別ループに
紐付いたままになり "attached to a different loop" の RuntimeError になる。
"""

import pytest
from sqlalchemy import select

from app.db import repository as repo
from app.db.models import Conversation, ConversationSummary, Faq, Ticket

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
        tool_calls=[{"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c1"}],
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


async def test_search_faq_percent_does_not_false_positive_match(db_session_factory, db_clean):
    """`%` は LIKE のメタ文字なので、無関係な行(パーセント記号を含まない)に
    誤ヒットしてはいけない。"""
    async with db_session_factory() as s:
        s.add(Faq(question="50Xオフセール中です", answer="dummy", category="キャンペーン"))
        await s.commit()
    assert await repo.search_faq("50%オフ") == []


async def test_search_faq_underscore_does_not_false_positive_match(db_session_factory, db_clean):
    """`_` は LIKE の1文字ワイルドカードなので、無関係な行に誤ヒットしてはいけない。"""
    async with db_session_factory() as s:
        s.add(Faq(question="aXbという別の型番", answer="dummy", category="商品"))
        await s.commit()
    assert await repo.search_faq("a_b") == []


async def test_search_faq_literal_percent_still_matches(db_session_factory, db_clean):
    """メタ文字をエスケープしても、キーワードそのものを文字通り含む行は
    引き続きヒットしなければならない。"""
    async with db_session_factory() as s:
        s.add(Faq(question="本当に50%オフのキャンペーン", answer="dummy", category="キャンペーン"))
        await s.commit()
    hits = await repo.search_faq("50%オフ")
    assert len(hits) == 1
    assert hits[0].question == "本当に50%オフのキャンペーン"


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


# ---------------------------------------------------------------------------
# 07 会話の要約(スライディングウィンドウの境界)
# ---------------------------------------------------------------------------

async def test_conversation_summary_roundtrip(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv.summary is None and conv.summary_upto_msg_id is None

    await repo.update_conversation_summary(cid, "ユーザーは注文1001の配送状況を問い合わせた", 5)
    conv = await repo.get_conversation(cid)
    assert conv.summary == "ユーザーは注文1001の配送状況を問い合わせた"
    assert conv.summary_upto_msg_id == 5


async def test_count_messages_after(db_session_factory, db_clean):
    """要約を起動するかどうかの判定材料。境界が空なら「まだ一度も要約していない」。"""
    cid = await repo.create_conversation("u1")
    ids = [await repo.append_message(cid, "user", content=f"q{i}") for i in range(4)]
    assert await repo.count_messages_after(cid, None) == 4
    assert await repo.count_messages_after(cid, ids[1]) == 2
    assert await repo.count_messages_after(cid, ids[3]) == 0


async def test_list_dialog_messages_filters_tool_rows(db_session_factory, db_clean):
    """要約に渡すのは人とサポートの発話だけ。tool の生の JSON を要約させない。"""
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="注文1001は今どこですか")
    await repo.append_message(cid, "tool", content='{"s":1}', tool_call_id="c1")
    await repo.append_message(cid, "assistant", content="配送中です")
    msgs = await repo.list_dialog_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert [m.content for m in msgs] == ["注文1001は今どこですか", "配送中です"]


async def test_summary_of_a_missing_conversation_is_a_no_op(db_session_factory, db_clean):
    """存在しない会話への書き込みで落ちないこと(要約は非同期なので、書き戻す頃には
    会話が消えていることがありうる)。"""
    await repo.update_conversation_summary(999999, "x", 1)


async def test_append_summary_fragment_numbers_segments_per_conversation(
    db_session_factory, db_clean
):
    """断片は追記のみ。seq は会話ごとに 1 から始まり、追記のたびに 1 つ進む。"""
    cid = await repo.create_conversation("u1")
    assert await repo.append_summary_fragment(cid, 1, 10, "注文1001の配送を問い合わせた") == 1
    assert await repo.append_summary_fragment(cid, 11, 20, "電話番号を伝えた") == 2

    other = await repo.create_conversation("u2")
    assert await repo.append_summary_fragment(other, 1, 4, "別の会話") == 1

    async with db_session_factory() as s:
        rows = list((await s.execute(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == cid)
            .order_by(ConversationSummary.seq)
        )).scalars())
    assert [(r.seq, r.from_msg_id, r.upto_msg_id) for r in rows] == [(1, 1, 10), (2, 11, 20)]
    assert rows[0].content == "注文1001の配送を問い合わせた"   # 先の断片は書き換えられない
