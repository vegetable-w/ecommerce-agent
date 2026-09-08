"""app.db.repository のテスト。

_test_engine はセッションスコープのイベントループ上で作成される(tests/conftest.py 参照)。
このモジュールのテストも同じループで動かさないと asyncmy のコネクションが別ループに
紐付いたままになり "attached to a different loop" の RuntimeError になる。
"""

import pytest
from sqlalchemy import event, select

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


# ---------------------------------------------------------------------------
# 07 会話の一覧(受け入れ検証の sidebar)
# ---------------------------------------------------------------------------

async def test_list_conversations_newest_first_with_preview(db_session_factory, db_clean):
    """新しい会話が上。preview は**最初の**ユーザーの発話(会話の入口が分かる)。"""
    first = await repo.create_conversation("u1")
    await repo.append_message(first, "user", content="返品したいのですが")
    await repo.append_message(first, "assistant", content="承知しました")
    await repo.append_message(first, "user", content="いつ届きますか")

    second = await repo.create_conversation("u1")
    await repo.append_message(second, "user", content="注文1001はどこですか")

    rows = await repo.list_conversations("u1")
    assert [r["id"] for r in rows] == [second, first]
    assert rows[0]["preview"] == "注文1001はどこですか"
    # 2 件目以降ではなく 1 件目。ここが最後の発話になると、一覧が毎ターン
    # 書き換わって「どの会話だったか」を目で追えなくなる
    assert rows[1]["preview"] == "返品したいのですが"


async def test_list_conversations_ignores_other_users(db_session_factory, db_clean):
    """他人の会話を混ぜない。混ざると別人の問い合わせ内容が preview に出る。"""
    mine = await repo.create_conversation("u1")
    await repo.append_message(mine, "user", content="私の質問")
    other = await repo.create_conversation("u2")
    await repo.append_message(other, "user", content="他人の質問")

    rows = await repo.list_conversations("u1")
    assert [r["id"] for r in rows] == [mine]
    assert all("他人の質問" not in r["preview"] for r in rows)


async def test_list_conversations_preview_skips_non_user_rows(db_session_factory, db_clean):
    """preview はユーザーの発話。assistant や tool の行を拾わない。"""
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "assistant", content="いらっしゃいませ")
    await repo.append_message(cid, "tool", content='{"s":1}', tool_call_id="c1")
    await repo.append_message(cid, "user", content="注文について聞きたい")

    rows = await repo.list_conversations("u1")
    assert rows[0]["preview"] == "注文について聞きたい"


async def test_list_conversations_includes_empty_conversations(db_session_factory, db_clean):
    """発話が 1 件も無い会話も一覧に出す(preview は空)。

    落とすと、作った直後の会話が sidebar から消える。画面はその会話を
    開いたまま操作しているので、いま居る場所が一覧に無いことになる。
    """
    cid = await repo.create_conversation("u1")
    rows = await repo.list_conversations("u1")
    assert [r["id"] for r in rows] == [cid]
    assert rows[0]["preview"] == ""


async def test_list_conversations_has_summary_flag(db_session_factory, db_clean):
    """要約済みかどうかを画面へ渡す。要約本文は一覧に載せない(長い)。"""
    plain = await repo.create_conversation("u1")
    summarized = await repo.create_conversation("u1")
    await repo.update_conversation_summary(summarized, "ユーザーは注文1001を問い合わせた", 4)

    rows = {r["id"]: r for r in await repo.list_conversations("u1")}
    assert rows[summarized]["has_summary"] is True
    assert rows[plain]["has_summary"] is False
    assert "summary" not in rows[summarized]


async def test_list_conversations_returns_status_and_updated_at(db_session_factory, db_clean):
    """状態は DB の英語識別子のまま返す。日本語への変換は API 層の仕事。"""
    cid = await repo.create_conversation("u1")
    await repo.create_ticket(cid, "返品したい", "after_sales")
    row = (await repo.list_conversations("u1"))[0]
    assert row["status"] == "escalated"
    assert row["updated_at"] is not None


async def test_list_conversations_respects_the_limit(db_session_factory, db_clean):
    ids = [await repo.create_conversation("u1") for _ in range(4)]
    rows = await repo.list_conversations("u1", limit=2)
    assert [r["id"] for r in rows] == [ids[3], ids[2]]


async def test_list_conversations_truncates_a_long_preview(db_session_factory, db_clean):
    """一覧の 1 行に収める。全文を送ると sidebar が長文で埋まる。"""
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="あ" * 300)
    preview = (await repo.list_conversations("u1"))[0]["preview"]
    assert len(preview) <= 60 and preview.startswith("あ")


async def test_list_conversations_does_not_query_per_conversation(
    _test_engine, db_session_factory, db_clean
):
    """会話ごとに問い合わせを往復しないこと(50 件で 51 クエリにしない)。

    件数を変えてもクエリ数が増えないことで確かめる。件数に比例する実装なら
    3 件と 12 件で必ず差が出る。
    """

    async def _seed(n: int) -> None:
        for _ in range(n):
            cid = await repo.create_conversation("u1")
            await repo.append_message(cid, "user", content=f"質問{cid}")

    async def _count_queries(n: int) -> int:
        seen = 0

        def _on_execute(*a, **k):
            nonlocal seen
            seen += 1

        event.listen(_test_engine.sync_engine, "before_cursor_execute", _on_execute)
        try:
            rows = await repo.list_conversations("u1")
        finally:
            event.remove(_test_engine.sync_engine, "before_cursor_execute", _on_execute)
        assert len(rows) == n
        return seen

    await _seed(3)
    few = await _count_queries(3)
    await _seed(9)
    many = await _count_queries(12)
    assert few == many
