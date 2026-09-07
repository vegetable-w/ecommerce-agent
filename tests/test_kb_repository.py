"""ナレッジ chunk / staging の repository CRUD テスト。

_test_engine はセッションスコープのイベントループ上で作成される(tests/conftest.py 参照)。
このモジュールのテストも同じループで動かさないと asyncmy のコネクションが別ループに
紐付いたままになり "attached to a different loop" の RuntimeError になる。
"""

import pytest

from app.db import repository

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_insert_and_list_pending(db_session_factory):
    cid = await repository.insert_knowledge_chunk(
        "cat", "送料について", "9900円以上で送料無料", content_type="policy"
    )
    pending = await repository.list_pending_chunks()
    assert [c.id for c in pending] == [cid]
    assert pending[0].questions == "送料について"


async def test_mark_vectorized_removes_from_pending(db_session_factory):
    cid = await repository.insert_knowledge_chunk("cat", "q", "a")
    await repository.mark_chunk_vectorized(cid, str(cid))
    assert await repository.list_pending_chunks() == []
    assert await repository.count_chunks_by_status("done") == 1


async def test_set_neighbors(db_session_factory):
    a = await repository.insert_knowledge_chunk("c", "qa", "aa")
    b = await repository.insert_knowledge_chunk("c", "qb", "ab")
    await repository.set_chunk_neighbors(b, prev_id=a, next_id=None)
    pending = {c.id: c for c in await repository.list_pending_chunks()}
    assert pending[b].prev_chunk_id == a


async def test_staging_flow(db_session_factory):
    i1 = await repository.insert_staging("b1", "conv:1", "送料はいくらですか", "9900円以上で無料")
    await repository.insert_staging("b1", "conv:2", "返品はできますか", "7日以内なら可能")
    extracted = await repository.list_staging_by_status("extracted")
    assert len(extracted) == 2
    await repository.set_staging_status([i1], "discarded")
    assert len(await repository.list_staging_by_status("extracted")) == 1
    assert len(await repository.list_staging_by_status("discarded")) == 1


async def test_list_all_questions(db_session_factory):
    await repository.insert_knowledge_chunk("c", "送料はどう計算されますか", "9900円以上で無料")
    assert "送料はどう計算されますか" in await repository.list_all_questions()


async def test_list_chunk_sections_returns_every_section_with_its_body(db_session_factory):
    """評価セットの検証(scripts/validate_eval_04.py)が読む一覧。

    section_path が無い chunk も落とさずに返す(捨てるのは呼び出し側の仕事)。
    """
    await repository.insert_knowledge_chunk(
        "c", "送料はいくらですか", "3,000円以上は無料", section_path="FAQ / 送料はいくらですか")
    await repository.insert_knowledge_chunk("c", "見出しなし", "本文だけ")
    rows = await repository.list_chunk_sections()
    assert ("FAQ / 送料はいくらですか", "3,000円以上は無料") in rows
    assert ("", "本文だけ") in rows
