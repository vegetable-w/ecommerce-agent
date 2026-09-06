"""mining の入力整形部分だけを検証する。

抽出そのものは純 Prompt なので単体テストを書かない(品質は Task 16 のラベル付きサンプル eval)。
ここで固定するのは、モデルへ渡す前の deterministic な処理だけ:
どの発話を prompt に載せるか、と source_ref に何を記録するか。
"""

import pytest

from app.db import repository
from app.kb import mining

# 同期テストも同居するのでモジュール全体には付けない。モジュール級の pytestmark は
# 同期関数まで asyncio マーク済みと見なして PytestWarning を出す(02 章で踏んだ)。
session_loop = pytest.mark.asyncio(loop_scope="session")


@session_loop
async def test_only_user_and_assistant_rows_reach_the_prompt(db_session_factory):
    """tool 行の生 JSON を抽出プロンプトに混ぜない。"""
    conv = await repository.create_conversation("u1")
    await repository.append_message(conv, "user", content="送料はいくらですか")
    await repository.append_message(conv, "assistant", content=None,
                                    tool_calls=[{"name": "query_faq", "args": {}, "id": "c1"}])
    await repository.append_message(conv, "tool", content='{"hits": [{"question": "x"}]}')
    await repository.append_message(conv, "assistant", content="3,000円以上で送料無料です")

    texts = await mining._load_conversation_texts()
    assert len(texts) == 1
    _, body = texts[0]
    assert "送料はいくらですか" in body
    assert "3,000円以上" in body
    assert "hits" not in body, "tool の実行結果 JSON が prompt に混ざっている"
    assert "None" not in body


@session_loop
async def test_conversation_with_no_usable_rows_is_dropped(db_session_factory):
    conv = await repository.create_conversation("u2")
    await repository.append_message(conv, "assistant", content=None,
                                    tool_calls=[{"name": "query_faq", "args": {}, "id": "c1"}])
    await repository.append_message(conv, "tool", content='{"hits": []}')
    assert await mining._load_conversation_texts() == []


def test_batch_source_ref_lists_every_conversation():
    """先頭 1 件だけ記録すると、既定 batch_size=20 では最大 19 件が誤帰属になる。"""
    assert mining._batch_source_ref(["conv:1", "conv:3"]) == "conv:1,conv:3"


def test_batch_source_ref_truncates_within_column_width():
    refs = [f"conv:{i}" for i in range(1, 101)]
    got = mining._batch_source_ref(refs)
    assert len(got) <= mining._SOURCE_REF_MAX
    assert got.startswith("conv:1,conv:2,")
    assert got.endswith("...(全100件)")
