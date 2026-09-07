"""クエリ理解。上流は呼ばず、差し替えたモデルで契約と縮退を確かめる。

プロンプトの品質そのものは tests/data/query_rewrite_samples.jsonl の
ラベル付きサンプルで別途検証する(実際の上流を叩くのでここには置かない)。
"""

from langchain_core.runnables import RunnableLambda

from app.core import query_understanding as qu
from app.core.prompts import QUERY_REWRITE_PROMPT


class _FakeModel:
    """with_structured_output が result を返す(または例外を送出する)最小のモデル。"""

    def __init__(self, result):
        self._result = result
        self.seen = None

    def with_structured_output(self, schema, **kw):
        async def _run(prompt_value):
            self.seen = prompt_value
            if isinstance(self._result, Exception):
                raise self._result
            return self._result

        return RunnableLambda(_run)


def test_prompt_renders_user_query():
    msgs = QUERY_REWRITE_PROMPT.format_messages(query="送料っていくら")
    assert msgs[0].type == "system"
    assert "送料っていくら" in msgs[-1].content


async def test_understand_returns_flat_fields():
    model = _FakeModel(qu._Rewrite(standard="送料はいくらですか", expanded=["配送料", "郵送料"]))
    assert await qu.understand("送料って結局いくら", model=model) == {
        "standard": "送料はいくらですか", "expanded": ["配送料", "郵送料"]}
    # 原文がプロンプトへ渡っていること
    assert "送料って結局いくら" in model.seen.to_messages()[-1].content


async def test_understand_drops_blank_expansions():
    model = _FakeModel(qu._Rewrite(standard="返品できますか", expanded=["  返金 ", "", "\u3000"]))
    out = await qu.understand("返したい", model=model)
    assert out["expanded"] == ["返金"]


async def test_understand_falls_back_when_standard_is_blank():
    model = _FakeModel(qu._Rewrite(standard="  ", expanded=[]))
    assert (await qu.understand("壊れた", model=model))["standard"] == "壊れた"


async def test_understand_degrades_to_raw_query_when_upstream_fails():
    """上流が落ちても例外を投げず、原文をそのまま standard として返す。"""
    model = _FakeModel(RuntimeError("upstream 502"))
    assert await qu.understand("送料って結局いくら", model=model) == {
        "standard": "送料って結局いくら", "expanded": []}


async def test_understand_degrades_when_chat_model_cannot_be_built(monkeypatch):
    """model 未指定で get_chat_model 自体が失敗する場合も縮退する。"""
    def boom():
        raise RuntimeError("チャット上流の設定が不正")

    monkeypatch.setattr(qu, "get_chat_model", boom)
    assert await qu.understand("返品したい") == {"standard": "返品したい", "expanded": []}


async def test_retrieval_query_is_the_raw_query_when_understanding_fails():
    """縮退したときに検索へ渡る文字列が原文と一致すること(03 章と同じ挙動)。"""
    u = await qu.understand("送料って結局いくら", model=_FakeModel(RuntimeError("timeout")))
    search_query = u["standard"] + (" " + " ".join(u["expanded"]) if u["expanded"] else "")
    assert search_query == "送料って結局いくら"
