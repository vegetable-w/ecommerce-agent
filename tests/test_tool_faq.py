"""app.tools.business.query_faq の入力契約とツール説明のテスト。

pipeline そのものの振る舞い(2 段階ゲート / 番号付き evidence / 引用)は
tests/test_query_faq_rag.py が見る。ここは 03 章から変えてはいけない入口の契約を守る。

検索・クエリ理解・セルフチェックはいずれも上流(埋め込み / Milvus / チャット)を呼ぶので
すべて差し替える。DB には触れないため、conftest.py の
pytest.mark.asyncio(loop_scope="session") は不要。
"""

import pytest
from pydantic import ValidationError

from app.tools.business import query_faq


@pytest.fixture(autouse=True)
def _no_upstream(monkeypatch):
    """このモジュールが本物の上流モデルを呼ばないことの保証(カバレッジではなく運用の安全網)。

    query_faq は検索の前後でクエリ理解とセルフチェックを呼ぶ。どちらも失敗を握り潰す
    設計なので、差し替え漏れがあってもテストは緑のまま実 API を叩き続けてしまう。
    """
    def _boom(*a, **kw):
        raise AssertionError("本物の上流モデルが呼ばれた")

    monkeypatch.setattr("app.core.llm.get_chat_model", _boom)


def _stub_search(monkeypatch, hits):
    """search_knowledge を差し替え、渡された検索文と引数を記録して返す。"""
    seen = {}

    async def fake_search(query, **kw):
        seen["query"] = query
        seen["kw"] = kw
        return hits

    async def fake_understand(q, **kw):
        return {"standard": q, "expanded": []}

    async def fake_check(q, texts, **kw):
        return {"useful": True, "reason": "十分"}

    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    return seen


async def test_query_faq_hit(monkeypatch):
    seen = _stub_search(monkeypatch, [
        {"id": 1, "rerank_score": 0.72, "question": "返品ポリシー", "answer": "7日間の返品に対応",
         "section_path": "返品ポリシー", "content_type": "policy", "category": "返品"},
    ])
    r = await query_faq.ainvoke({"keyword": "返品"})
    assert r["sufficient"] is True
    assert r["citations"][0]["question"] == "返品ポリシー"
    # キーワードが検索へそのまま渡ること(ツールが問い合わせ文を握り潰さない)
    assert seen["query"] == "返品"


async def test_query_faq_miss_returns_refusal(monkeypatch):
    _stub_search(monkeypatch, [])
    r = await query_faq.ainvoke({"keyword": "靴"})
    assert r["sufficient"] is False
    assert r["source"] == "retrieval_low_conf"
    assert r["citations"] == []


async def test_query_faq_empty_keyword_raises_validation_error():
    """Empty keyword should raise ValidationError, not return all rows."""
    with pytest.raises(ValidationError):
        await query_faq.ainvoke({"keyword": ""})


@pytest.mark.parametrize("blank", [
    pytest.param("   ", id="ascii-space"),
    pytest.param("　　", id="ideographic-space-u3000"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
])
async def test_query_faq_whitespace_only_raises_validation_error(blank):
    """Whitespace-only keyword should raise ValidationError.

    U+3000(全角スペース)は日本語 IME がそのまま出す「ありがちな空入力」であり、
    bare な .strip() は落とすが .strip(" ") への「明示化」では落ちなくなる。
    通してしまうと実モデル呼び出しを 1 回焼き、空の検索が走る。
    """
    with pytest.raises(ValidationError) as exc_info:
        await query_faq.ainvoke({"keyword": blank})
    assert "空白のみの値は許可されない" in str(exc_info.value)


def test_query_faq_description_covers_product_specs():
    """商品仕様を query_faq の守備範囲として明示し続けること。

    ナレッジベースには商品仕様マニュアルが入っている。この docstring がモデルの
    ツール選択の根拠なので、「ポリシー・ルール・操作方法」だけに戻すと、仕様の質問が
    mock データを返す query_product へ流れ、実測ではモデルが型番の仕様を
    「確認できませんでした」と答えるか、一般常識で作文する状態に戻る。
    """
    desc = query_faq.description
    assert "仕様" in desc, "商品仕様が守備範囲だと書かれていない"
    assert "型番" in desc, "型番での問い合わせが守備範囲だと書かれていない"
    # 在庫・価格は query_product 側の仕事であることも書き残す(重複した説明で迷わせない)
    assert "query_product" in desc


def test_query_faq_category_is_optional():
    """category は任意の絞り込み。必須にすると、モデルが存在しないカテゴリを
    でっち上げて recall 0 → 偽の回答拒否になる。"""
    schema = query_faq.args_schema.model_json_schema()
    assert schema["required"] == ["keyword"]
