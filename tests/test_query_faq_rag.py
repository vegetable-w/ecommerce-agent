"""query_faq を RAG pipeline へ上げた後の振る舞い(04 章 Task 10)。

クエリ理解 → ハイブリッド検索 + リランク → 2 段階の evidence gate → 番号付き evidence。
上流(埋め込み / リランク / チャット)は一切呼ばない。DB にも触れないので
conftest の loop_scope マーカーは不要。
"""

import json

import pytest

from app.config import settings
from app.core.prompts import RAG_INSUFFICIENT_NOTICE
from app.tools import engine, registry
from app.tools.builtin.faq import SNAPSHOT_KEY, query_faq


def _hit(i, score, q, a, section="送料ポリシー", ctype="faq", cat="送料"):
    h = {"id": i, "question": q, "answer": a, "section_path": section,
         "content_type": ctype, "category": cat}
    if score is not None:
        h["rerank_score"] = score
    return h


def _wire(out: dict) -> str:
    """モデルが実際に読む本文。**engine の整形(format_result)を通す。**

    dict のキーを見るだけでは、内部用の key がワイヤ上でモデルまで届いていても
    気づけない(09 章で足した検索の写しがそれ)。
    """
    spec = registry.get_builtin_spec("query_faq")
    return engine._format_content(spec, out)


def _stub(monkeypatch, hits, *, useful=True, reason="十分",
          expanded=("配送料",), standard=None):
    """理解 / 検索 / セルフチェックをまとめて差し替え、呼び出し引数を記録して返す。"""
    seen = {}

    async def fake_understand(q, **kw):
        seen["understand"] = q
        return {"standard": standard or q, "expanded": list(expanded)}

    async def fake_search(query, **kw):
        seen["search_query"] = query
        seen["search_kw"] = kw
        return list(hits)

    async def fake_check(q, texts, **kw):
        seen["check_query"] = q
        seen["check_texts"] = list(texts)
        return {"useful": useful, "reason": reason}

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    return seen


# --- 十分な根拠が得られた経路 -------------------------------------------------


async def test_sufficient_returns_numbered_evidence_and_aligned_citations(monkeypatch):
    _stub(monkeypatch, [
        _hit(1, 0.9, "送料はいくらですか", "3,000円以上で送料無料"),
        _hit(2, 0.6, "返品はできますか", "7日間の返品に対応", section="返品ポリシー"),
    ])
    out = await query_faq.ainvoke({"keyword": "送料はいくら?"})

    assert out["sufficient"] is True
    assert [c["n"] for c in out["citations"]] == [1, 2]
    assert out["citations"][0]["id"] == 1
    assert out["citations"][0]["section_path"] == "送料ポリシー"
    assert out["citations"][0]["content_type"] == "faq"
    # evidence の [n] と citations の n が同じ chunk を指すこと
    for c in out["citations"]:
        assert f"[{c['n']}] {c['question']}: {c['answer']}" in out["evidence"]
    # 拒否側のフィールドは混ぜない
    # source は拒否側だけの欄。足りている側には出ない
    assert "source" not in out
    # notice は足りている側にもある。ただし中身は「拒否せよ」ではなく「引用せよ」。
    # 収束生成が読む AGENT_SYSTEM(02 章、凍結)に引用ルールが無いため、ここでしか指示できない
    assert "[1] [2] の形式" in out["notice"]
    assert "回答できる根拠が見つかりませんでした" not in out["notice"]


async def test_citation_numbers_follow_head_tail_order(monkeypatch):
    """番号は head/tail 配置「後」の並びに振る。2 位が末尾に回るのが可視的な証拠。"""
    _stub(monkeypatch, [
        _hit(10, 0.95, "q1", "a1"),
        _hit(20, 0.90, "q2", "a2"),
        _hit(30, 0.85, "q3", "a3"),
        _hit(40, 0.80, "q4", "a4"),
    ])
    out = await query_faq.ainvoke({"keyword": "送料"})
    assert [c["id"] for c in out["citations"]] == [10, 30, 40, 20]
    assert out["evidence"].startswith("[1] q1: a1")
    assert out["evidence"].endswith("[4] q2: a2")


async def test_search_is_called_with_hybrid_rerank_and_ungated_threshold(monkeypatch):
    """既定戦略は dense なので、hybrid_rerank を明示的に渡し続けること。

    さらに min_score は「足切りなし」を渡す。足切りを search_knowledge に任せると
    閾値未満の hit がここへ届かず、拒否理由に載せる本当の top スコアが失われる。
    """
    seen = _stub(monkeypatch, [_hit(1, 0.9, "q", "a")])
    await query_faq.ainvoke({"keyword": "送料", "category": "送料"})
    kw = seen["search_kw"]
    assert kw["strategy"] == "hybrid_rerank"
    assert kw["category"] == "送料"
    assert kw["min_score"] < 0.0  # 足切りなし(rerank スコアは 0〜1)


async def test_expanded_terms_go_to_retrieval_only(monkeypatch):
    """同義語は検索文にだけ足す。セルフチェックへ渡す質問は標準質問のまま。"""
    seen = _stub(monkeypatch, [_hit(1, 0.9, "q", "a")],
                 standard="送料はいくらですか", expanded=("配送料", "郵送料"))
    await query_faq.ainvoke({"keyword": "送料っていくらなの?"})
    assert seen["understand"] == "送料っていくらなの?"
    assert seen["search_query"] == "送料はいくらですか 配送料 郵送料"
    assert seen["check_query"] == "送料はいくらですか"


async def test_no_expansion_does_not_append_trailing_space(monkeypatch):
    seen = _stub(monkeypatch, [_hit(1, 0.9, "q", "a")], expanded=())
    await query_faq.ainvoke({"keyword": "送料"})
    assert seen["search_query"] == "送料"


# --- 機械ゲート(retrieval_low_conf) ------------------------------------------


async def test_no_recall_is_retrieval_low_conf(monkeypatch):
    _stub(monkeypatch, [])
    out = await query_faq.ainvoke({"keyword": "火星探査車はどう買えますか"})
    assert out["sufficient"] is False
    assert out["source"] == "retrieval_low_conf"
    assert out["citations"] == []
    assert out["reason"]


async def test_below_threshold_reason_carries_the_real_top_score(monkeypatch):
    """拒否理由には実際の top リランクスコアが載ること。

    足切りを search_knowledge 側に任せたままだと、ここへ届く hits が既に空になり
    理由が必ず top=0.000 になってしまう。低信頼プールを人が見るとき、その 1 つの
    数字だけが「惜しかったのか、全く外れていたのか」を区別する材料になる。
    """
    monkeypatch.setattr(settings, "rerank_min_score", 0.3)
    _stub(monkeypatch, [
        _hit(1, 0.123, "q1", "a1"),
        _hit(2, 0.010, "q2", "a2"),
    ])
    out = await query_faq.ainvoke({"keyword": "Pro モデルは自動清掃できますか"})
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"
    assert "0.123" in out["reason"]
    assert "0.000" not in out["reason"]


async def test_hits_below_threshold_are_dropped_from_evidence(monkeypatch):
    """top が閾値を越えても、閾値未満の hit は根拠に混ぜない。"""
    monkeypatch.setattr(settings, "rerank_min_score", 0.3)
    seen = _stub(monkeypatch, [
        _hit(1, 0.90, "q1", "a1"),
        _hit(2, 0.05, "q2", "a2"),
    ])
    out = await query_faq.ainvoke({"keyword": "送料"})
    assert [c["id"] for c in out["citations"]] == [1]
    assert "q2" not in out["evidence"]
    # セルフチェックにも足切り後の根拠だけを見せる
    assert seen["check_texts"] == ["q1 a1"]


# --- 意味ゲート(self_check) --------------------------------------------------


async def test_self_check_failure_is_self_check_source(monkeypatch):
    _stub(monkeypatch, [_hit(1, 0.8, "送料", "3,000円以上で送料無料")],
          useful=False, reason="型番の質問だが根拠は送料のみ")
    out = await query_faq.ainvoke({"keyword": "Pro モデルは自動清掃できますか"})
    assert out["sufficient"] is False
    assert out["source"] == "self_check"
    assert out["reason"] == "型番の質問だが根拠は送料のみ"
    assert out["citations"] == []


# --- 回答拒否の指示が「モデルが読む本文」として届くこと ------------------------


@pytest.mark.parametrize("useful,hits,source", [
    (True, [], "retrieval_low_conf"),
    (False, [{"id": 1, "rerank_score": 0.9, "question": "q", "answer": "a",
              "section_path": "s", "content_type": "faq", "category": "c"}], "self_check"),
])
async def test_insufficient_result_carries_the_refusal_instruction_as_text(
    monkeypatch, useful, hits, source
):
    """02 章の実測: ToolMessage(status="error") の status は上流へ届かず、モデルには
    本文しか渡らない。したがって {"sufficient": false} という「フラグ」は指示にならず、
    回答拒否の文面そのものがツール結果の JSON に載っていなければならない。

    ここでは infra.execute_tool_call と同じ直列化を通し、モデルが実際に読む文字列を
    組み立てて確認する(dict のキーを見るだけでは、ワイヤ上で落ちても気づけない)。
    """
    _stub(monkeypatch, hits, useful=useful, reason="根拠不足")
    out = await query_faq.ainvoke({"keyword": "火星探査車"})
    assert out["sufficient"] is False and out["source"] == source

    wire = _wire(out)
    assert RAG_INSUFFICIENT_NOTICE in wire, "回答拒否の指示が本文として届いていない"
    assert "推測で回答を作らず" in wire
    assert "時間をおいての再試行は案内しない" in wire


# --- 断ったときの検索の写し(09 章。低信頼プールと査読画面が読む)---------------


async def test_a_score_refusal_carries_the_snapshot_taken_before_the_cutoff(monkeypatch):
    """足切りで断ったときの写しは**足切り前**の Top3。

    足切り後は空なので、そちらから取ると「何を引いていたのか」が残らない。査読で
    いちばん知りたいのは、ナレッジに無いのか、有るのに引けていないのかの区別。
    """
    monkeypatch.setattr(settings, "rerank_min_score", 0.5)
    _stub(monkeypatch, [
        _hit(1, 0.21, "置き配はできますか", "対応していません"),
        _hit(2, 0.11, "宅配ボックス", "利用できます", section="配送/受け取り"),
        _hit(3, 0.10, "q3", "a3"),
        _hit(4, 0.09, "q4", "a4"),
    ])
    out = await query_faq.ainvoke({"keyword": "置き配"})

    snap = out[SNAPSHOT_KEY]
    assert [c["question"] for c in snap] == ["置き配はできますか", "宅配ボックス", "q3"]
    assert snap[0]["rerank_score"] == 0.21
    assert snap[1]["section_path"] == "配送/受け取り"
    # 形は app/core/confidence.py の snapshot_from_hits と同じ(査読画面がこの形を読む)
    assert set(snap[0]) == {"question", "answer", "rerank_score", "section_path"}


async def test_a_self_check_refusal_carries_the_snapshot(monkeypatch):
    """意味ゲートで断った場合も写しを残す。こちらは足切りを通った根拠そのもの。"""
    _stub(monkeypatch, [_hit(1, 0.8, "送料", "3,000円以上で送料無料")],
          useful=False, reason="型番の質問だが根拠は送料のみ")
    out = await query_faq.ainvoke({"keyword": "Pro モデルは自動清掃できますか"})

    assert out["source"] == "self_check"
    assert [c["question"] for c in out[SNAPSHOT_KEY]] == ["送料"]


async def test_zero_hits_is_an_empty_snapshot_not_none(monkeypatch):
    """**検索を通ったのに写しが None** になると、査読画面が嘘の説明を出す。

    None は「検索を通っていない」の意味で、DDL もそう使っている(app/db/models.py)。
    1 件も返らなかったことは [] で表す(static/review.html は別の文言で出し分ける)。
    """
    _stub(monkeypatch, [])
    out = await query_faq.ainvoke({"keyword": "火星探査車"})

    assert out[SNAPSHOT_KEY] == []
    assert out[SNAPSHOT_KEY] is not None


async def test_the_snapshot_never_reaches_the_model(monkeypatch):
    """写しはモデルへ渡す本文に載らないこと。

    断ったのと同じ turn で、その根拠らしきものを読ませることになる(04 章の回答拒否の
    契約が濁る)。本物の ToolSpec の format_result を通して確かめる。
    """
    _stub(monkeypatch, [_hit(1, 0.8, "送料はいくらですか", "3,000円以上で送料無料")],
          useful=False, reason="根拠不足")
    out = await query_faq.ainvoke({"keyword": "火星探査車"})
    assert out[SNAPSHOT_KEY], "写し自体は取れていること(前提)"

    wire = _wire(out)
    assert SNAPSHOT_KEY not in wire
    assert "3,000円以上で送料無料" not in wire, "断った根拠の本文がモデルへ渡っている"
    # 04 章からの契約は落とさない
    assert json.loads(wire)["sufficient"] is False
    assert json.loads(wire)["source"] == "self_check"


async def test_a_sufficient_result_is_unchanged_by_the_stripping(monkeypatch):
    """足りている側の本文は 04 章のまま(写しの key はそもそも載らない)。"""
    _stub(monkeypatch, [_hit(1, 0.9, "送料はいくらですか", "3,000円以上で送料無料")])
    out = await query_faq.ainvoke({"keyword": "送料"})

    assert SNAPSHOT_KEY not in out
    assert json.loads(_wire(out))["evidence"] == out["evidence"]


# --- リランク上流が落ちた場合の縮退 -------------------------------------------


async def test_degrades_to_hybrid_order_when_rerank_is_unavailable(monkeypatch):
    """rerank が [] を返すと search_knowledge は rerank_score なしの並びを返す。

    そのとき機械ゲートに掛ける数字が存在しない。ここで拒否に倒すと、リランク上流の
    一時障害がそのまま「答えられません」+ 低信頼プール投入に化けてしまうので、
    意味ゲートへ委ねて回答を続ける。
    """
    _stub(monkeypatch, [
        _hit(1, None, "送料はいくらですか", "3,000円以上で送料無料"),
        _hit(2, None, "返品はできますか", "7日間の返品に対応"),
    ])
    out = await query_faq.ainvoke({"keyword": "送料"})
    assert out["sufficient"] is True
    assert [c["id"] for c in out["citations"]] == [1, 2]


async def test_rerank_unavailable_still_honours_the_self_check(monkeypatch):
    _stub(monkeypatch, [_hit(1, None, "送料", "3,000円以上で送料無料")],
          useful=False, reason="質問と無関係")
    out = await query_faq.ainvoke({"keyword": "火星探査車"})
    assert out["sufficient"] is False and out["source"] == "self_check"


# --- 入力契約(03 章から不変) --------------------------------------------------


async def test_category_defaults_to_none(monkeypatch):
    seen = _stub(monkeypatch, [_hit(1, 0.9, "q", "a")])
    await query_faq.ainvoke({"keyword": "送料"})
    assert seen["search_kw"]["category"] is None
