"""knowledge route の強制 retrieval node と、その手前後の node。

business route と違い、knowledge は Agent に「検索するかどうか」を選ばせない。
必ず引いてから、引けた根拠が答えるに足るかを 2 段階のゲート(リランクスコアの機械ゲートと
セルフチェックの意味ゲート)で判定する。判定結果の evidence_strong だけが
confidence_gate の入力になるので、ここが誤ると根拠なしの回答か、答えられる質問の
取りこぼしのどちらかが起きる。

上流(埋め込み / リランク / チャット)にも Milvus にも触れない。
"""

import pytest
from langchain_core.messages import HumanMessage

from app.config import settings
from app.graph import nodes


def _hit(i, score, q, a, section="返品ポリシー", ctype="faq"):
    h = {"id": i, "question": q, "answer": a, "section_path": section, "content_type": ctype}
    if score is not None:
        h["rerank_score"] = score
    return h


def _stub(monkeypatch, hits, *, useful=True, reason="十分",
          expanded=(), standard=None):
    """理解 / 検索 / セルフチェックをまとめて差し替え、呼び出し引数を記録して返す。"""
    seen = {}

    async def fake_understand(q, **kw):
        seen["understand"] = q
        return {"standard": standard or q, "expanded": list(expanded)}

    async def fake_search(query, **kw):
        seen["search_query"] = query
        seen["search_kw"] = kw
        # 本物の search_knowledge は hybrid_rerank で min_score を省略されると
        # settings.rerank_min_score で足切りしてから返す。その既定を偽物にも持たせる。
        # 足切り前の hits が届くかどうかが評価対象なので、ここを素通しにすると
        # 「min_score を渡し忘れた」実装がテストを通ってしまう。
        cut = kw.get("min_score")
        cut = settings.rerank_min_score if cut is None else cut
        return [h for h in hits if h.get("rerank_score", 0.0) >= cut]

    async def fake_check(q, texts, **kw):
        seen["check_query"] = q
        seen["check_texts"] = list(texts)
        return {"useful": useful, "reason": reason}

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    return seen


def _state(text="返品ポリシーを教えて"):
    from langchain_core.messages import HumanMessage
    return {"messages": [HumanMessage(text)]}


# --- 根拠が十分な経路 ---------------------------------------------------------


async def test_forced_rag_strong_builds_numbered_evidence(monkeypatch):
    _stub(monkeypatch, [_hit(1, 0.9, "返品ポリシー", "7日以内は返品可能")])
    out = await nodes.forced_rag(_state())

    assert out["evidence_strong"] is True
    assert "[1] 返品ポリシー: 7日以内は返品可能" in out["evidence"]
    assert out["citations"][0]["n"] == 1
    assert out["citations"][0]["id"] == 1
    assert out["citations"][0]["section_path"] == "返品ポリシー"
    assert out["trace"]["forced_rag"] is True
    assert out["trace"]["evidence_top"] == pytest.approx(0.9)


async def test_citation_numbers_follow_head_tail_order(monkeypatch):
    """番号は head/tail 配置「後」の並びに振る(query_faq と同じ規則)。"""
    _stub(monkeypatch, [_hit(10, 0.95, "q1", "a1"), _hit(20, 0.90, "q2", "a2"),
                        _hit(30, 0.85, "q3", "a3"), _hit(40, 0.80, "q4", "a4")])
    out = await nodes.forced_rag(_state())
    assert [c["id"] for c in out["citations"]] == [10, 30, 40, 20]
    assert out["evidence"].startswith("[1] q1: a1")
    assert out["evidence"].endswith("[4] q2: a2")


async def test_missing_fields_do_not_crash_the_graph(monkeypatch):
    """hit に欠けた field があっても KeyError で graph を落とさない。"""
    _stub(monkeypatch, [{"question": "q", "answer": "a", "rerank_score": 0.9}])
    out = await nodes.forced_rag(_state())
    assert out["evidence_strong"] is True
    assert out["citations"][0]["section_path"] is None


# --- 機械ゲート ---------------------------------------------------------------


async def test_weak_when_nothing_is_recalled(monkeypatch):
    _stub(monkeypatch, [])
    out = await nodes.forced_rag(_state("火星探査車はどう買えますか"))
    assert out["evidence_strong"] is False
    assert out["trace"]["forced_rag"] is True
    # 弱いときは根拠を**空で上書きする**。書かないだけでは、checkpointer が
    # 持ち越した前の turn の出典が残る(test_weak_clears_evidence_and_citations)
    assert out["evidence"] == "" and out["citations"] == []


async def test_weak_below_threshold_keeps_the_real_top_score(monkeypatch):
    """閾値未満で倒れたとき、trace に「本当の」top スコアが残ること。

    足切りを search_knowledge 側へ任せると閾値未満の hit がここへ届かず、
    evidence_top が必ず 0.000 になる。fallback_reply がその数字をそのまま
    低信頼プールの理由に載せるので、惜しかったのか全く外れていたのかが
    人から見えなくなる。
    """
    monkeypatch.setattr(settings, "rerank_min_score", 0.3)
    _stub(monkeypatch, [_hit(1, 0.123, "q1", "a1"), _hit(2, 0.010, "q2", "a2")])
    out = await nodes.forced_rag(_state("Pro モデルは自動清掃できますか"))
    assert out["evidence_strong"] is False
    assert out["trace"]["evidence_top"] == pytest.approx(0.123)
    assert out["trace"]["evidence_top"] != 0.0


async def test_hits_below_threshold_are_dropped_from_evidence(monkeypatch):
    """top が閾値を越えても、閾値未満の hit は根拠に混ぜない。"""
    monkeypatch.setattr(settings, "rerank_min_score", 0.3)
    seen = _stub(monkeypatch, [_hit(1, 0.90, "q1", "a1"), _hit(2, 0.05, "q2", "a2")])
    out = await nodes.forced_rag(_state())
    assert [c["id"] for c in out["citations"]] == [1]
    assert "q2" not in out["evidence"]
    assert seen["check_texts"] == ["q1 a1"]


async def test_search_is_called_ungated_with_hybrid_rerank(monkeypatch):
    """既定戦略は dense なので hybrid_rerank を明示する。min_score は足切りなしを渡す。"""
    seen = _stub(monkeypatch, [_hit(1, 0.9, "q", "a")])
    await nodes.forced_rag(_state())
    kw = seen["search_kw"]
    assert kw["strategy"] == "hybrid_rerank"
    assert kw["min_score"] < 0.0  # 足切りなし(リランクスコアは 0〜1)


async def test_rerank_outage_defers_to_the_semantic_gate(monkeypatch):
    """リランク上流が落ちると rerank_score なしで返る。掛ける数字が無いので
    機械ゲートは飛ばし、意味ゲートへ委ねる(拒否に倒すと一時障害が回答拒否に化ける)。"""
    _stub(monkeypatch, [_hit(1, None, "q", "a")])
    out = await nodes.forced_rag(_state())
    assert out["evidence_strong"] is True


# --- 意味ゲート ---------------------------------------------------------------


async def test_weak_when_self_check_rejects(monkeypatch):
    _stub(monkeypatch, [_hit(1, 0.8, "送料", "3,000円以上で送料無料")],
          useful=False, reason="型番の質問だが根拠は送料のみ")
    out = await nodes.forced_rag(_state("Pro モデルは自動清掃できますか"))
    assert out["evidence_strong"] is False
    assert out["trace"]["self_check"] == "型番の質問だが根拠は送料のみ"
    assert out["trace"]["evidence_top"] == pytest.approx(0.8)


# --- 同義語の扱い -------------------------------------------------------------


async def test_expanded_terms_go_to_retrieval_only(monkeypatch):
    """同義語は検索文にだけ足す。セルフチェックへ渡す質問は標準質問のまま
    (同義語は当たりを良くするためのもので、質問の意味を変えてはいけない)。"""
    seen = _stub(monkeypatch, [_hit(1, 0.9, "q", "a")],
                 standard="送料はいくらですか", expanded=("配送料", "郵送料"))
    await nodes.forced_rag(_state("送料っていくらなの?"))
    assert seen["understand"] == "送料っていくらなの?"
    assert seen["search_query"] == "送料はいくらですか 配送料 郵送料"
    assert seen["check_query"] == "送料はいくらですか"


async def test_no_expansion_does_not_append_trailing_space(monkeypatch):
    seen = _stub(monkeypatch, [_hit(1, 0.9, "q", "a")], expanded=())
    await nodes.forced_rag(_state("送料"))
    assert seen["search_query"] == "送料"


# --- 前後の node --------------------------------------------------------------


async def test_coref_is_passthrough():
    out = await nodes.coref(_state())
    assert out == {"trace": {"coref": "passthrough"}}
    # 本章では発話を書き換えない。messages を触ると素通しではなくなる
    assert "messages" not in out


async def test_classify_intent_writes_intent_and_trace(monkeypatch):
    async def fake_classify(query, history=""):
        assert query == "注文はどこですか"
        return {"intent": "配送", "confidence": 0.9}

    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent(_state("注文はどこですか"))
    assert out["intent"] == "配送"
    assert out["trace"]["intent"] == "配送"


async def test_confidence_check_traces_the_decision():
    assert (await nodes.confidence_check({"evidence_strong": True}))["trace"]["confidence"] == "strong"
    assert (await nodes.confidence_check({"evidence_strong": False}))["trace"]["confidence"] == "weak"
    # 書き忘れた上流がいた場合も weak 扱い(confidence_gate と同じ既定)
    assert (await nodes.confidence_check({}))["trace"]["confidence"] == "weak"


async def test_weak_clears_evidence_and_citations(monkeypatch):
    """weak では evidence / citations を**空で上書きする**。書かないでは足りない。

    checkpointer が State を turn 間で持ち越すので、前の turn が strong だった場合の
    citations がそのまま残り、拒否の返答の横に前回の出典が並ぶ。
    """
    _stub(monkeypatch, hits=[])
    out = await nodes.forced_rag({"messages": [HumanMessage("アカウント削除")]})
    assert out["evidence_strong"] is False
    assert out["evidence"] == ""
    assert out["citations"] == []


async def test_classify_intent_also_writes_route(monkeypatch):
    """conditional edge は State を書けないので、route はここで確定させる。"""
    async def _fake(q, history=""):
        return {"intent": "苦情", "confidence": 0.95}

    monkeypatch.setattr(nodes.intent_mod, "classify", _fake)
    out = await nodes.classify_intent({"messages": [HumanMessage("苦情です")]})
    assert out["intent"] == "苦情"
    # 06 で苦情は有人対応の出口(05 の complaint から名前が変わっている)
    assert out["route"] == "escalate"
    assert out["trace"]["route"] == "escalate"
