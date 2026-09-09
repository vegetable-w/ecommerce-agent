"""09 確信度ゲートを graph へ繋いだところ。

05 章から入っていた「リランク Top1 が閾値を越えたか」という単一スコアのゲートを、
app/core/confidence.py の総合点へ置き換える。合わせて、断ったターンを
**なぜ断ったか(fallback_source)**と**そのとき何を引いていたか(retrieved_snapshot)**
の 2 つ付きで低信頼プールへ残す。写しは査読画面で「ナレッジに本当に無いのか、
有るのに引けていないのか」を見分ける材料になる。

このファイルは検索 / セルフチェック / プール投入をすべて差し替える。
**ネットワークにも DB にも触らない**(support は本番相当の DB であり、
テストから書き込まない)。
"""

import pytest
from langchain_core.messages import HumanMessage

from app.config import settings
from app.graph import nodes, runtime


def _hit(i, score, q, a, section="返品ポリシー"):
    h = {"id": i, "question": q, "answer": a, "section_path": section,
         "content_type": "faq"}
    if score is not None:
        h["rerank_score"] = score
    return h


def _stub(monkeypatch, hits, *, useful=True, reason="十分"):
    """理解 / 検索 / セルフチェックを差し替える。

    偽の search_knowledge が min_score で足切りするのは本物と同じ振る舞い。
    ここを素通しにすると、min_score を渡し忘れた実装がテストを通ってしまう。
    """
    async def fake_understand(q, **kw):
        return {"standard": q, "expanded": []}

    async def fake_search(query, **kw):
        cut = kw.get("min_score")
        cut = settings.rerank_min_score if cut is None else cut
        return [h for h in hits if h.get("rerank_score", 0.0) >= cut]

    async def fake_check(q, texts, **kw):
        return {"useful": useful, "reason": reason}

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)


def _state(text="返品ポリシーを教えて"):
    return {"messages": [HumanMessage(text)]}


# --- ゲートの 3 つの倒れ方 -----------------------------------------------------


async def test_low_confidence_blocks_with_source_and_snapshot(monkeypatch):
    """根拠そのものが弱いターン。source は retrieval_low_conf、写しは残る。

    写しを**足切りの手前**から取ることが要点。断ったターンほどスコアは低いので、
    足切り後から取ると、いちばん写しが要る行だけ空になる。

    rerank_min_score を hit より下げておくのは、**断った理由を確信度だけに絞る**ため。
    既定の 0.3 のままだと 04 章の hit 単位の足切りが先に全部落としてしまい、
    確信度ゲートを外しても同じ結果になる(実際、外して素通ししてみると気付けない)。
    """
    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.5)
    monkeypatch.setattr(settings, "rerank_min_score", 0.1)
    _stub(monkeypatch, [_hit(1, 0.15, "送料の目安", "地域で異なります"),
                        _hit(2, 0.14, "配送日数", "3〜5 日です")])
    out = await nodes.forced_rag(_state("Pro モデルは自動清掃できますか"))

    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "retrieval_low_conf"
    assert len(out["retrieved_snapshot"]) == 2
    assert out["evidence_confidence"] < 0.5
    assert out["trace"]["confidence_signals"]["top1_score"] == pytest.approx(0.15)
    # 弱いときは根拠を空で上書きする(checkpointer の持ち越し対策。05 章から同じ)
    assert out["evidence"] == "" and out["citations"] == []


async def test_single_borderline_hit_is_refused_by_the_total_score(monkeypatch):
    """hit 単位の足切りは越えるが、総合点では足りないターン。

    05 章の単一スコアのゲートなら「0.35 ≧ 0.3 だから答える」で通っていた。
    根拠が 1 本きりで、その 1 本も辛うじて下限という状態は、答えるには薄い。
    **確信度ゲートだけがここを止められる**ので、置き換えの意味そのものを見る。
    """
    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.5)
    monkeypatch.setattr(settings, "rerank_min_score", 0.3)
    _stub(monkeypatch, [_hit(1, 0.35, "配送日数の目安", "地域により異なります")])
    out = await nodes.forced_rag(_state("Pro モデルは自動清掃できますか"))

    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "retrieval_low_conf"
    assert out["evidence_confidence"] < 0.5
    # 足切りは越えているので写しは 1 件残る(「惜しかった」ことが査読で分かる)
    assert len(out["retrieved_snapshot"]) == 1
    assert out["trace"]["evidence_top"] == pytest.approx(0.35)


async def test_selfcheck_fail_labels_self_check(monkeypatch):
    """根拠は取れたが答えきれなかったターン。source を分け、写しは残す。

    プールでの直し方が違う。retrieval_low_conf はナレッジに書き足す話、
    self_check は書き方か索き方を直す話で、同じ札にすると査読で混ざる。
    """
    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.3)
    _stub(monkeypatch, [_hit(1, 0.9, "返品の期限", "7日以内は返品可能"),
                        _hit(2, 0.3, "返品の送料", "お客様負担です"),
                        _hit(3, 0.3, "交換の可否", "同一商品のみ")],
          useful=False, reason="型番の質問だが根拠は返品のみ")
    out = await nodes.forced_rag(_state("Pro モデルは自動清掃できますか"))

    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "self_check"
    assert out["retrieved_snapshot"], "意味ゲートで落ちた場合も写しは残す"
    assert out["trace"]["self_check"] == "型番の質問だが根拠は返品のみ"


async def test_strong_evidence_keeps_snapshot_for_feedback(monkeypatch):
    """通ったターンでも写しは残す。後から付く 👎 がこれを拾う。

    👎 は「答えたが外していた」回答に付くもので、そのとき何を引いていたかが
    無いと直しようがない。
    """
    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.3)
    _stub(monkeypatch, [_hit(1, 0.9, "返品の期限", "7日以内は返品可能"),
                        _hit(2, 0.3, "返品の送料", "お客様負担です"),
                        _hit(3, 0.3, "交換の可否", "同一商品のみ")])
    out = await nodes.forced_rag(_state())

    assert out["evidence_strong"] is True
    assert [c["question"] for c in out["retrieved_snapshot"]] == [
        "返品の期限", "返品の送料", "交換の可否"]
    assert out["evidence_confidence"] >= 0.3
    # 通ったターンに前の turn の別が残らないよう、書く側で空に戻す
    assert out["fallback_source"] == ""


async def test_snapshot_is_empty_when_nothing_is_recalled(monkeypatch):
    """1 件も引けなければ写しは空。写しが無いことと弱いことは両立する。"""
    _stub(monkeypatch, [])
    out = await nodes.forced_rag(_state("火星探査車はどう買えますか"))
    assert out["evidence_strong"] is False
    assert out["retrieved_snapshot"] == []
    assert out["evidence_confidence"] == 0.0
    assert out["fallback_source"] == "retrieval_low_conf"


# --- 護柵: リランク上流が落ちたとき ---------------------------------------------


async def test_rerank_outage_still_skips_the_machine_gate(monkeypatch):
    """確信度の 4 信号はどれもリランクスコアから作る。数字が無ければゲートを飛ばす。

    04 章から続く護柵で、09 でも同じ。ここで拒否に倒すと、上流の一時障害が
    そのまま回答拒否 + 低信頼プールへの投入に化ける(閾値をいくら上げても)。
    """
    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.99)
    _stub(monkeypatch, [_hit(1, None, "返品の期限", "7日以内は返品可能")])
    out = await nodes.forced_rag(_state())

    assert out["evidence_strong"] is True
    assert out["trace"]["rerank"] == "unavailable"


async def test_rerank_outage_falls_through_to_the_semantic_gate(monkeypatch):
    """飛ばすのは機械ゲートだけ。意味ゲートは通す(素通しにはしない)。"""
    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.99)
    _stub(monkeypatch, [_hit(1, None, "送料", "3,000円以上で無料")],
          useful=False, reason="質問と根拠が噛み合わない")
    out = await nodes.forced_rag(_state("Pro モデルは自動清掃できますか"))

    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "self_check"


# --- fallback_reply が写しと別をプールへ渡すこと --------------------------------


def _capture(monkeypatch):
    calls = {}

    async def fake_insert(conversation_id, raw_question, source, reason,
                          retrieved_chunks=None):
        calls.update(conversation_id=conversation_id, raw=raw_question,
                     source=source, reason=reason, chunks=retrieved_chunks)
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    return calls


async def test_fallback_uses_the_source_and_snapshot_from_state(monkeypatch):
    calls = _capture(monkeypatch)
    snapshot = [{"question": "返品の期限", "answer": "7日以内", "rerank_score": 0.9,
                 "section_path": "返品ポリシー"}]
    await nodes.fallback_reply({
        "messages": [HumanMessage("Pro モデルは自動清掃できますか")],
        "conversation_id": 3,
        "fallback_source": "self_check",
        "evidence_confidence": 0.71,
        "retrieved_snapshot": snapshot,
        "trace": {"evidence_top": 0.9,
                  "confidence_signals": {"top1_score": 0.9, "valid_count": 3,
                                         "margin": 0.6, "key_clause_hit": True}},
    })

    assert calls["source"] == "self_check"
    assert calls["chunks"] == snapshot


async def test_fallback_reason_carries_the_confidence_signals(monkeypatch):
    """理由から確信度の信号が読めること。査読画面で開く前の一次情報になる。"""
    calls = _capture(monkeypatch)
    await nodes.fallback_reply({
        "messages": [HumanMessage("質問")],
        "conversation_id": 3,
        "evidence_confidence": 0.176,
        "retrieved_snapshot": [],
        "trace": {"evidence_top": 0.15,
                  "confidence_signals": {"top1_score": 0.15, "valid_count": 0,
                                         "margin": 0.01, "key_clause_hit": False}},
    })

    reason = calls["reason"]
    assert "0.150" in reason        # 04 章から載せている top スコア
    assert "0.176" in reason        # 09 の総合点
    assert "valid=0" in reason
    assert "margin=0.010" in reason


async def test_fallback_without_the_gate_keys_does_not_crash(monkeypatch):
    """ゲートを通らずここへ倒れてきた経路でも落ちないこと。

    fallback_source も retrieved_snapshot も trace も無い State が来る。
    整形で落とすと、穏当な回答拒否がそのまま 500 に化ける。
    """
    calls = _capture(monkeypatch)
    out = await nodes.fallback_reply({"messages": [HumanMessage("質問")]})

    assert out["answer"] == nodes.FALLBACK_REPLY
    assert calls["source"] == "retrieval_low_conf"   # 既定は 04 章から同じ
    assert calls["chunks"] is None                   # 空 list ではなく NULL


async def test_empty_snapshot_is_written_as_null(monkeypatch):
    """写しが空のときは NULL。DDL はこの列の NULL を「検索を通っていない」に使う。

    空 list を入れると JSON の [] になり、`retrieved_chunks IS NULL` で
    数える側から漏れる。
    """
    calls = _capture(monkeypatch)
    await nodes.fallback_reply({"messages": [HumanMessage("質問")],
                                "retrieved_snapshot": []})
    assert calls["chunks"] is None


# --- turn の入口でのリセット ---------------------------------------------------


def test_graph_input_resets_the_confidence_channels():
    """checkpointer は State を持ち越す。retrieved_snapshot が残ると、👎 が
    別の質問の検索結果を拾う(07 章で同種の実害が出ている)。"""
    inp = runtime._graph_input("u1", "こんにちは", 7, 12, "", 0)
    assert inp["evidence_confidence"] == 0.0
    assert inp["fallback_source"] == ""
    assert inp["retrieved_snapshot"] == []
