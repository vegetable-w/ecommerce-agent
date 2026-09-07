"""06 章の intent 分類(8 分類 + confidence)と、その周りの helper。

分類は 5 出口すべての入口なので、ここが崩れると会話全体の経路が変わる。
プロンプトの良し悪しは scripts/eval_intent.py の実測で見る決まりなので、
このモジュールでは**契約と縮退だけ**を固定する:

- classify_intent が State へ書く key(intent / intent_confidence / route)
- trace の key が confidence_check とぶつからないこと
- 上流が落ちたとき、値域外の confidence が返ったときの倒れ方

上流(チャットモデル)は一切呼ばない。
"""

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.core import intent as intent_mod
from app.core.llm import get_chat_model
from app.graph import nodes


class _FakeModel:
    """with_structured_output が result を返す(または例外を送出する)最小のモデル。

    tests/test_query_understanding.py と同じ形。渡された prompt を seen に残す。
    """

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


def _state(text="注文はどこですか", **extra):
    return {"messages": [HumanMessage(text)], **extra}


def _stub_classify(monkeypatch, result, seen=None):
    """classify_intent が呼ぶ分類器を差し替える。引数は seen に残す。"""

    async def _fake(query, history=""):
        if seen is not None:
            seen["query"] = query
            seen["history"] = history
        return result

    monkeypatch.setattr(nodes.intent_mod, "classify", _fake)


# --- classify_intent node -----------------------------------------------------


async def test_classify_intent_writes_intent_confidence_and_route(monkeypatch):
    _stub_classify(monkeypatch, {"intent": "配送", "confidence": 0.92})
    out = await nodes.classify_intent(_state("注文1001の荷物は今どこ?"))

    assert out["intent"] == "配送"
    assert out["intent_confidence"] == pytest.approx(0.92)
    assert out["route"] == "business"
    assert out["trace"]["intent"] == "配送"
    assert out["trace"]["intent_confidence"] == pytest.approx(0.92)
    assert out["trace"]["route"] == "business"


async def test_classify_intent_trace_key_does_not_collide_with_confidence_check(monkeypatch):
    """trace の key は intent_confidence。**confidence は使わない**。

    confidence は confidence_check が strong / weak を入れる key で、trace の reducer は
    同じ key を後勝ちで上書きする。ぶつけると knowledge route の trace から
    分類の確信度と evidence gate の判定のどちらかが黙って消える。
    """
    _stub_classify(monkeypatch, {"intent": "商品相談", "confidence": 0.71})
    out = await nodes.classify_intent(_state("この商品はいくら?"))

    assert "confidence" not in out["trace"]
    assert "intent_confidence" in out["trace"]

    # 実際に畳んでも両方残ること(reducer は state.py の merge_dict)
    from app.graph.state import merge_dict
    merged = merge_dict(out["trace"], (await nodes.confidence_check({}))["trace"])
    assert merged["intent_confidence"] == pytest.approx(0.71)
    assert merged["confidence"] == "weak"


async def test_classify_intent_prefers_resolved_query(monkeypatch):
    """coref が書き下した完全な質問があればそちらを分類する(元の発話ではない)。"""
    seen = {}
    _stub_classify(monkeypatch, {"intent": "注文", "confidence": 0.8}, seen)
    await nodes.classify_intent(_state("それいくらだった?", resolved_query="注文1001の金額はいくらだった?"))
    assert seen["query"] == "注文1001の金額はいくらだった?"


async def test_classify_intent_falls_back_to_last_human_message(monkeypatch):
    """resolved_query が無い / 空なら最後の human message を使う。"""
    seen = {}
    _stub_classify(monkeypatch, {"intent": "雑談", "confidence": 0.9}, seen)
    await nodes.classify_intent(_state("こんにちは", resolved_query=""))
    assert seen["query"] == "こんにちは"


async def test_classify_intent_passes_history(monkeypatch):
    seen = {}
    _stub_classify(monkeypatch, {"intent": "配送", "confidence": 0.8}, seen)
    await nodes.classify_intent({"messages": [
        HumanMessage("注文1001について"), AIMessage("承知しました"), HumanMessage("まだ届きません")]})
    assert seen["query"] == "まだ届きません"
    assert "注文1001について" in seen["history"]
    # 今回の発話は履歴に含めない
    assert "まだ届きません" not in seen["history"]


async def test_classify_intent_routes_unknown_intent_to_business(monkeypatch):
    """routing 表に無い値でも node は落ちず business へ倒す(route_by_intent と同じ既定)。"""
    _stub_classify(monkeypatch, {"intent": "未知", "confidence": 0.1})
    out = await nodes.classify_intent(_state("???"))
    assert out["route"] == "business"


async def test_classify_intent_sends_other_to_fallback_script(monkeypatch):
    _stub_classify(monkeypatch, {"intent": "その他", "confidence": 0.2})
    out = await nodes.classify_intent(_state("うーん"))
    assert out["route"] == "fallback_script"


# --- _history_text ------------------------------------------------------------


def test_history_text_excludes_the_current_utterance():
    """今回の発話を含めると、分類器から見て同じ文が「発話」と「履歴」に 2 回出る。"""
    h = nodes._history_text({"messages": [
        HumanMessage("注文1001について"), AIMessage("承知しました"), HumanMessage("まだ届きません")]})
    assert "まだ届きません" not in h
    assert h == "ユーザー:注文1001について\nサポート:承知しました"


def test_history_text_is_empty_for_the_first_turn():
    assert nodes._history_text({"messages": [HumanMessage("こんにちは")]}) == ""
    assert nodes._history_text({"messages": []}) == ""
    assert nodes._history_text({}) == ""


def test_history_text_keeps_only_the_recent_turns():
    msgs = [HumanMessage(f"q{i}") for i in range(10)]
    h = nodes._history_text({"messages": msgs}, max_turns=3)
    # 末尾の q9 は今回の発話なので除外され、その手前 3 件が残る
    assert h == "ユーザー:q6\nユーザー:q7\nユーザー:q8"


def test_history_text_skips_non_text_content():
    """content は str とは限らない(block の list)。整形で落とさず読み飛ばす。"""
    ai = AIMessage(content=[{"type": "text", "text": "x"}])
    h = nodes._history_text({"messages": [HumanMessage("q"), ai, HumanMessage("いま")]})
    assert h == "ユーザー:q"


# --- get_chat_model の model 上書き -------------------------------------------


def test_get_chat_model_overrides_the_model_name():
    m = get_chat_model(model="distinct-test-marker-intent-model")
    assert m.model_name == "distinct-test-marker-intent-model"


def test_get_chat_model_defaults_to_chat_model(monkeypatch):
    monkeypatch.setattr(settings, "chat_model", "distinct-test-marker-chat-model")
    assert get_chat_model().model_name == settings.chat_model
    assert get_chat_model(model=None).model_name == settings.chat_model
    # 空文字も「未設定」として既定へ倒す(settings.intent_model の既定が "" のため)
    assert get_chat_model(model="").model_name == settings.chat_model


# --- classify(app/core/intent.py)---------------------------------------------


async def test_classify_returns_flat_intent_and_confidence():
    model = _FakeModel(intent_mod._Intent(intent="返金返品", confidence=0.87))
    assert await intent_mod.classify("返品したい", model=model) == {
        "intent": "返金返品", "confidence": 0.87}


async def test_classify_puts_query_and_history_into_the_prompt():
    model = _FakeModel(intent_mod._Intent(intent="配送", confidence=0.9))
    await intent_mod.classify("まだ届きません", "ユーザー:注文1001について", model=model)
    text = model.seen.to_messages()[-1].content
    assert "まだ届きません" in text
    assert "注文1001について" in text


async def test_classify_degrades_to_other_when_upstream_fails(caplog):
    """上流が落ちても例外を投げず「その他」へ倒し、warning を 1 行残す。

    握り潰して黙っていると、上流障害が「その他が増えた」という分類精度の劣化にしか
    見えなくなる。倒したことを運用が後から区別できるように必ず 1 行残す。
    """
    model = _FakeModel(RuntimeError("upstream 502"))
    with caplog.at_level(logging.WARNING, logger="app.core.intent"):
        out = await intent_mod.classify("返品したい", model=model)
    assert out == {"intent": "その他", "confidence": 0.0}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


async def test_classify_rejects_values_outside_the_eight_classes():
    """Literal を素通りした値は INTENTS で弾いて「その他」にする。

    routing 表に無い文字列は route_by_intent を素通りして business へ流れる。
    structured output の実装は上流と版に依存するので、こちら側でも必ず確かめる。
    """
    model = _FakeModel(intent_mod._Intent.model_construct(intent="配送料", confidence=0.9))
    assert (await intent_mod.classify("送料は?", model=model))["intent"] == "その他"


@pytest.mark.parametrize("raw,expect", [(1.5, 1.0), (-0.2, 0.0), (0.0, 0.0), (1.0, 1.0)])
async def test_classify_clamps_confidence_into_zero_to_one(raw, expect):
    """上流は 1.5 や -0.2 を返すことがある。そのまま State に入れると閾値の比較が壊れる。"""
    model = _FakeModel(intent_mod._Intent.model_construct(intent="配送", confidence=raw))
    assert (await intent_mod.classify("荷物どこ", model=model))["confidence"] == pytest.approx(expect)


async def test_classify_treats_unreadable_confidence_as_zero():
    model = _FakeModel(intent_mod._Intent.model_construct(intent="配送", confidence=None))
    assert (await intent_mod.classify("荷物どこ", model=model))["confidence"] == 0.0


async def test_classify_uses_the_configured_intent_model(monkeypatch):
    """settings.intent_model が空なら chat_model(get_chat_model の既定)へ倒す。"""
    seen = {}

    def _fake_get_chat_model(**kw):
        seen.update(kw)
        return _FakeModel(intent_mod._Intent(intent="雑談", confidence=0.9))

    monkeypatch.setattr(intent_mod, "get_chat_model", _fake_get_chat_model)
    monkeypatch.setattr(settings, "intent_model", "")
    await intent_mod.classify("こんにちは")
    assert seen["model"] is None
    # 分類は接客の返答ではなく判定。同じ発話は毎回同じ出口へ入ってほしい
    assert seen["temperature"] == 0

    monkeypatch.setattr(settings, "intent_model", "distinct-test-marker-intent-model")
    await intent_mod.classify("こんにちは")
    assert seen["model"] == "distinct-test-marker-intent-model"


def test_eight_classes_and_the_fallback_class():
    assert set(intent_mod.INTENTS) == {
        "配送", "注文", "商品相談", "返金返品", "アフターサービス", "苦情", "雑談", "その他"}
    # 迷ったときの退避先は「その他」。05 章の「雑談」から変わっている
    assert intent_mod.FALLBACK_INTENT == "その他"


def test_intent_settings_have_seed_defaults(monkeypatch):
    """09 章で使う設定の置き場所。本章では runtime の切り替えを実装しない。"""
    from app.config import Settings
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k",
                 "EMBED_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert s.intent_model == "" and s.intent_small_model == ""
    assert s.intent_mode == "accuracy"
    assert s.intent_conf_threshold == pytest.approx(0.6)
