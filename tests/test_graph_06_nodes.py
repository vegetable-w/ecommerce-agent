"""06 章の分類前 2 段(指示対象の解決 / intent 分類)、検索クエリの展開、注文の特定。

分類は 5 出口すべての入口なので、ここが崩れると会話全体の経路が変わる。
プロンプトの良し悪しは scripts/eval_intent.py の実測で見る決まりなので、
このモジュールでは**契約と縮退だけ**を固定する:

- classify_intent が State へ書く key(intent / intent_confidence / route)
- trace の key が confidence_check とぶつからないこと
- 上流が落ちたとき、値域外の confidence が返ったときの倒れ方
- coref node が書く resolved_query と trace の coref(rewrite / passthrough)
- resolve / expand_queries の縮退(履歴が無ければ呼ばない、失敗と空は原文へ倒す)
- fetch_order が注文をどう特定するか(発話から拾う / State の値 / 画面で選ばせる)と、
  番号が無いときに **model に推測させず interrupt する**こと

interrupt を通るテストは、fetch_order だけの最小 graph(_fetch_order_graph)を組んで
ainvoke する。interrupt() は compiled graph の中でしか動かず、node を直に呼ぶと
例外になるため。checkpointer は InMemorySaver で、本番の checkpoint ファイルには触らない。

書き下しと展開の**中身の良し悪し**は scripts/eval_coref.py / scripts/eval_expand.py の
実測で見る。ここで固定するのは契約と縮退だけ。

上流(チャットモデル)は一切呼ばない。
"""

import logging

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.config import settings
from app.core import coref as coref_mod
from app.core import intent as intent_mod
from app.core import query_understanding as qu
from app.core.llm import get_chat_model
from app.core.prompts import AGENT_SYSTEM, COREF_REWRITE_PROMPT, EXPAND_QUERIES_PROMPT
from app.graph import nodes
from app.graph.state import ConversationState
from app.tools import business
from app.tools.engine import ToolRun


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


async def test_classify_rejects_values_outside_the_known_classes():
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


def test_the_classes_and_the_fallback_class():
    """08 で「人工対応」が加わって 9 分類。routing 表との一致は
    tests/test_graph_routing.py が見る。"""
    assert set(intent_mod.INTENTS) == {
        "配送", "注文", "商品相談", "返金返品", "アフターサービス", "苦情",
        "人工対応", "雑談", "その他"}
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


# --- 指示対象の解決(app/core/coref.py + coref node)-----------------------------


class _UpstreamCalled(BaseException):
    """呼ばれたこと自体を失敗にするための番兵。

    BaseException にするのは、resolve / expand_queries が `except Exception` で
    上流の失敗を握って原文へ倒すため。Exception を継承すると「呼ばれたのに握られて
    原文が返る」ので、呼んでいないことを確かめたいテストが素通りで green になる。
    """


def _fake_chat(result, seen=None):
    """prompt | model の model 位置に差せる最小のモデル。

    構造化出力を使わない coref 用。result が Exception なら送出し、そうでなければ
    その文字列を本文とする AIMessage を返す。
    """

    async def _run(prompt_value):
        if seen is not None:
            seen["prompt"] = prompt_value
        if isinstance(result, BaseException):
            raise result
        return AIMessage(result)

    return RunnableLambda(_run)


def _stub_resolve(monkeypatch, result, seen=None):
    async def _fake(query, history=""):
        if seen is not None:
            seen["query"] = query
            seen["history"] = history
        return result

    monkeypatch.setattr(nodes.coref_mod, "resolve", _fake)


def test_coref_prompt_renders_history_and_query():
    msgs = COREF_REWRITE_PROMPT.format_messages(
        history="ユーザー:注文1001のスマート家電について", query="それは返品できる?")
    assert msgs[0].type == "system"
    assert "注文1001のスマート家電について" in msgs[-1].content
    assert "それは返品できる?" in msgs[-1].content


async def test_coref_node_writes_resolved_query_and_marks_rewrite(monkeypatch):
    _stub_resolve(monkeypatch, "注文1001のBluetoothイヤホンは返品できますか")
    out = await nodes.coref({"messages": [
        HumanMessage("注文1001のBluetoothイヤホンはいつ届きますか"),
        AIMessage("明日到着予定です"),
        HumanMessage("これは返品できますか")]})

    assert out["resolved_query"] == "注文1001のBluetoothイヤホンは返品できますか"
    assert out["trace"]["coref"] == "rewrite"


async def test_coref_node_marks_passthrough_when_unchanged(monkeypatch):
    """書き下しが原文と同じなら passthrough。trace の key は 05 章からの coref を保つ。"""
    _stub_resolve(monkeypatch, "保証期間はどのくらいですか")
    out = await nodes.coref(_state("保証期間はどのくらいですか"))

    assert out["resolved_query"] == "保証期間はどのくらいですか"
    assert out["trace"]["coref"] == "passthrough"


async def test_coref_node_passes_history_without_the_current_utterance(monkeypatch):
    seen = {}
    _stub_resolve(monkeypatch, "x", seen)
    await nodes.coref({"messages": [
        HumanMessage("注文1001について"), AIMessage("発送済みです"), HumanMessage("それは今どこ?")]})

    assert seen["query"] == "それは今どこ?"
    assert "注文1001について" in seen["history"]
    assert "それは今どこ?" not in seen["history"]


async def test_resolve_returns_the_rewritten_sentence():
    seen = {}
    model = _fake_chat("注文1001のスマート家電は返品できますか", seen)
    got = await coref_mod.resolve("それは返品できる?", "ユーザー:注文1001のスマート家電について",
                                  model=model)
    assert got == "注文1001のスマート家電は返品できますか"
    # 原文と履歴の両方がプロンプトへ渡っていること
    text = seen["prompt"].to_messages()[-1].content
    assert "それは返品できる?" in text
    assert "注文1001のスマート家電について" in text


async def test_resolve_skips_the_upstream_when_there_is_no_history():
    """履歴が無ければ補える文脈も無い。呼ぶだけ無駄で、課金だけが増える。"""
    model = _fake_chat(_UpstreamCalled("履歴が空なのに上流を呼んだ"))
    assert await coref_mod.resolve("送料はいくらですか", model=model) == "送料はいくらですか"
    assert await coref_mod.resolve("送料はいくらですか", "   \u3000\n", model=model) == "送料はいくらですか"


async def test_resolve_degrades_to_the_original_query_when_upstream_fails(caplog):
    """書き換えに失敗したら元の文で進める。止まるより害が小さい。"""
    model = _fake_chat(RuntimeError("upstream 502"))
    with caplog.at_level(logging.WARNING, logger="app.core.coref"):
        got = await coref_mod.resolve("それは返品できる?", "ユーザー:注文1001について", model=model)
    assert got == "それは返品できる?"
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


@pytest.mark.parametrize("blank", ["", "   ", "\u3000\n"])
async def test_resolve_falls_back_to_the_original_when_the_model_returns_blank(blank):
    """空を返してくることがある。そのまま通すと分類器へ空文字が渡る。"""
    model = _fake_chat(blank)
    assert await coref_mod.resolve("それは返品できる?", "ユーザー:注文1001について",
                                   model=model) == "それは返品できる?"


async def test_resolve_falls_back_when_the_content_is_not_text():
    """content は str とは限らない(block の list)。読めなければ原文へ倒す。"""

    async def _run(prompt_value):
        return AIMessage(content=[{"type": "text", "text": "書き下し"}])

    got = await coref_mod.resolve("それは?", "ユーザー:注文1001について",
                                  model=RunnableLambda(_run))
    assert got == "それは?"


# --- 検索クエリの展開(app/core/query_understanding.expand_queries)---------------


def test_expand_prompt_renders_user_query():
    msgs = EXPAND_QUERIES_PROMPT.format_messages(query="このイヤホンは返品できますか")
    assert msgs[0].type == "system"
    assert "このイヤホンは返品できますか" in msgs[-1].content


async def test_expand_queries_returns_exactly_three():
    model = _FakeModel(qu._Expanded(queries=[
        "イヤホン 返品 ポリシー", "イヤホン 自己都合返品 条件", "イヤホン 返品 申請期限"]))
    got = await qu.expand_queries("このイヤホンは返品できますか", model=model)
    assert got == ["イヤホン 返品 ポリシー", "イヤホン 自己都合返品 条件", "イヤホン 返品 申請期限"]
    assert "このイヤホンは返品できますか" in model.seen.to_messages()[-1].content


async def test_expand_queries_keeps_only_the_first_three():
    model = _FakeModel(qu._Expanded(queries=["a", "b", "c", "d"]))
    assert await qu.expand_queries("q", model=model) == ["a", "b", "c"]


async def test_expand_queries_drops_blank_entries():
    model = _FakeModel(qu._Expanded(queries=["  返品 ポリシー ", "", "\u3000", "返品 期限"]))
    assert await qu.expand_queries("q", model=model) == ["返品 ポリシー", "返品 期限"]


async def test_expand_queries_removes_duplicates():
    """同じクエリを 3 回投げても検索結果は 1 通りしか増えない。"""
    model = _FakeModel(qu._Expanded(queries=["返品 ポリシー", "返品 ポリシー", " 返品 ポリシー "]))
    assert await qu.expand_queries("q", model=model) == ["返品 ポリシー"]


async def test_expand_queries_degrades_to_the_raw_query_when_upstream_fails(caplog):
    """展開は検索の前処理。失敗しても呼び出し側の検索を止めない。"""
    model = _FakeModel(RuntimeError("upstream 502"))
    with caplog.at_level(logging.WARNING, logger="app.core.query_understanding"):
        got = await qu.expand_queries("このイヤホンは返品できますか", model=model)
    assert got == ["このイヤホンは返品できますか"]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


async def test_expand_queries_degrades_to_the_raw_query_when_everything_is_blank():
    model = _FakeModel(qu._Expanded(queries=["", "  ", "\u3000"]))
    assert await qu.expand_queries("返品したい", model=model) == ["返品したい"]


async def test_trailing_punctuation_alone_is_not_a_rewrite(monkeypatch):
    """末尾の句読点だけ足された回を「書き下した」と記録しない。

    モデルは素通しのつもりでも「。」を付けてくる(実測: 素通しすべき 9 件中 4 件)。
    そのままだと trace が rewrite だらけになり、本当に書き下した回を見分けられない。
    """
    async def _echo_with_period(q, history="", model=None):
        return q + "。"

    monkeypatch.setattr(nodes.coref_mod, "resolve", _echo_with_period)
    out = await nodes.coref({"messages": [HumanMessage("送料はいくらですか")]})
    assert out["trace"]["coref"] == "passthrough"
    assert out["resolved_query"] == "送料はいくらですか。"   # 中身は書き下し結果のまま


async def test_a_real_rewrite_is_still_recorded(monkeypatch):
    async def _rewrite(q, history="", model=None):
        return "注文1001のスマート家電は返品できますか"

    monkeypatch.setattr(nodes.coref_mod, "resolve", _rewrite)
    out = await nodes.coref({"messages": [HumanMessage("これは返品できますか")]})
    assert out["trace"]["coref"] == "rewrite"


# --- 返金フローの注文特定(_extract_order_id / fetch_order / list_user_orders)----


def _fetch_order_graph():
    """fetch_order だけの最小 graph。**interrupt() は compiled graph の中でしか動かない。**

    graph の外から node を直接呼ぶと例外になるので、interrupt を通るテストは必ず
    ここを経由する。checkpointer は InMemorySaver で、本番の
    data/05_checkpoints.sqlite には触らない。
    """
    b = StateGraph(ConversationState)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_edge(START, "fetch_order")
    b.add_edge("fetch_order", END)
    return b.compile(checkpointer=InMemorySaver())


@pytest.mark.parametrize(("text", "expected"), [
    ("注文1001について教えてください", "1001"),
    ("注文番号は 1001 です", "1001"),
    ("注文番号: 1001", "1001"),
    ("1001番の商品を返品したい", "1001"),
    ("末尾 20260701 の注文を返品したい", "20260701"),
    ("注文１００１を返品したい", "1001"),          # 全角(日本語入力でそのまま出る)
    ("order 1001 を返品したい", "1001"),
])
def test_extract_order_id_reads_common_ways_of_writing(text, expected):
    assert nodes._extract_order_id(text) == expected


@pytest.mark.parametrize("text", [
    "3日以内に返品したいです",       # 「3」を注文番号にすると存在しない注文を引く
    "2回目です",
    "10日前に届きました",
    # 単位が続かないので、桁数の下限だけが誤認を止めている 2 例
    "3営業日以内に返品できますか",
    "2番目に届いた商品を返品したい",  # 「番」は注文番号の目印でもあるので特に危ない
    "500円の商品です",
    "1500円の商品を返品したい",      # 桁数は足りるが単位付き
    "2026年に買った商品を返品したい",  # 4 桁だが年
    "返品したいのですが",
    "",
])
def test_extract_order_id_does_not_mistake_a_quantity_for_an_order(text):
    assert nodes._extract_order_id(text) is None


async def test_fetch_order_uses_the_order_number_written_in_the_question():
    out = await nodes.fetch_order(
        {"messages": [HumanMessage("注文1001を返品したい")], "user_id": "u-1"})
    assert out["order_id"] == "1001"
    assert out["order_data"] == business.order_snapshot("1001")
    assert out["trace"]["fetch_order"] == {"order_id": "1001", "source": "query"}


async def test_fetch_order_prefers_the_resolved_query():
    """「それ」を coref が書き下した文にだけ注文番号が出る場合。"""
    out = await nodes.fetch_order({
        "messages": [HumanMessage("それを返品したい")],
        "resolved_query": "注文1001を返品したい",
        "user_id": "u-1"})
    assert out["order_id"] == "1001"


async def test_fetch_order_uses_the_order_id_already_in_state():
    """State に注文番号があれば一覧は出さない。

    node を直接呼んでいるので、ここで interrupt を呼べば例外になる
    (graph の外では interrupt() は動かない)。
    """
    out = await nodes.fetch_order({"messages": [HumanMessage("返品したい")],
                                   "order_id": "2002", "user_id": "u-1"})
    assert out["order_id"] == "2002"
    assert out["order_data"] == business.order_snapshot("2002")
    assert out["trace"]["fetch_order"] == {"order_id": "2002", "source": "state"}


async def test_fetch_order_does_not_show_the_list_when_the_state_has_an_order_id():
    g = _fetch_order_graph()
    cfg = {"configurable": {"thread_id": "fetch-order-state"}}
    out = await g.ainvoke({"messages": [HumanMessage("返品したい")],
                           "order_id": "2002", "user_id": "u-1"}, cfg)
    assert "__interrupt__" not in out
    assert out["order_id"] == "2002"


async def test_fetch_order_interrupts_with_the_order_list_when_no_number_is_given():
    """注文番号が無いときは model に推測させず、一覧を画面へ出して選ばせる。"""
    g = _fetch_order_graph()
    cfg = {"configurable": {"thread_id": "fetch-order-ask"}}
    out = await g.ainvoke(
        {"messages": [HumanMessage("返品したいのですが")], "user_id": "u-1"}, cfg)

    payload = out["__interrupt__"][0].value
    assert payload["type"] == "select_order"
    assert payload["orders"] == business.list_user_orders("u-1")
    assert payload["orders"]
    for o in payload["orders"]:
        assert set(o) == {"order_id", "product", "status", "amount"}
    # 選ばれるまで注文は確定しない
    assert "order_id" not in out


@pytest.mark.parametrize(("label", "resumed"), [
    ("str", "1001"),
    ("int", 1001),
    ("dict", {"order_id": "1001"}),
    ("fullwidth", "１００１"),
    ("sentence", "注文1001"),
])
async def test_fetch_order_resume_fills_the_order_whatever_shape_the_value_has(
        label, resumed):
    """画面が何を返してくるかは決めきれない。dict でも数値でも文字列でも同じ注文に落ちる。"""
    g = _fetch_order_graph()
    cfg = {"configurable": {"thread_id": f"fetch-order-resume-{label}"}}
    await g.ainvoke({"messages": [HumanMessage("返品したいのですが")],
                     "user_id": "u-1"}, cfg)

    out = await g.ainvoke(Command(resume=resumed), cfg)
    assert out["order_id"] == "1001"
    assert out["order_data"] == business.order_snapshot("1001")
    assert out["trace"]["fetch_order"] == {"order_id": "1001", "source": "selected"}


async def test_fetch_order_does_not_invent_an_order_when_the_selection_is_unreadable():
    """読めない値で再開されても注文をでっち上げない(もう一度 interrupt もしない)。"""
    g = _fetch_order_graph()
    cfg = {"configurable": {"thread_id": "fetch-order-broken"}}
    await g.ainvoke({"messages": [HumanMessage("返品したいのですが")],
                     "user_id": "u-1"}, cfg)

    out = await g.ainvoke(Command(resume={"cancelled": True}), cfg)
    assert "__interrupt__" not in out
    assert not out.get("order_id")
    assert out["trace"]["fetch_order"] == {"order_id": None, "source": "unresolved"}


def test_list_user_orders_is_stable_for_the_same_user():
    assert business.list_user_orders("u-1") == business.list_user_orders("u-1")
    assert business.list_user_orders("u-1") != business.list_user_orders("u-2")


def test_list_user_orders_entries_match_order_snapshot():
    """一覧と詳細が食い違うと、画面で選んだ注文と後段が読む注文が別物になる。"""
    orders = business.list_user_orders("u-1")
    assert 2 <= len(orders) <= 5
    assert len({o["order_id"] for o in orders}) == len(orders)
    for o in orders:
        snap = business.order_snapshot(o["order_id"])
        assert o == {k: snap[k] for k in ("order_id", "product", "status", "amount")}


async def test_order_snapshot_is_the_only_source_of_query_order():
    from app.tools.builtin.orders import query_order

    got = await query_order.ainvoke({"order_id": "1001"})
    assert got == business.order_snapshot("1001")


def test_order_snapshot_keeps_the_chapter_02_golden_value():
    """抽出で乱数の引き順が変わっていないこと(tests/test_tools_mock.py と同じ値)。

    注文日だけは「今日から何日前か」で決まるので、絶対の日付ではなく経過日数を固定する。
    固定の日付にすると、時間が経つほど全注文が古くなり、規約の「受取後 7 日以内」を
    満たす注文が 1 件も作れなくなる(返品可能と判断される経路に到達できなくなる)。
    """
    from datetime import datetime

    snap = business.order_snapshot("1001")
    assert {k: v for k, v in snap.items() if k != "created_at"} == {
        "order_id": "1001",
        "status": "支払い済み",
        "amount": 1739,
        "product": "自動猫トイレ",
        "tracking_no": "JP213502378238",
    }
    ordered = datetime.strptime(snap["created_at"], "%Y-%m-%d %H:%M")
    assert (datetime.now().date() - ordered.date()).days == 13


# --- 返金フローの規約検索(retrieve_policy)--------------------------------------


def _policy_hit(i, score, q, a, section="ポリシー/返品", ctype="policy"):
    """規約 chunk 1 件。score=None は rerank 上流が落ちた場合(rerank_score が付かない)。"""
    h = {"id": i, "question": q, "answer": a, "section_path": section, "content_type": ctype}
    if score is not None:
        h["rerank_score"] = score
    return h


def _stub_policy(monkeypatch, per_query, queries=None):
    """クエリ展開と検索を差し替え、呼び出し引数を記録して返す。

    偽の検索にも「min_score を省略されたら rerank_min_score で足切りする」という
    本物の search_knowledge の既定を持たせてある。素通しにすると、min_score を
    渡し忘れた実装がそのままテストを通ってしまう(test_graph_forced_rag.py と同じ形)。
    """
    seen = {"searched": [], "search_kw": []}
    qs = list(per_query) if queries is None else list(queries)

    async def _fake_expand(q, **kw):
        seen["expand_arg"] = q
        return qs

    async def _fake_search(query, **kw):
        seen["searched"].append(query)
        seen["search_kw"].append(kw)
        cut = kw.get("min_score")
        cut = settings.rerank_min_score if cut is None else cut
        return [h for h in per_query.get(query, []) if h.get("rerank_score", 0.0) >= cut]

    monkeypatch.setattr(nodes.query_understanding, "expand_queries", _fake_expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", _fake_search)
    return seen


async def test_retrieve_policy_merges_the_same_chunk_and_keeps_the_highest_score(monkeypatch):
    """同じ chunk を複数のクエリが引いたら 1 件にまとめ、スコアの高い方を残す。

    3 つのクエリは観点違いなので同じ規約を引きやすい。そのまま並べると同じ条項が
    3 回出て、Agent の入力が同じ文で埋まる。
    """
    # 同じ id に別の本文を持たせているのは「どちらの hit が残ったか」を見分けるため。
    # 実際のナレッジベースでは同じ id なら本文も同じ。
    seen = _stub_policy(monkeypatch, {
        "返品 ポリシー": [_policy_hit(1, 0.7, "返品", "低いスコアで引いた方")],
        "自己都合返品 条件": [_policy_hit(1, 0.9, "返品", "受取後7日以内は返品可能"),
                        _policy_hit(2, 0.6, "返品送料", "品質不良の場合は当店負担")],
        "返品 申請期限": [],
    })
    out = await nodes.retrieve_policy({"resolved_query": "注文1001は返品できますか",
                                       "messages": [HumanMessage("これは返品できますか")]})

    assert [c["id"] for c in out["citations"]] == [1, 2]
    assert "受取後7日以内は返品可能" in out["evidence"]
    assert "低いスコアで引いた方" not in out["evidence"]
    assert seen["searched"] == ["返品 ポリシー", "自己都合返品 条件", "返品 申請期限"]


async def test_retrieve_policy_numbers_the_evidence_in_score_order(monkeypatch):
    """スコアの降順に並べ、evidence 本文の [n] と citations の n が同じ chunk を指すこと。"""
    _stub_policy(monkeypatch, {
        "q1": [_policy_hit(2, 0.6, "送料", "品質不良は当店負担")],
        "q2": [_policy_hit(1, 0.9, "返品", "受取後7日以内は返品可能")],
    })
    out = await nodes.retrieve_policy({"resolved_query": "返品できますか"})

    assert [(c["n"], c["id"]) for c in out["citations"]] == [(1, 1), (2, 2)]
    assert out["evidence"] == ("[1] 返品: 受取後7日以内は返品可能\n"
                               "[2] 送料: 品質不良は当店負担")
    assert out["citations"][0]["section_path"] == "ポリシー/返品"
    assert out["citations"][0]["content_type"] == "policy"


async def test_retrieve_policy_numbers_after_the_head_tail_arrangement(monkeypatch):
    """番号は head/tail 配置「後」の並びに振る(forced_rag / query_faq と同じ規律)。"""
    _stub_policy(monkeypatch, {
        "q1": [_policy_hit(10, 0.95, "q1", "a1"), _policy_hit(40, 0.80, "q4", "a4")],
        "q2": [_policy_hit(20, 0.90, "q2", "a2"), _policy_hit(30, 0.85, "q3", "a3")],
    })
    out = await nodes.retrieve_policy({"resolved_query": "返品できますか"})

    assert [c["id"] for c in out["citations"]] == [10, 30, 40, 20]
    assert out["evidence"].startswith("[1] q1: a1")
    assert out["evidence"].endswith("[4] q2: a2")


async def test_retrieve_policy_records_the_expanded_queries_and_the_hit_count(monkeypatch):
    """trace には展開したクエリと、重複を除いた後の件数を残す。"""
    _stub_policy(monkeypatch, {
        "返品 ポリシー": [_policy_hit(1, 0.9, "返品", "7日以内")],
        "返品 条件": [_policy_hit(1, 0.8, "返品", "7日以内"), _policy_hit(2, 0.7, "送料", "当店負担")],
        "返品 期限": [],
    })
    out = await nodes.retrieve_policy({"resolved_query": "返品できますか"})

    assert out["trace"]["retrieve_policy"] == {
        "queries": ["返品 ポリシー", "返品 条件", "返品 期限"], "hits": 2}


async def test_retrieve_policy_asks_the_search_not_to_cut_by_score(monkeypatch):
    """min_score は必ず明示する。**省略すると本当のスコアが失われる。**

    search_knowledge は hybrid_rerank で min_score を省略されると rerank_min_score で
    足切りしてから返す。こちら側は並べ替えと重複除去にそのスコアを使うので、
    向こう側で切られると「観点違いのクエリが低めに拾った同じ条項」が黙って消える。
    """
    seen = _stub_policy(monkeypatch, {
        "q1": [_policy_hit(1, 0.9, "返品", "7日以内")],
        "q2": [_policy_hit(2, 0.05, "送料", "当店負担")],   # 既定の足切り(0.3)未満
    })
    out = await nodes.retrieve_policy({"resolved_query": "返品できますか"})

    assert [kw.get("strategy") for kw in seen["search_kw"]] == \
        ["hybrid_rerank", "hybrid_rerank"]
    assert [kw.get("min_score") for kw in seen["search_kw"]] == [float("-inf")] * 2
    assert [c["id"] for c in out["citations"]] == [1, 2]


async def test_retrieve_policy_survives_hits_without_a_rerank_score(monkeypatch):
    """rerank 上流が落ちると rerank_score の無い hit が返る。並べ替えで落ちないこと。

    search_knowledge はリランクできなかった場合、ハイブリッド検索の並びのまま
    rerank_score を付けずに返す(app/core/retrieval.py)。h["rerank_score"] の直接添字は
    KeyError になり、node ではなく graph 全体が止まる。
    """
    _stub_policy(monkeypatch, {
        "q1": [_policy_hit(1, None, "返品", "7日以内"), _policy_hit(2, None, "送料", "当店負担")],
        "q2": [_policy_hit(3, 0.9, "交換", "同一商品のみ")],
    })
    out = await nodes.retrieve_policy({"resolved_query": "返品できますか"})

    # スコア順は [3, 1, 2](付いていない 2 件は検索が返した順のまま後ろへ残る)。
    # citations はそこへ head/tail 配置を掛けた後の並びなので 2 位が末尾へ回る
    assert [c["id"] for c in out["citations"]] == [3, 2, 1]
    assert out["trace"]["retrieve_policy"]["hits"] == 3


async def test_retrieve_policy_keeps_the_scored_hit_when_the_same_chunk_lacks_a_score(monkeypatch):
    """同じ chunk が「スコアあり」と「スコアなし」で来たら、スコアのある方を残す。"""
    _stub_policy(monkeypatch, {
        "q1": [_policy_hit(1, None, "返品", "リランクできなかった方")],
        "q2": [_policy_hit(1, 0.9, "返品", "受取後7日以内は返品可能")],
    })
    out = await nodes.retrieve_policy({"resolved_query": "返品できますか"})
    assert "受取後7日以内は返品可能" in out["evidence"]
    assert "リランクできなかった方" not in out["evidence"]


async def test_retrieve_policy_returns_empty_evidence_when_nothing_is_found(monkeypatch):
    """1 件も引けなくても止めない。判断材料が無いことは空の evidence として伝わる。

    **空でも必ず書く。** checkpointer が State を turn をまたいで保持するので、
    書かないと前の turn の citations が残り、根拠なしの判断の横に前回の出典が並ぶ。
    """
    _stub_policy(monkeypatch, {"q1": [], "q2": []})
    out = await nodes.retrieve_policy({"resolved_query": "返品できますか"})

    assert out["evidence"] == ""
    assert out["citations"] == []
    assert out["trace"]["retrieve_policy"] == {"queries": ["q1", "q2"], "hits": 0}


async def test_retrieve_policy_expands_the_resolved_query(monkeypatch):
    """展開するのは coref が書き下した完全な質問。"""
    seen = _stub_policy(monkeypatch, {"q1": []})
    await nodes.retrieve_policy({"resolved_query": "注文1001のイヤホンは返品できますか",
                                 "messages": [HumanMessage("これ返品できる?")]})
    assert seen["expand_arg"] == "注文1001のイヤホンは返品できますか"


async def test_retrieve_policy_falls_back_to_the_user_text(monkeypatch):
    """書き下しが無ければ元の発話で展開する(coref を通らない経路でも動く)。"""
    seen = _stub_policy(monkeypatch, {"q1": []})
    await nodes.retrieve_policy({"messages": [HumanMessage("返品したいのですが")]})
    assert seen["expand_arg"] == "返品したいのですが"


# --- 返金申請の横取り(submit_refund)と Agent への文脈注入 ------------------------


def _forbid_tool_execution(monkeypatch):
    """ツール実行を「呼ばれたら落ちる」に差し替える。

    submit_refund は create_ticket と同じで**実行してはいけない**ツールなので、
    「実行されていないこと」をテストの書き方ではなく仕組みで確かめる。
    """

    async def _boom(*args, **kwargs):
        raise AssertionError("横取りすべきツールを実行した")

    monkeypatch.setattr(nodes.engine, "execute_tool_call", _boom)


def _ai_calling(*tool_calls):
    return AIMessage(content="", tool_calls=list(tool_calls))


async def test_submit_refund_is_never_executed(monkeypatch):
    """submit_refund は**実行しない**。DB にも書かない。

    実際に返金申請を作るのは、ユーザーが画面のボタンを押したときだけ。Agent が
    呼んだ時点で申請を立てると、ユーザーが望んでいない申請が運用側へ流れる
    (create_ticket / complaint_reply と同じ方針)。
    """
    _forbid_tool_execution(monkeypatch)
    tc = {"name": "submit_refund", "args": {"order_id": "1001", "reason": "初期不良"},
          "id": "r1"}
    out = await nodes.agent_tools({"messages": [_ai_calling(tc)], "conversation_id": 42})

    assert out["suggested_actions"] == [
        {"type": "refund_form", "draft": {"order_id": "1001", "reason": "初期不良"}}
    ]


async def test_intercepted_refund_answers_the_original_tool_call_id(monkeypatch):
    """合成 ToolMessage の tool_call_id が元の tool_call と一致すること。

    上流は tool_calls と ToolMessage の対応を検査しており、食い違うと 400 になって
    loop がその場で止まる(create_ticket の横取りと同じ制約)。
    """
    _forbid_tool_execution(monkeypatch)
    tc = {"name": "submit_refund", "args": {"order_id": "1001"}, "id": "call_r_abc"}
    out = await nodes.agent_tools({"messages": [_ai_calling(tc)]})

    assert len(out["messages"]) == 1
    tm = out["messages"][0]
    assert tm.tool_call_id == "call_r_abc"
    assert tm.name == "submit_refund"
    assert "返金" in tm.content


async def test_refund_draft_falls_back_to_the_order_id_in_state(monkeypatch):
    """args に注文番号が無ければ State の値で補う。

    Agent は文脈から番号を落とすことがあり、空の draft を画面へ出すと
    ユーザーには何の申請フォームか分からない。
    """
    _forbid_tool_execution(monkeypatch)
    tc = {"name": "submit_refund", "args": {"reason": "サイズが合わない"}, "id": "r1"}
    out = await nodes.agent_tools({"messages": [_ai_calling(tc)], "order_id": "2002"})

    assert out["suggested_actions"][0]["draft"] == {"order_id": "2002",
                                                    "reason": "サイズが合わない"}


async def test_refund_draft_keeps_reason_none_when_the_agent_gives_none(monkeypatch):
    """理由は任意。無ければ None のまま渡す(画面の選択肢で確定させる)。"""
    _forbid_tool_execution(monkeypatch)
    tc = {"name": "submit_refund", "args": {"order_id": "1001"}, "id": "r1"}
    out = await nodes.agent_tools({"messages": [_ai_calling(tc)]})
    assert out["suggested_actions"][0]["draft"] == {"order_id": "1001", "reason": None}


async def test_a_refund_beside_another_tool_keeps_every_tool_message(monkeypatch):
    """1 step に submit_refund と別のツールが並んでも、submit_refund は横取りされ、
    tool_calls と同じ順・同じ id で ToolMessage が揃うこと。

    片方だけ処理して step を打ち切ると、ToolMessage の対応が揃わずに上流が 400 を返す。

    08 で create_ticket は横取りではなく確認フロー(interrupt)になったので、
    ここでは横取りが残っている submit_refund と通常ツールの組み合わせで見る
    (確認フローは tests/test_graph_08_confirm_ticket.py)。
    """
    async def _exec(tc, cid, specs):
        assert tc["name"] == "query_order"      # 横取りすべき submit_refund は来ない
        return ToolRun(tool_call_id=tc["id"], name=tc["name"], ok=True, status="success",
                       tool_message=ToolMessage(content="{}", tool_call_id=tc["id"],
                                                name=tc["name"]))

    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    order = {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
    refund = {"name": "submit_refund", "args": {"order_id": "1001"}, "id": "r1"}
    out = await nodes.agent_tools({"messages": [_ai_calling(order, refund)],
                                   "conversation_id": 7})

    assert [m.tool_call_id for m in out["messages"]] == ["c1", "r1"]
    assert [a["type"] for a in out["suggested_actions"]] == ["refund_form"]


# 07 章で置き場所が変わった: 注文と規約と判断の指示は system への連結をやめ、
# 最後の HumanMessage の直後へ 1 通の SystemMessage(材料)として入れる。
# **届く中身は 06 章のまま**で、どの message に入るかだけが変わっている。


def test_agent_messages_injects_the_order_the_policy_and_the_judgement_task():
    """refund route では注文の中身・規約・判断の指示をまとめて材料に入れる。"""
    msgs = nodes._agent_messages({
        "route": "refund_flow",
        "order_data": business.order_snapshot("1001"),
        "evidence": "[1] 返品: 受取後7日以内は返品可能",
        "messages": [HumanMessage("この注文は返品できますか")]})
    ctx = msgs[-1]

    assert msgs[0].content == AGENT_SYSTEM                      # system は素のまま
    assert msgs[1] == HumanMessage("この注文は返品できますか")
    assert isinstance(ctx, SystemMessage)
    assert "[1] 返品: 受取後7日以内は返品可能" in ctx.content   # 規約
    assert "自動猫トイレ" in ctx.content and "1001" in ctx.content  # 注文の中身
    assert "submit_refund" in ctx.content                       # 可能なら呼ぶ
    assert "[n]" in ctx.content                                 # 不可なら番号を引いて説明
    assert "規約に無い条件" in ctx.content                       # 条件をでっち上げない


def test_agent_messages_does_not_escape_japanese_in_the_order_data():
    """注文の中身は日本語のまま入れる。unicode escape に化けるとモデルが読む文字数が増える。"""
    ctx = nodes._agent_messages({"route": "refund_flow",
                                 "order_data": {"status": "発送済み"}, "messages": []})[-1]
    assert "発送済み" in ctx.content


def test_agent_messages_still_injects_evidence_on_the_knowledge_route():
    """05 の挙動を壊していないこと(条件を緩めた後も knowledge route は同じ)。"""
    ctx = nodes._agent_messages({
        "route": "knowledge", "evidence": "[1] 送料: 3,000円以上は送料無料",
        "messages": [HumanMessage("送料はいくらですか")]})[-1]
    assert "3,000円以上は送料無料" in ctx.content
    assert "query_faq" in ctx.content and "再度呼ばないでください" in ctx.content


def test_agent_messages_does_not_inject_empty_evidence():
    """evidence が空なら見出しごと足さない。

    retrieve_policy も forced_rag も、引けなかったときは evidence="" を書く。
    見出しだけ付いた空の evidence は「根拠はあるが中身が無い」という誤った指示になる。
    """
    assert nodes._agent_messages(
        {"route": "knowledge", "evidence": "", "messages": []}) == [SystemMessage(AGENT_SYSTEM)]

    ctx = nodes._agent_messages({"route": "refund_flow", "evidence": "",
                                 "order_data": {"order_id": "1001"}, "messages": []})[-1]
    assert "retrieval 済み" not in ctx.content   # evidence の見出しは出ない
    assert "submit_refund" in ctx.content        # 判断の指示は残る


def test_agent_messages_leaves_the_business_route_alone():
    """business route は注文の指示も規約も足さない(検索そのものをしていない)。"""
    msgs = nodes._agent_messages({"route": "business",
                                  "messages": [HumanMessage("注文1001はどこ")]})
    assert msgs[0].content == AGENT_SYSTEM
    assert len(msgs) == 2 and isinstance(msgs[1], HumanMessage)   # 材料は付かない


def test_submit_refund_is_registered_for_the_model():
    """モデルへ渡すツール一覧に入っていること(入っていないと呼びようがない)。

    08 章で一覧は built-in と MCP の合成になったが、submit_refund は built-in 側に
    留まる(横取りする以上、外部のサーバへ出す意味がない)。MCP を待たずに
    built-in の一覧だけで確かめられるので、ここでは生きた Server に依存させない。
    """
    from app.tools import registry
    assert registry.get_builtin_spec("submit_refund") is not None
    assert "submit_refund" in {s.name for s in registry.builtin_specs()}


async def test_no_refund_form_without_an_order(monkeypatch):
    """注文が特定できていないのに申請フォームを出さない。

    submit_refund は全 route で bind されるので、注文を確かめていない business route
    からも呼ばれうる。空の draft を出すとユーザーには何の申請か分からないフォームが出て、
    押しても弾かれる。モデルには「提示していない」と正直に返して聞き直させる。
    """
    async def _boom(tc, cid, specs):
        raise AssertionError("横取りすべきツールを実行した")

    monkeypatch.setattr(nodes.engine, "execute_tool_call", _boom)
    ai = AIMessage("", tool_calls=[
        {"id": "r1", "name": "submit_refund", "args": {"reason": "初期不良"}}])
    out = await nodes.agent_tools({"messages": [HumanMessage("返金して"), ai]})
    assert not out.get("suggested_actions")
    assert "提示していません" in out["messages"][0].content


async def test_the_refund_form_appears_once_the_order_is_known(monkeypatch):
    async def _boom(tc, cid, specs):
        raise AssertionError("横取りすべきツールを実行した")

    monkeypatch.setattr(nodes.engine, "execute_tool_call", _boom)
    ai = AIMessage("", tool_calls=[
        {"id": "r1", "name": "submit_refund", "args": {"reason": "初期不良"}}])
    out = await nodes.agent_tools({"messages": [HumanMessage("返金して"), ai],
                                   "order_id": "1001"})
    assert out["suggested_actions"] == [
        {"type": "refund_form", "draft": {"order_id": "1001", "reason": "初期不良"}}]
    assert "提示しました" in out["messages"][0].content


async def test_policy_retrieval_is_capped(monkeypatch):
    """規約を渡しすぎない。3 クエリ分をそのまま入れると判断に効く条項が埋もれる。"""
    async def _expand(q, model=None):
        return ["a", "b", "c"]

    def _hit(i):
        return {"id": i, "question": f"q{i}", "answer": f"a{i}", "rerank_score": 1.0 - i / 100,
                "section_path": "p", "content_type": "policy"}

    async def _search(q, **k):
        return [_hit(i) for i in range(30)]

    monkeypatch.setattr(nodes.query_understanding, "expand_queries", _expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", _search)
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)
    out = await nodes.retrieve_policy({"resolved_query": "返品できますか"})
    assert len(out["citations"]) == nodes._POLICY_TOP_K
    assert out["citations"][0]["id"] == 0          # スコアの高い順に残る


# --- 雑談 / その他の出口(script_reply)------------------------------------------


async def test_script_reply_guides_chitchat_back_to_shopping():
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT

    out = await nodes.script_reply({"intent": "雑談", "messages": [HumanMessage("いい天気ですね")]})
    assert out["answer"] == SCRIPT_REPLY_CHITCHAT
    assert out["trace"]["route"] == "fallback_script"


async def test_script_reply_asks_the_other_intent_to_be_specific():
    from app.core.prompts import SCRIPT_REPLY_OTHER

    out = await nodes.script_reply({"intent": "その他", "messages": [HumanMessage("うーん")]})
    assert out["answer"] == SCRIPT_REPLY_OTHER
    assert out["trace"]["route"] == "fallback_script"


async def test_script_reply_texts_differ():
    """2 つの文面が同じなら intent で出し分ける意味が無い。"""
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT, SCRIPT_REPLY_OTHER

    assert SCRIPT_REPLY_CHITCHAT != SCRIPT_REPLY_OTHER


@pytest.mark.parametrize("intent", ["", "未知", "配送"])
async def test_script_reply_asks_to_be_specific_when_the_intent_is_not_chitchat(intent):
    """雑談と確信できないものは、挨拶ではなく「具体的に」へ倒す。

    ここへ来るのは 雑談 / その他 の 2 つだけ(routing.INTENT_TO_ROUTE)なので、
    それ以外が届いた時点で分類側が壊れている。買い物の話題へ案内し直す挨拶文は
    「雑談だと分かっている」ことが前提なので、分からないときに出すと的外れになる。
    用件を聞き直す方が、どちらに転んでも会話が進む。
    """
    from app.core.prompts import SCRIPT_REPLY_OTHER

    out = await nodes.script_reply({"intent": intent, "messages": [HumanMessage("?")]})
    assert out["answer"] == SCRIPT_REPLY_OTHER


async def test_script_reply_offers_no_actions():
    """選択肢を出す出口ではない。雑談に有人対応の導線を付けない。"""
    out = await nodes.script_reply({"intent": "雑談", "messages": [HumanMessage("やあ")]})
    assert "suggested_actions" not in out


async def test_script_reply_names_this_store():
    """名乗る店名は既存プロンプトと同じ「STORE」であること。"""
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT

    assert "STORE" in SCRIPT_REPLY_CHITCHAT


async def test_script_reply_does_not_call_the_model(monkeypatch):
    """model を呼ばないことがこの node の存在理由。取りに行った時点で落とす。"""
    def _boom(*args, **kwargs):
        raise AssertionError("script_reply が上流のモデルを呼んだ")

    monkeypatch.setattr(nodes, "get_chat_model", _boom)
    await nodes.script_reply({"intent": "雑談", "messages": [HumanMessage("やあ")]})
    await nodes.script_reply({"intent": "その他", "messages": [HumanMessage("うーん")]})


# ---------------------------------------------------------------------------
# 返品可否の判断に要る「今日」
#
# 規約の条件はどれも「受取後 7 日以内」のように日数で書かれているのに、モデルには
# 今日が何日かが分からない。実測: 経過日数を渡さないと、返品できる注文でも
# submit_refund を呼ばず「受取からの日数を教えてください」と聞き返して turn が
# 終わる(5 回中 3 回)。ユーザーが「一昨日届いて」と自分で言った回だけ先へ進んだ。
# ---------------------------------------------------------------------------

def test_the_refund_context_carries_todays_date_and_elapsed_days():
    from datetime import datetime, timedelta

    ordered = datetime.now() - timedelta(days=3)
    msgs = nodes._agent_messages({
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "created_at": ordered.strftime("%Y-%m-%d %H:%M")},
        "evidence": "[1] 受取後7日以内",
        "messages": [HumanMessage("返品できますか")]})
    ctx = msgs[-1].content
    assert "本日は" in ctx
    assert "経過日数は 3 日" in ctx
    assert "尋ね直さないでください" in ctx


def test_an_unreadable_order_date_adds_nothing():
    """日付が読めないときは黙って足さない。誤った日数を渡すより聞き返させる方がよい。"""
    for bad in [{"created_at": "不明"}, {"created_at": None}, {}]:
        msgs = nodes._agent_messages({
            "route": "refund_flow", "order_data": bad,
            "messages": [HumanMessage("返品できますか")]})
        assert "本日は" not in "".join(str(m.content) for m in msgs)


def test_the_date_note_is_only_for_the_refund_route():
    msgs = nodes._agent_messages({
        "route": "business",
        "order_data": {"created_at": "2026-07-04 10:00"},
        "messages": [HumanMessage("注文1001は?")]})
    assert "本日は" not in "".join(str(m.content) for m in msgs)


# ---------------------------------------------------------------------------
# 規約そのものを聞かれたときは注文を要求しない
#
# 返金返品の intent には「自分の注文を返品したい」と「返品の規約を知りたい」の
# 両方が入る(intent の分類はここを分けていない)。後者で注文の一覧を出すと、規約を
# 聞いただけのユーザーが本文なしのカードの山を見て、選ぶ以外に進めなくなる。
# 実測: 「返品交換ポリシー 教えて」で select_order の中断が起き、回答が空になった。
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "返品交換ポリシー　教えて",
    "返品ポリシーを教えてください",
    "返品の条件は?",
    "返品は何日以内ですか",
    "交換のルールを知りたい",
    "返金はいつまでにできますか",
])
def test_a_policy_question_does_not_need_an_order(text):
    assert nodes._needs_an_order(text) is False


@pytest.mark.parametrize("text", [
    "返金したい",
    "返品したいです",
    "この前買ったものを返したいです",
    "返品手続きをお願いします",
    "返金してほしいのですが",
    "返品してください",
])
def test_acting_on_an_item_needs_an_order(text):
    assert nodes._needs_an_order(text) is True


def test_polite_request_phrasing_is_not_an_action():
    """「教えてください」を動作の意思表示と取らないこと。

    「ください」単体を動作の印にすると、丁寧に規約を尋ねただけで注文の一覧が出る。
    """
    assert nodes._needs_an_order("返品ポリシーを教えてください") is False
    assert nodes._needs_an_order("返品してください") is True


async def test_fetch_order_skips_the_selector_for_a_policy_question(monkeypatch):
    """規約の質問では一覧を出さず、注文なしで先へ進む。"""
    def _boom(uid):
        raise AssertionError("規約の質問で注文の一覧を出してはいけない")

    monkeypatch.setattr(nodes, "list_user_orders", _boom)
    out = await nodes.fetch_order({"resolved_query": "返品ポリシーを教えてください",
                                   "user_id": "u1"})
    assert out["trace"]["fetch_order"]["source"] == "not_needed"
    assert "order_id" not in out          # 注文は書かない
