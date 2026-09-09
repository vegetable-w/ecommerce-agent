"""app/core/observability.py — Langfuse の取り付けが「付加価値」に留まっていることを確かめる。

このモジュールで一番守りたいのは機能そのものではなく、**無いときに何も起きないこと**。
observability は本番で外部サービスに依存する唯一の任意機能で、Langfuse が落ちていても
設定が抜けていても、ユーザーの会話は今までどおり成立しなければならない。

だから確かめる中心は 3 つ。

1. **3 config が揃わない限り enabled にしない。** 部分設定で「とりあえず有効」にすると、
   base_url が空のとき SDK の既定が cloud へ向き、self-hosted のつもりで会話の中身を
   外へ送る。揃っていなければ off、が唯一安全な既定。
2. **disabled のとき graph に一切触らない。** with_config は新しい Runnable を返すので、
   同一性(`is`)で見れば「本当に素通ししたか」が分かる。
3. **disabled のとき tag_intent が langfuse を要求しない。** node から毎ターン呼ばれる
   関数なので、ここが import や通信を試みると Langfuse 無しの環境で毎ターン失敗する。

enabled 側は _init_client / _make_handler を差し替えて、**callback の付け方**だけを見る。
本物の Langfuse server も上流も呼ばない。
"""

import pytest

from app.core import observability


@pytest.fixture
def enabled(monkeypatch):
    """3 config を揃え、client の singleton を空に戻す。"""
    monkeypatch.setattr(observability.settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(observability.settings, "langfuse_secret_key", "sk-test")
    monkeypatch.setattr(observability.settings, "langfuse_base_url", "http://localhost:3000")
    monkeypatch.setattr(observability, "_client", None)


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.setattr(observability.settings, "langfuse_public_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_secret_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_base_url", "")
    monkeypatch.setattr(observability, "_client", None)


# --- 1. 3 config が揃ったときだけ enabled ---

def test_3つ揃えば有効(enabled):
    assert observability.langfuse_enabled() is True


@pytest.mark.parametrize("missing", ["langfuse_public_key",
                                     "langfuse_secret_key",
                                     "langfuse_base_url"])
def test_1つでも欠ければ無効(enabled, monkeypatch, missing):
    # 「key はあるが base_url が無い」を有効にすると self-hosted のつもりで
    # cloud へ送ることになる。3 つは全か無かで扱う
    monkeypatch.setattr(observability.settings, missing, "")
    assert observability.langfuse_enabled() is False


# --- 2. disabled のとき graph に触らない ---

def test_無効なら_graph_をそのまま返す(disabled):
    sentinel = object()
    assert observability.attach_observability(sentinel) is sentinel


def test_無効なら_langfuse_を初期化しない(disabled, monkeypatch):
    # 「そのまま返した」だけでなく「client を作らなかった」ことまで見る。
    # 未設定の環境では langfuse が入っていない可能性もあるので、
    # import すること自体が失敗になりうる
    def _boom():
        raise AssertionError("無効なのに Langfuse client を初期化した")

    monkeypatch.setattr(observability, "_init_client", _boom)
    monkeypatch.setattr(observability, "_make_handler", _boom)

    sentinel = object()
    assert observability.attach_observability(sentinel) is sentinel
    assert observability.get_langfuse() is None


# --- 3. disabled のとき tag_intent は完全な no-op ---

def test_無効なら_tag_intent_は何もしない(disabled, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("無効なのに Langfuse を触った")

    monkeypatch.setattr(observability, "_init_client", _boom)

    # 例外を投げないこと自体が assertion。node は戻り値を見ない
    assert observability.tag_intent("配送", 0.9) is None


def test_有効でも_tag_intent_の失敗は外へ出さない(enabled, monkeypatch):
    # Langfuse server が落ちている / key が古い、はこの機能の通常の故障で、
    # そのたびに turn が落ちては本末転倒。client の生成から握れていることを見る
    def _boom():
        raise RuntimeError("Langfuse に繋がらない")

    monkeypatch.setattr(observability, "_init_client", _boom)
    assert observability.tag_intent("配送", 0.9) is None


# --- 4. enabled のときは with_config で callback を 1 つ付ける ---

class _FakeGraph:
    def __init__(self):
        self.configs = []

    def with_config(self, config):
        self.configs.append(config)
        return f"configured:{len(self.configs)}"


def test_有効なら_callback_付きの_graph_を返す(enabled, monkeypatch):
    handler = object()
    inits = []
    monkeypatch.setattr(observability, "_init_client", lambda: inits.append(1))
    monkeypatch.setattr(observability, "_make_handler", lambda: handler)

    graph = _FakeGraph()
    result = observability.attach_observability(graph)

    # 返るのは with_config の結果であって元の graph ではない
    assert result == "configured:1"
    # callback は「1 回だけ」。2 つ付くと同じ span が二重に送られる
    assert graph.configs == [{"callbacks": [handler]}]
    # handler を作る前に client の singleton を用意していること
    # (CallbackHandler は singleton の設定を見に行くので順序が意味を持つ)
    assert inits == [1]


def test_有効なら_get_langfuse_は_client_を返す(enabled, monkeypatch):
    client = object()
    monkeypatch.setattr(observability, "_init_client", lambda: client)
    assert observability.get_langfuse() is client


# --- 5. tag_intent が書く属性 ---

def test_tag_intent_は現在の_span_へ_tag_と_metadata_を書く(enabled, monkeypatch):
    """plan の update_current_trace は 4.15.1 に無く、代替案の ingestion upsert も
    v4 server に 400 で弾かれる。残る公式の口が propagate_attributes。

    確かめるのは 2 点。**intent が tag と metadata の両方に入ること**
    (tag は Langfuse 上での絞り込み、metadata は Task 11 の集計に使うので片方では足りない)と、
    **context を開きっぱなしにしないこと**(閉じ忘れると以降の node がすべてこの
    context の中に入り、intent が後続の全 span へ漏れる)。
    """
    calls = []
    exits = []

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            exits.append(True)
            return False

    def _fake_propagate(**kwargs):
        calls.append(kwargs)
        return _Ctx()

    import langfuse
    monkeypatch.setattr(observability, "_init_client", lambda: object())
    monkeypatch.setattr(langfuse, "propagate_attributes", _fake_propagate)

    observability.tag_intent("配送", 0.87)

    assert calls == [{
        "tags": ["intent:配送"],
        "metadata": {"intent": "配送", "intent_confidence": 0.87},
    }]
    assert exits == [True]


def test_tag_intent_は_propagate_の失敗を外へ出さない(enabled, monkeypatch):
    """Langfuse が落ちている / SDK の API が変わった、で turn を落とさない。"""
    import langfuse

    def _boom(**kwargs):
        raise RuntimeError("SDK が変わった")

    monkeypatch.setattr(observability, "_init_client", lambda: object())
    monkeypatch.setattr(langfuse, "propagate_attributes", _boom)

    assert observability.tag_intent("配送", 0.87) is None


# --- 6. runtime の配線 ---

def test_runtime_の_config_に_session_id_が入る():
    """会話 ID がそのまま Langfuse の session になる。

    turn の入口 3 つはどれも runtime._config() から config を得ているので、
    ここを見れば run_turn / resume_turn / _stream_events の 3 つを同時に押さえられる。
    thread_id を壊していないことも一緒に見る(こちらは checkpointer の生命線)。
    """
    from app.graph import runtime

    cfg = runtime._config(42)
    assert cfg["configurable"]["thread_id"] == "42"
    assert cfg["metadata"]["langfuse_session_id"] == "42"
