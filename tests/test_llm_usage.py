"""09 コスト集計の前提：streaming の呼び出しでも usage が返ること。

**これは机上の設定確認ではなく、回帰の番人**。`streaming=True` の既定では
usage_metadata が None になり(実測)、agent_llm が積む tokens_used が常に 0 に
なっていた。intent 別のコスト集計はこの値の上に立つので、`stream_usage` が
外れたら気づけるようにする。

上流は呼ばない。constructor に渡った設定だけを見る。
"""

from app.core.llm import get_chat_model


def test_streaming_model_asks_the_upstream_for_usage():
    """streaming でも usage を返させる設定になっている。"""
    m = get_chat_model(streaming=True)
    assert m.stream_usage is True


def test_non_streaming_model_also_carries_the_flag():
    """非 streaming 側も同じ設定にする。呼び方で有無が変わると集計が歯抜けになる。"""
    assert get_chat_model().stream_usage is True


def test_model_override_keeps_usage_enabled():
    """モデルを差し替える呼び方でも落ちない(judge や intent 用の別モデル)。"""
    assert get_chat_model(model="some-other-model").stream_usage is True
