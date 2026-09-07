from app.config import settings
from app.core.llm import DEFAULT_TEMPERATURE, get_chat_model


def test_factory_points_to_chat_upstream(monkeypatch):
    # ハードコードされた値でも偶然一致してしまわないよう、settingsの値を実在しそうにない
    # マーカー文字列に差し替えてから、factoryが本当にsettingsを読んでいることを検証する
    monkeypatch.setattr(settings, "chat_model", "distinct-test-marker-model")
    monkeypatch.setattr(settings, "chat_base_url", "http://distinct-test-marker-url/v1")
    m = get_chat_model()
    assert m.model_name == settings.chat_model
    assert str(m.openai_api_base) == settings.chat_base_url


def test_factory_streaming_flag():
    assert get_chat_model(streaming=True).streaming is True


def test_factory_uses_configured_request_timeout(monkeypatch):
    monkeypatch.setattr(settings, "request_timeout", 12.5)
    m = get_chat_model()
    assert m.request_timeout == settings.request_timeout


def test_factory_default_temperature_is_for_conversation():
    """既定は接客の返答向けの 0.3。judge 以外の呼び出し側の挙動を変えていないこと。"""
    assert get_chat_model().temperature == DEFAULT_TEMPERATURE
    assert DEFAULT_TEMPERATURE == 0.3


def test_factory_forwards_temperature():
    """judge が 0 を要求できること。ここが素通しでないと判定が実行ごとに揺れる。"""
    assert get_chat_model(temperature=0).temperature == 0
    assert get_chat_model(temperature=0, streaming=True).temperature == 0
