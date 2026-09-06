from app.config import Settings

import pytest
from pydantic import ValidationError

def test_アカウント関連の3項目が必須(monkeypatch):
    # この3項目はアカウントごとに変わる。デフォルト値を設定すると推測で補うことになり、誤っていても起動時には失敗せず、初回の上流呼び出し時に
    # 分かりにくい401を返すことになる
    for k in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_settings_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TOKEN_BUDGET", "500")
    assert Settings(_env_file=None).token_budget == 500

def test_chat_api_keyはSecretStrでreprに漏れない(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "super-secret-key"}.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert "super-secret-key" not in repr(s)
    assert s.chat_api_key.get_secret_value() == "super-secret-key"

def test_token_budgetはゼロ以下だとエラー(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TOKEN_BUDGET", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_token_budgetは負数だとエラー(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TOKEN_BUDGET", "-5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_token_budgetは1なら許可される(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TOKEN_BUDGET", "1")
    assert Settings(_env_file=None).token_budget == 1

def test_request_timeoutはゼロ以下だとエラー(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("REQUEST_TIMEOUT", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_request_timeoutは負数だとエラー(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("REQUEST_TIMEOUT", "-1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_env_fileから読み込まれる(tmp_path, monkeypatch):
    for k in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY", "TOKEN_BUDGET"):
        monkeypatch.delenv(k, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHAT_MODEL=from-file-model\n"
        "CHAT_BASE_URL=http://from-file/v1\n"
        "CHAT_API_KEY=from-file-key\n"
        "TOKEN_BUDGET=777\n",
        encoding="utf-8",
    )
    s = Settings(_env_file=env_file)
    assert s.chat_model == "from-file-model"
    assert s.token_budget == 777
    assert s.token_budget != 2000
