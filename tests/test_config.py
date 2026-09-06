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
