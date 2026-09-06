from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # この3項目には意図的にデフォルト値を設定しない。不足時は起動時にField requiredとなり、.envの不足を直接示す
    chat_model: str
    chat_base_url: str
    chat_api_key: SecretStr
    # 0以下だとtrim_messagesが履歴を毎回無音で全消去するため、下限を設ける
    token_budget: int = Field(default=2000, gt=0)

settings = Settings()
