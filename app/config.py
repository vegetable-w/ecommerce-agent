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
    # クライアント層はデフォルトでタイムアウトなし(無制限に待つ)。上流がハングすると
    # ワーカーが無期限に塞がれるため、下限を設けた上で明示的な既定値を持たせる
    request_timeout: float = Field(default=60.0, gt=0)

settings = Settings()
