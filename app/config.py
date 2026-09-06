from typing import Literal

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
    # with_structured_output()の抽出方式。json_schemaはOpenAI固有のStructured Outputs機能で、
    # スキーマ強制力が最も強いため、本番の上流であるOpenAIを前提にデフォルトとする(弱い方式に
    # 下げない)。DeepSeekやOllamaなどOpenAI互換だが json_schema 未対応の上流に切り替える場合は、
    # 通常function_callingを選ぶ必要がある。移植性は設定で担保し、デフォルトは弱めない
    extract_method: Literal["json_schema", "function_calling", "json_mode"] = "json_schema"
    database_url: str = "mysql+asyncmy://root:root@localhost:3306/support"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3306/support_test"

    # 埋め込み上流。OpenAI 互換なので openai SDK の base_url を差し替えるだけで話せる。
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: SecretStr                # アカウント依存なので既定値を持たせない
    embed_model: str = "BAAI/bge-m3"        # 上流の実名。別名レイヤーは設けない
    # Milvus は docker-compose の standalone(ポート 19530)へ接続する。
    # Milvus Lite(埋め込みファイル DB)は sys_platform != 'win32' の marker で
    # Windows を除外しているため、このマシンでは使えない(実測で確認済み)。
    milvus_uri: str = "http://localhost:19530"
    retrieval_top_k: int = Field(default=3, gt=0)
    retrieval_min_score: float = 0.4

settings = Settings()
