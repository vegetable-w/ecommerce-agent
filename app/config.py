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
    # 実測に基づく既定値(31 chunk のナレッジ、ラベル付きサンプル 8 件 + 無関係な質問 7 件で計測):
    #   業務の質問のスコアは 0.504〜0.785、無関係な質問は 0.313〜0.451。分離幅は 0.053 しかない。
    #   閾値 0.4 では無関係な質問 7 件中 4 件が通ってしまうため、中点の 0.48 を採る。
    #   top_k は 3 だと「買ったものを返したい」の正解が 4 位で圏外になる(1〜4 位のスコア差が
    #   0.02 しかなく dense 単路では並べ替えきれない)。5 にすると eval が 8/8 になる。
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_min_score: float = 0.48
    # /kb の再実行ボタンは make 経由でジョブを起動する。Windows では make が PATH に
    # 載っていないことがあり(winget の Packages 配下に実体だけある)、アプリのプロセスから
    # shutil.which("make") が None になる。その場合はここに実体のパスを設定する。
    make_bin: str = "make"

    # リランク上流。OpenAI protocol ではなく Jina / Cohere 系の /rerank shape を使うため、
    # embedding とは別の設定群を持たせる(HTTP request も手書きする)。
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"   # 上流の実名。BAAI/ prefix を含む
    # 未設定なら埋め込みの key を使う。どちらも同じ SiliconFlow のアカウントであり、
    # .env に同じ値を 2 度書かせない。上流を分ける場合だけ RERANK_API_KEY を設定する。
    rerank_api_key: SecretStr | None = None

    @property
    def rerank_key(self) -> SecretStr:
        return self.rerank_api_key or self.embed_api_key

settings = Settings()
