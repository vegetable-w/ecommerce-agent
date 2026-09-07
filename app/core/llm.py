from langchain_openai import ChatOpenAI

from app.config import settings


DEFAULT_TEMPERATURE = 0.3


def get_chat_model(streaming: bool = False,
                   temperature: float = DEFAULT_TEMPERATURE,
                   model: str | None = None) -> ChatOpenAI:
    """チャット上流へ直接接続する。URL、モデル名、APIキーはすべて.envから取得し、上流を変更してもここは変更しない。

    temperature の既定は接客の返答向けの 0.3。評価の judge のように「同じ入力には
    同じ判定を返してほしい」用途では 0 を明示して呼ぶ。既定のまま judge を回すと
    判定が実行ごとに揺れる(同一入力を 0.3 で 4 回流して 5/6、5/6、6/6、6/6 になり、
    しかも外したケースが毎回違った)。

    model は「この呼び出しだけ別のモデルを使いたい」用途の上書き。空(None または "")なら
    settings.chat_model へ倒す。**空文字も未設定として扱う**のは、上書き元になる設定
    (settings.intent_model など)の既定が空文字であり、呼び出し側それぞれに
    `or None` を書かせると 1 箇所書き忘れた時点でモデル名が空のまま上流へ飛ぶため。
    base_url と api_key は分けない(上流ごと差し替える話は 06 の範囲ではない)。
    """
    return ChatOpenAI(
        model=model or settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=temperature,
        request_timeout=settings.request_timeout,
    )
