from langchain_openai import ChatOpenAI

from app.config import settings


DEFAULT_TEMPERATURE = 0.3


def get_chat_model(streaming: bool = False,
                   temperature: float = DEFAULT_TEMPERATURE) -> ChatOpenAI:
    """チャット上流へ直接接続する。URL、モデル名、APIキーはすべて.envから取得し、上流を変更してもここは変更しない。

    temperature の既定は接客の返答向けの 0.3。評価の judge のように「同じ入力には
    同じ判定を返してほしい」用途では 0 を明示して呼ぶ。既定のまま judge を回すと
    判定が実行ごとに揺れる(同一入力を 0.3 で 4 回流して 5/6、5/6、6/6、6/6 になり、
    しかも外したケースが毎回違った)。
    """
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=temperature,
        request_timeout=settings.request_timeout,
    )
