from langchain_openai import ChatOpenAI

from app.config import settings


def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    """チャット上流へ直接接続する。URL、モデル名、APIキーはすべて.envから取得し、上流を変更してもここは変更しない。"""
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=0.3,
    )
