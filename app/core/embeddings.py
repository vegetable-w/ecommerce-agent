from openai import AsyncOpenAI

from app.config import settings


def _client() -> AsyncOpenAI:
    """埋め込み上流(SiliconFlow の bge-m3、OpenAI 互換)へ直接接続する。"""
    return AsyncOpenAI(
        base_url=settings.embed_base_url,
        # SecretStr のまま渡すと SDK が str を期待して落ちるので、ここで開く
        api_key=settings.embed_api_key.get_secret_value(),
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = await _client().embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
