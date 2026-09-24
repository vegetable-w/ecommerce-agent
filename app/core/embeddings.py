"""埋め込み上流の入口。**文書側と質問側を別の関数にしている。**

モデルによっては、埋め込む文字列の頭に役割を示す接頭辞を付ける必要がある
(ruri v3 の `検索文書: ` / `検索クエリ: ` など)。非対称なモデルでこれを付けないと
**エラーにはならず、検索の精度だけが静かに落ちる**。付け忘れを構造的に防ぐため、
汎用の 1 関数ではなく用途ごとの入口にし、接頭辞は設定から引く。

bge-m3 のような対称モデルでは接頭辞を空にしておけば、両方とも素の文字列を送る。
"""

from openai import AsyncOpenAI

from app.config import settings


def _client() -> AsyncOpenAI:
    """埋め込み上流(OpenAI 互換)へ直接接続する。

    ローカルの推論サービスへ向ける場合も同じ形で話せるので、差し替えは
    embed_base_url と embed_model の 2 つで済む。
    """
    return AsyncOpenAI(
        base_url=settings.embed_base_url,
        # SecretStr のまま渡すと SDK が str を期待して落ちるので、ここで開く
        api_key=settings.embed_api_key.get_secret_value(),
    )


async def _embed(texts: list[str], prefix: str) -> list[list[float]]:
    payload = [prefix + t for t in texts] if prefix else list(texts)
    resp = await _client().embeddings.create(model=settings.embed_model, input=payload)
    return [d.embedding for d in resp.data]


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """ナレッジ側(索引に入れる文書)を埋め込む。"""
    return await _embed(texts, settings.embed_doc_prefix)


async def embed_query(text: str) -> list[float]:
    """検索する質問を埋め込む。"""
    return (await _embed([text], settings.embed_query_prefix))[0]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """**文書側の別名。** 03 章から使われている名前なので残しているが、
    新しい呼び出しでは用途がはっきりする embed_documents を使うこと。
    """
    return await embed_documents(texts)
