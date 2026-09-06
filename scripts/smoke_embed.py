"""スモーク: 埋め込み上流(SiliconFlow の BAAI/bge-m3)へ直接接続し、疎通と次元数を確認する。

このスクリプトは 03 章の go/no-go リスクゲート。ここが通らなければベクトル検索は成立しない。
.env に EMBED_API_KEY が必要。
実行: uv run --env-file .env python scripts/smoke_embed.py
"""

import asyncio

from openai import AsyncOpenAI

from app.config import settings

EXPECTED_DIM = 1024


async def main() -> None:
    client = AsyncOpenAI(
        base_url=settings.embed_base_url,
        # SecretStr のまま渡すと SDK が str を期待して落ちるので、ここで開く
        api_key=settings.embed_api_key.get_secret_value(),
    )
    # 言い換えの 2 文。次元の確認に加えて、意味が近い 2 文の類似度が高く出ることも見ておく
    texts = ["送料はいくらですか", "配送料はどう計算されますか"]
    resp = await client.embeddings.create(model=settings.embed_model, input=texts)

    dims = [len(d.embedding) for d in resp.data]
    print(f"model={settings.embed_model}  ベクトル {len(resp.data)} 件  次元={dims}")
    assert len(resp.data) == len(texts), f"{len(texts)} 件を期待、実際は {len(resp.data)} 件"
    assert dims[0] == EXPECTED_DIM, f"次元 {EXPECTED_DIM} を期待、実際は {dims[0]}"
    assert len(set(dims)) == 1, f"次元が揃っていない: {dims}"

    a, b = resp.data[0].embedding, resp.data[1].embedding
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    cos = dot / (na * nb)
    print(f"L2ノルム: {na:.4f} / {nb:.4f}  (1.0 に近ければ正規化済み)")
    print(f"言い換え 2 文の COSINE 類似度: {cos:.4f}")

    print(f"GO: 埋め込み上流へ直接接続、次元 {EXPECTED_DIM}")


if __name__ == "__main__":
    asyncio.run(main())
