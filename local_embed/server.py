"""ローカルの埋め込み推論サービス。OpenAI 互換の /v1/embeddings だけを出す。

**接頭辞はここでは付けない。** ruri v3 のような非対称モデルの
「検索文書: 」「検索クエリ: 」は、呼び出し側(app/core/embeddings.py)が
文書用と質問用の入口を分けて付ける。サーバ側で推測すると、どちらの用途で
呼ばれたのかが分からないまま片方の接頭辞を当てることになる。

依存(torch / sentence-transformers)は本体の pyproject に入れない。
この 1 ファイルだけを別の環境で走らせる:

    uv run --with sentence-transformers --with torch --with fastapi --with uvicorn \\
        python local_embed/server.py

環境変数:
    EMBED_MODEL_ID  読み込むモデル(既定 cl-nagoya/ruri-v3-310m)
    PORT            待ち受けポート(既定 8200)
"""

import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBED_MODEL_ID", "cl-nagoya/ruri-v3-310m")
PORT = int(os.environ.get("PORT", "8200"))

app = FastAPI(title="local embeddings")
_model: SentenceTransformer | None = None


def model() -> SentenceTransformer:
    """初回の呼び出しで読み込む。起動を待たせないため。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_ID)
    return _model


class EmbedRequest(BaseModel):
    # model は OpenAI 互換のために受けるが、このサービスは 1 モデルしか持たないので使わない。
    # 取り違えに気づけるよう、応答には実際に使ったモデル ID を返す。
    model: str | None = None
    input: str | list[str]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest) -> dict:
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    # normalize_embeddings=True で単位ベクトルにする。Milvus 側は COSINE なので
    # 正規化の有無で順位は変わらないが、内積で見たい場面と値を揃えておく。
    vectors = model().encode(texts, normalize_embeddings=True)
    return {
        "object": "list",
        "model": MODEL_ID,
        "data": [
            {"object": "embedding", "index": i, "embedding": v.tolist()}
            for i, v in enumerate(vectors)
        ],
        # 本家と同じ形にするための埋め草。トークン数は数えない。
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    print(f"loading {MODEL_ID} ...", flush=True)
    model()
    dim = _model.get_sentence_embedding_dimension()
    print(f"ready: {MODEL_ID} dim={dim} port={PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
