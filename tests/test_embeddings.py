from app.config import settings
from app.core import embeddings


class _FakeEmb:
    def __init__(self, vec):
        self.embedding = vec


class _FakeResp:
    def __init__(self, vecs):
        self.data = [_FakeEmb(v) for v in vecs]


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, model, input):
        self.calls.append((model, list(input)))
        return _FakeResp([[float(i), 0.0, 1.0] for i, _ in enumerate(input)])


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


async def test_embed_texts_passes_model_and_input(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(embeddings, "_client", lambda: fake)
    out = await embeddings.embed_texts(["a", "b"])
    assert out == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    assert fake.embeddings.calls == [("BAAI/bge-m3", ["a", "b"])]


async def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "_client", lambda: _FakeClient())
    v = await embeddings.embed_query("送料")
    assert v == [0.0, 0.0, 1.0]


def test_client_unwraps_api_key_and_uses_configured_base_url():
    """他の2テストは _client() を monkeypatch で潰すので、中身を通るのはここだけ。

    api_key に SecretStr をそのまま渡す(= .get_secret_value() を外す)退行は、
    上流に本物のリクエストを投げるまで認証エラーとして現れず、発見に実 API 呼び出しの
    コストがかかる。str であることと値が一致することの両方を見ることで、
    素通し(SecretStr のまま)も str(SecretStr)(= '**********' にマスクされる)も
    ここで落ちる。AsyncOpenAI の構築自体は接続を張らないのでネットワークには出ない。
    """
    client = embeddings._client()
    assert isinstance(client.api_key, str)
    # 比較結果を先に bool へ畳んでから assert する。`assert a == b` と直接書くと、
    # 失敗時に pytest の assert introspection が両辺を展開し、生の API キーが
    # テスト出力(CI ログを含む)に平文で出てしまう。畳んでおけば表示は False だけ。
    key_matches = client.api_key == settings.embed_api_key.get_secret_value()
    assert key_matches, "api_key が SecretStr のまま、またはマスクされた値になっている"
    # openai SDK は base_url を httpx.URL に正規化して末尾に '/' を足すため、比較時に落とす
    assert str(client.base_url).rstrip("/") == settings.embed_base_url.rstrip("/")
