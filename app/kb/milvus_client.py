from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024


def get_client(uri: str | None = None) -> MilvusClient:
    return MilvusClient(uri=uri or settings.milvus_uri)


def ensure_collection(client: MilvusClient) -> None:
    """冪等: 既存ならロードのみ。無ければ schema で作成 + AUTOINDEX/COSINE + ロード。

    主キー id = MySQL knowledge_chunks.id(auto_id=False、こちらで採番)。
    これにより再実行時に id 単位の upsert となり、自然に冪等になる。

    consistency_level="Strong": standalone の既定(Bounded)では upsert 直後の search が
    0 件を返す(実測確認済み)。ナレッジ登録の直後に検索テストで確認する受け入れ 3 と、
    テストの upsert→search が間欠的に落ちるため Strong にする。数千 chunk 規模では
    latency のコストは無視できる。"""
    if client.has_collection(COLLECTION):
        client.load_collection(COLLECTION)
        return
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema("question", DataType.VARCHAR, max_length=2048),
        FieldSchema("answer", DataType.VARCHAR, max_length=8192),
    ])
    client.create_collection(COLLECTION, schema=schema, consistency_level="Strong")
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_index(COLLECTION, index_params)
    client.load_collection(COLLECTION)


def upsert_vectors(client: MilvusClient, rows: list[dict]) -> None:
    if rows:
        client.upsert(COLLECTION, rows)


def search(client: MilvusClient, vector: list[float], top_k: int) -> list[dict]:
    res = client.search(
        COLLECTION, data=[vector], limit=top_k,
        output_fields=["question", "answer"],
        search_params={"metric_type": "COSINE"},
    )
    return [
        {"id": h["id"], "score": float(h["distance"]),
         "question": h["entity"]["question"], "answer": h["entity"]["answer"]}
        for h in res[0]
    ]


def count(client: MilvusClient) -> int:
    return client.query(COLLECTION, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]


def drop_collection(client: MilvusClient) -> None:
    """テストの teardown と kb-reset ジョブ用。通常の読み書き経路からは呼ばない。"""
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
