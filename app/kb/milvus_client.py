"""Milvus Standalone の薄いラッパ。dense / BM25 / ハイブリッドの 3 経路を提供する。

03 章との違いが 2 つある。

1. **schema が変わった**。`vector` は `dense` に改名され、BM25 用の `text`(analyzer 付き)と
   Function 出力の `sparse`、それに絞り込み用の `section_path` / `content_type` / `category` が
   増えた。旧 schema の collection に対して新しいコードを走らせると、search の奥で
   `field section_path not exist` や `fieldName(sparse) not found` という原因の分かりにくい
   例外が出る(実測)。そのため ensure_collection は既存 collection の欠けているフィールドを
   先に検出して、作り直しを促す日本語のエラーで止める。
2. **collection 名を引数で受け取る**。03 章はモジュール定数 COLLECTION を monkeypatch して
   テストを隔離していたが、Standalone は 1 サーバ共有なので、呼び出し側が明示的に名前を
   渡せる方が安全で、評価用の一時 collection も扱える。
"""

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024

# 日本語向けの analyzer。built-in の japanese は存在せず(実測: unknown build-in
# analyzer type: japanese)、standard/english は助詞ごと巨大な 1 token になり BM25 が
# 機能しない。chinese は語を断片化し(稼/働、掃/除機)、lindera は型番 EC-RV300 を
# ec/rv/300 に割ってしまう。型番をそのまま保てるのは icu だけだった。
ANALYZER_PARAMS = {
    "tokenizer": "icu",
    "filter": [
        "lowercase",
        {"type": "length", "max": 40},
        {"type": "stop",
         "stop_words": ["の", "は", "が", "を", "に", "で", "と", "も", "や",
                        "。", "、", "（", "）", " ", "-"]},
    ],
}

# 検索が返すべきフィールド。ヒットの整形(_hit)と 1 対 1 で対応させる
_OUTPUT = ["question", "answer", "section_path", "content_type", "category"]
_REQUIRED_FIELDS = {"id", "dense", "text", "sparse", *_OUTPUT}

_CLIENT: MilvusClient | None = None
_ensured: set[str] = set()


def get_client(uri: str | None = None) -> MilvusClient:
    """既定はプロセス内で使い回す singleton。uri を渡すと独立した client を返す(テスト用)。"""
    global _CLIENT
    if uri is not None:
        return MilvusClient(uri=uri)
    if _CLIENT is None:
        _CLIENT = MilvusClient(uri=settings.milvus_uri)
    return _CLIENT


def _check_schema(client: MilvusClient, collection: str) -> None:
    """既存 collection が現行 schema と互換かを、フィールド名の有無で判定する。

    max_length や index の細部までは見ない。互換性を壊すのは「フィールドがあるか」で、
    そこさえ合っていれば search は通るため。ここで止めておかないと、
    `field section_path not exist` のような search 内部の例外として遅れて現れる。
    """
    names = {f["name"] for f in client.describe_collection(collection)["fields"]}
    missing = _REQUIRED_FIELDS - names
    if missing:
        raise RuntimeError(
            f"Milvus collection「{collection}」は現行の schema と互換ではありません"
            f"(不足フィールド: {', '.join(sorted(missing))})。"
            "04 章で schema が変わったため、ナレッジの再構築(make kb-reset の後に "
            "make kb-build / make kb-vectorize)が必要です。"
        )


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """冪等: 無ければ作り、あれば schema を検査してロードするだけ。

    dense(COSINE) + text(icu analyzer) + sparse(BM25 Function) + 絞り込み用の scalar。
    BM25 の Function は create_collection より**前に** schema へ足す必要がある(実測)。

    consistency_level="Strong": 既定の Bounded では upsert 直後の search が 0 件を返す
    (03 章で実測)。ナレッジ登録直後の検索テストもテストの upsert→search も
    read-after-write に依存しているので Strong で作る。

    singleton client のときだけ「検査済み」を記憶する。テストが渡す使い捨ての
    collection まで覚えると、同名を再利用したときに検査を飛ばしてしまう。
    """
    if client is _CLIENT and collection in _ensured:
        return
    if client.has_collection(collection):
        _check_schema(client, collection)
        client.load_collection(collection)
    else:
        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
        # max_length は文字数ではなく UTF-8 の**バイト数**(実測: "あ"×5 が length 15 と
        # 判定される)。日本語は 1 文字 3 バイトなので、400 文字の answer で約 1,200 バイト。
        # 実データの最大は text 1,199 / answer 1,116 / section_path 127 バイトで、
        # いずれも下の値に十分収まる。
        schema.add_field("text", DataType.VARCHAR, max_length=8192,
                         enable_analyzer=True, analyzer_params=ANALYZER_PARAMS)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("question", DataType.VARCHAR, max_length=2048)
        schema.add_field("answer", DataType.VARCHAR, max_length=8192)
        schema.add_field("section_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("content_type", DataType.VARCHAR, max_length=32)
        schema.add_field("category", DataType.VARCHAR, max_length=512)
        schema.add_function(Function(
            name="text_bm25", input_field_names=["text"],
            output_field_names=["sparse"], function_type=FunctionType.BM25,
        ))
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                               metric_type="BM25")
        client.create_collection(collection, schema=schema, index_params=index_params,
                                 consistency_level="Strong")
        client.load_collection(collection)
    if client is _CLIENT:
        _ensured.add(collection)


def upsert_vectors(client: MilvusClient, rows: list[dict], collection: str = COLLECTION) -> None:
    """row = {id, dense, text, question, answer, section_path, content_type, category}。
    sparse は BM25 Function がサーバ側で text から作るので、こちらからは渡さない。"""
    if rows:
        client.upsert(collection, rows)


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    """growing segment を封じて永続化する。

    検索可能にするためではない(consistency_level=Strong なので upsert 直後の
    search で既に当たる。実測で確認済み)。取り込みの区切りで segment を確定させ、
    再起動後も同じ状態から始められるようにするための操作。
    """
    client.flush(collection)


def _cat_expr(category: str | None) -> str:
    """category 絞り込みの boolean expression。None/空なら絞り込まない。"""
    return f'category == "{category}"' if category else ""


def _hit(h: dict) -> dict:
    e = h["entity"]
    return {"id": h["id"], "score": float(h["distance"]),
            "question": e["question"], "answer": e["answer"],
            "section_path": e["section_path"], "content_type": e["content_type"],
            "category": e["category"]}


def dense_search(client: MilvusClient, vector: list[float], top_k: int,
                 category: str | None = None, collection: str = COLLECTION) -> list[dict]:
    """意味検索の単路。score は COSINE の類似度そのもの(大きいほど近い)。

    03 章の Milvus Lite は COSINE の distance に 1−similarity を返したが、Standalone は
    similarity を返す(実測: 同一方向のベクトルで 1.0)。ここで 1− してはいけない。
    """
    res = client.search(collection, data=[vector], anns_field="dense", limit=top_k,
                        output_fields=_OUTPUT, search_params={"metric_type": "COSINE"},
                        filter=_cat_expr(category))
    return [_hit(h) for h in res[0]]


def bm25_search(client: MilvusClient, text: str, top_k: int,
                category: str | None = None, collection: str = COLLECTION) -> list[dict]:
    """全文検索の単路。クエリは文字列のまま渡す(サーバ側の Function が sparse 化する)。"""
    res = client.search(collection, data=[text], anns_field="sparse", limit=top_k,
                        output_fields=_OUTPUT, search_params={"metric_type": "BM25"},
                        filter=_cat_expr(category))
    return [_hit(h) for h in res[0]]


def hybrid_search(client: MilvusClient, vector: list[float], text: str, top_k: int,
                  recall: int = 50, category: str | None = None,
                  collection: str = COLLECTION) -> list[dict]:
    """dense と BM25 をそれぞれ recall 件ずつ引き、RRF で融合して top_k 件返す。

    score は RRF の融合スコアであり、COSINE の類似度でも BM25 のスコアでもない。
    しきい値で足切りしたい場合は、この値ではなく rerank 後のスコアを使うこと。
    """
    expr = _cat_expr(category)
    dense_req = AnnSearchRequest(data=[vector], anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=recall, expr=expr)
    sparse_req = AnnSearchRequest(data=[text], anns_field="sparse",
                                  param={"metric_type": "BM25"}, limit=recall, expr=expr)
    res = client.hybrid_search(collection, reqs=[dense_req, sparse_req], ranker=RRFRanker(),
                               limit=top_k, output_fields=_OUTPUT)
    return [_hit(h) for h in res[0]]


def count(client: MilvusClient, collection: str = COLLECTION) -> int:
    return client.query(collection, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]


def drop(client: MilvusClient, collection: str) -> None:
    """テストの後片付けと kb-reset 用。collection 名は既定値を持たない
    (本番の knowledge をうっかり消せてしまうため、呼ぶ側に必ず書かせる)。"""
    if client.has_collection(collection):
        client.drop_collection(collection)
