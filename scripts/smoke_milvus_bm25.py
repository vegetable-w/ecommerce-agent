"""スモーク: Milvus Standalone のネイティブ BM25 全文検索と hybrid_search。04 章の red line。

Milvus Lite は BM25 を持たないため Standalone が前提(03 章で移行済み)。
analyzer は日本語コーパスでの実測に基づき icu + filter を使う。
実行: uv run --env-file .env python scripts/smoke_milvus_bm25.py
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

COLL = "smoke_bm25"

# 日本語向け。built-in の japanese は存在せず(実測)、standard/english は
# 助詞ごと巨大な token になり BM25 が機能しない。chinese は語を断片化し、
# lindera は型番 EC-RV300 を rv/300 に割ってしまう。icu が型番を保ったまま
# 「稼働」「吸引力」を 1 語に保てる唯一の候補だった。
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


def main() -> None:
    client = MilvusClient(uri=settings.milvus_uri)
    if client.has_collection(COLL):
        client.drop_collection(COLL)

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("text", DataType.VARCHAR, max_length=2048,
                     enable_analyzer=True, analyzer_params=ANALYZER_PARAMS)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    # BM25 Function は create_collection より前に schema へ追加する必要がある
    schema.add_function(Function(
        name="text_bm25", input_field_names=["text"],
        output_field_names=["sparse"], function_type=FunctionType.BM25,
    ))

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(COLL, schema=schema, index_params=index_params,
                             consistency_level="Strong")

    client.insert(COLL, [
        {"id": 1, "dense": [0.1, 0.2, 0.3, 0.4],
         "text": "送料 1回の注文金額が3,000円以上の場合は送料無料 3,000円未満は500円"},
        {"id": 2, "dense": [0.2, 0.1, 0.4, 0.3],
         "text": "ロボット掃除機 Max 型番 EC-RV300 稼働時間 約210分 吸引力 8,000Pa"},
    ])
    client.load_collection(COLL)

    # BM25 単路: 型番のキーワードは 2 行目に当たること(受け入れ基準 2 の根拠)
    res = client.search(COLL, data=["EC-RV300 の稼働時間"], anns_field="sparse",
                        limit=2, output_fields=["text"], search_params={"metric_type": "BM25"})
    print("BM25:", [(h["id"], round(float(h["distance"]), 3)) for h in res[0]])
    assert res[0][0]["id"] == 2, "型番のキーワードは 2 行目に当たるべき"

    # hybrid RRF
    dense_req = AnnSearchRequest(data=[[0.1, 0.2, 0.3, 0.4]], anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=2)
    sparse_req = AnnSearchRequest(data=["送料"], anns_field="sparse",
                                  param={"metric_type": "BM25"}, limit=2)
    hres = client.hybrid_search(COLL, reqs=[dense_req, sparse_req], ranker=RRFRanker(),
                                limit=2, output_fields=["text"])
    print("hybrid:", [(h["id"], round(float(h["distance"]), 3)) for h in hres[0]])
    assert len(hres[0]) >= 1, "hybrid_search は最低 1 件返すべき"

    client.drop_collection(COLL)
    print("GO: Milvus Standalone の BM25 + hybrid_search は動作する")


if __name__ == "__main__":
    main()
