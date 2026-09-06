"""ソース文書 → Chunk レコードの組み立て。"""

from app.kb.documents import Chunk, build_chunks


def test_policy_maps_heading_and_parent_path():
    md = ("# 返品ポリシー\n\n## 送料について\n\n"
          "1回のご注文が9900円以上で送料無料、未満の場合は800円を申し受けます。離島は別途。")
    chunks = build_chunks(md, content_type="policy")
    c = [c for c in chunks if c.questions == "送料について"][0]
    assert c.category == "返品ポリシー"
    assert c.section_path == "返品ポリシー / 送料について"
    assert "9900円" in c.answer
    assert c.content_type == "policy"
    assert c.is_key_clause == 1  # 「送料」を含む重要条項


def test_table_section_splits_by_rows_with_header():
    rows = "\n".join(f"| 商品{i} | {i} |" for i in range(1, 15))
    md = f"# 価格表\n\n## 価格一覧\n\n| 商品 | 価格 |\n| --- | --- |\n{rows}"
    chunks = build_chunks(md, content_type="manual", table_max_rows=5)
    price = [c for c in chunks if c.questions == "価格一覧"]
    assert len(price) >= 2
    for c in price:
        assert c.answer.startswith("| 商品 | 価格 |")


def test_returns_chunk_dataclass():
    chunks = build_chunks("# A\n\n## B\n\n本文。", content_type="faq")
    assert isinstance(chunks[0], Chunk)
