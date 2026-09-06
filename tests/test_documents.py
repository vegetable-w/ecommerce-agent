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


def test_is_key_clause_looks_at_heading_only():
    """本文の偶発的な語で重要条項と誤判定しないこと。

    「登録は無料です」のように、条項ではない節の本文に _KEY_TERMS の語が紛れることは多い。
    本文まで見る実装では実測で適合率 5/12 まで落ちた。見出しだけを見れば 5/5 になる。
    """
    md = "# 会員について\n\n## 会員登録の方法\n\nメールアドレスをご登録ください。登録は無料です。"
    c = build_chunks(md, content_type="faq")[0]
    assert c.questions == "会員登録の方法"
    assert c.is_key_clause == 0, "本文の「無料」で誤検出してはいけない"


def test_is_key_clause_still_catches_real_clause():
    md = "# 返品ポリシー\n\n## 送料の負担\n\n配送業者はヤマト運輸です。"
    c = build_chunks(md, content_type="policy")[0]
    assert c.is_key_clause == 1, "見出しに「送料」があれば本文に語が無くても重要条項"


def test_headingless_block_uses_lead_sentence_as_question():
    """questions はベクトル化テキストの一部なので、content_type をそのまま入れない。"""
    md = "当店をご利用いただきありがとうございます。以下は各種ご案内です。"
    c = build_chunks(md, content_type="faq")[0]
    assert c.questions == "当店をご利用いただきありがとうございます。"
    assert c.questions != "faq"
    assert c.category == "faq"  # category は content_type のままでよい
