"""分割プリミティブ: 見出し分割と再帰分割。"""

from app.kb import chunking


def test_split_sections_keeps_header_path():
    md = "# アフターサービス\n\n## 返品ポリシー\n\n7日以内なら返品可能。\n\n## 送料について\n\n9900円以上で送料無料。"
    secs = chunking.split_sections(md)
    shipping = [d for d in secs if d.metadata.get("h2") == "送料について"]
    assert shipping, "「送料について」の節が切り出されること"
    assert shipping[0].metadata.get("h1") == "アフターサービス"
    assert "9900円以上" in shipping[0].page_content


def test_recursive_split_breaks_oversized():
    text = "これは本文の一文です。" * 60
    parts = chunking.recursive_split(text, chunk_size=50, chunk_overlap=0)
    assert len(parts) > 1
    assert max(len(p) for p in parts) <= 50


def test_recursive_split_does_not_start_chunk_with_punctuation():
    """句点は「直前の文の末尾」に残す。

    langchain の既定(keep_separator=True)はセパレータを**次のチャンクの先頭**に付けるため、
    このテキストでは 2 チャンク目以降が軒並み「。これは…」で始まってしまう。それでは
    Task 7 の文末揃え overlap 以前にチャンク自体が文の途中から始まることになるので、
    keep_separator="end" を指定していることをここで固定する。
    """
    parts = chunking.recursive_split("これは本文の一文です。" * 60, chunk_size=50)
    assert len(parts) > 1
    assert not [p for p in parts if p[0] in "。！？!?；;、"]
