"""分割プリミティブ: 文末に揃える overlap。"""

from app.kb import chunking


def test_overlap_is_whole_trailing_sentence():
    chunks = ["前半の内容です。中ほどの一文。最後の締めの文。", "次のブロック本文。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=8)
    assert out[0] == chunks[0]
    # 末尾の「最後の締めの文。」は完全な一文 → まるごと重なりの前置きになる
    assert out[1] == "最後の締めの文。次のブロック本文。"


def test_overlap_never_starts_mid_sentence():
    """overlap に収まらない長さの一文でも、途中で切らずまるごと前置きにする。

    計画時の版は `out[1].startswith("とても") or out[1] == "新ブロック。"` という
    表明だったが、これは想定される 2 つの結果を両方許すため決して落ちない。
    「文の途中から始まらない」ことを、前置きの直前の文字が文末記号であることで
    直接確かめる形に変えてある。
    """
    prev = "とても長い一文で途中に句点がなく最後にだけ句点がある文です。"
    chunks = [prev, "新ブロック。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=5)

    assert out[1].endswith(chunks[1]), "後続チャンクの本文は必ず保たれる"
    prefix = out[1][: len(out[1]) - len(chunks[1])]
    assert prev.endswith(prefix), "前置きは直前チャンクの末尾そのものであること"
    # overlap=5 に収まらないが、半端な文を残すくらいなら一文まるごとを優先する
    assert prefix != "", "収まらないからといって重なりを捨てない"
    head = prev[: len(prev) - len(prefix)]
    assert head == "" or head[-1] in "。！？!?…\n", "前置きは文の先頭から始まること"


def test_overlap_of_empty_list():
    assert chunking.apply_sentence_overlap([], overlap=8) == []
