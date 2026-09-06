"""ソース文書(Markdown)を Chunk レコードへ組み立てる。

chunking のプリミティブを組み合わせるだけで、DB にも埋め込み API にも触れない。
"""

from dataclasses import dataclass

from app.kb import chunking

_KEY_TERMS = ("返金", "返品", "交換", "送料", "配送料", "費用", "保証", "賠償", "期限", "無料")


@dataclass
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: int = 0


_LEAD_MAX = 60


def _is_key(title: str) -> int:
    """重要条項かどうかを**見出しだけ**で判定する。

    本文の冒頭も見る実装にすると、「登録は無料です」「ポイントの有効期限」のような
    偶発的な語で誤検出が激増する(実測: 適合率 5/12 = 0.42、再現率は 5/5)。
    見出しだけに絞ると同じサンプルで 5/5 = 1.00 / 1.00 になった。
    条項として効力を持つ節は、見出しそのものが「返金の時期」「送料の負担」のように
    その語を名乗るため、本文を見なくても取りこぼさない。
    """
    return int(any(t in title for t in _KEY_TERMS))


def _lead_sentence(body: str) -> str:
    """見出しの無い塊に付ける仮の見出し。本文の最初の一文を使う。

    questions は category / answer と連結して**ベクトル化テキストになる**ため、
    ここに content_type("faq" など)を入れると、検索対象の意味表現に
    無関係な語を混ぜることになる。/kb 画面にも質問として "faq" と表示されてしまう。
    """
    sentences = chunking._split_sentences(body)
    lead = (sentences[0] if sentences else body).strip()
    return lead[:_LEAD_MAX] if lead else ""


def build_chunks(
    md: str, content_type: str,
    chunk_size: int = 400, overlap: int = 60, table_max_rows: int = 10,
) -> list[Chunk]:
    """見出しごとに節へ分け、長い節はさらに分割して Chunk のリストにする。"""
    out: list[Chunk] = []
    for sec in chunking.split_sections(md):
        path = [sec.metadata[k] for k in ("h1", "h2", "h3", "h4") if sec.metadata.get(k)]
        section_path = " / ".join(path)
        body = sec.page_content.strip()
        if not body:
            continue
        # 見出しがある節はそれを、無い塊(文書冒頭の前書きなど)は本文の一文目を見出しにする
        title = path[-1] if path else _lead_sentence(body)
        # category: 上位見出しパス。最上位しか無い場合はそれ自身、見出しが無ければ content_type
        category = " / ".join(path[:-1]) if len(path) > 1 else (path[0] if path else content_type)
        if chunking.is_table_block(body):
            pieces = chunking.split_table_rows(body, table_max_rows)
        else:
            base = chunking.recursive_split(body, chunk_size, chunk_overlap=0)
            pieces = chunking.apply_sentence_overlap(base, overlap)
        for piece in pieces:
            out.append(Chunk(
                category=category, questions=title, answer=piece,
                section_path=section_path, content_type=content_type,
                is_key_clause=_is_key(title),
            ))
    return out
