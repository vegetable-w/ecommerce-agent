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


def _is_key(title: str, body: str) -> int:
    head = title + body[:40]
    return int(any(t in head for t in _KEY_TERMS))


def build_chunks(
    md: str, content_type: str,
    chunk_size: int = 400, overlap: int = 60, table_max_rows: int = 10,
) -> list[Chunk]:
    """見出しごとに節へ分け、長い節はさらに分割して Chunk のリストにする。"""
    out: list[Chunk] = []
    for sec in chunking.split_sections(md):
        path = [sec.metadata[k] for k in ("h1", "h2", "h3", "h4") if sec.metadata.get(k)]
        section_path = " / ".join(path)
        title = path[-1] if path else content_type
        # category: 上位見出しパス。最上位しか無い場合はそれ自身、見出しが無ければ content_type
        category = " / ".join(path[:-1]) if len(path) > 1 else (path[0] if path else content_type)
        body = sec.page_content.strip()
        if not body:
            continue
        if chunking.is_table_block(body):
            pieces = chunking.split_table_rows(body, table_max_rows)
        else:
            base = chunking.recursive_split(body, chunk_size, chunk_overlap=0)
            pieces = chunking.apply_sentence_overlap(base, overlap)
        for piece in pieces:
            out.append(Chunk(
                category=category, questions=title, answer=piece,
                section_path=section_path, content_type=content_type,
                is_key_clause=_is_key(title, piece),
            ))
    return out
