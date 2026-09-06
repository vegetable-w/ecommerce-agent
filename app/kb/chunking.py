"""ナレッジ文書を chunk に切り分けるためのプリミティブ。

ここは純粋なテキスト処理のみで、DB にも Milvus にも埋め込み API にも触れない。
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
# 日本語には語境界がないため、段落 → 改行 → 句読点 → 一文字ずつ の順で分割する
JA_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "、", " ", ""]


def split_sections(md: str) -> list[Document]:
    """Markdown を見出し階層で節に分ける。見出し行自体は本文から除かれ metadata に入る。

    見出しが 1 つも無い文書では metadata が空の Document が 1 つだけ返る。また
    最初の見出しより前に本文があると、それも metadata 空の Document になる。
    """
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    return splitter.split_text(md)


def recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    """長い本文を chunk_size 以下の断片に切る。

    keep_separator="end": langchain の既定は True(= "start")で、区切り文字を**次の断片の
    先頭**に付ける。日本語では区切りが「。」なので、既定のままだと 2 つ目以降の断片が
    ことごとく「。〜」で始まってしまう(実測: 60 文のテキストで 59/60 が該当)。句点は
    それが終わらせた文の末尾に残すのが正しいので "end" を指定する。

    セパレータ末尾の "" は「どの区切りでも収まらないときは一文字単位で切る」という
    最終手段で、これがあるおかげで chunk_size 超過が起きない(区切りの無い長文や
    chunk_size より長い一文でも超えないことをランダム入力で確認済み)。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=JA_SEPARATORS, is_separator_regex=False, length_function=len,
        keep_separator="end",
    )
    return splitter.split_text(text)


_SENT_RE = re.compile(r"[^。！？!?…\n]*[。！？!?…\n]|[^。！？!?…\n]+$")


def _split_sentences(text: str) -> list[str]:
    """文末記号(改行を含む)ごとに区切る。記号は直前の文に付けたまま残す。"""
    return [m for m in _SENT_RE.findall(text) if m]


def _trailing_sentences(text: str, max_chars: int) -> str:
    """text 末尾から**完全な文**を何文か取り、合計が max_chars を超えないようにする。
    1 文が単独で超える場合はその 1 文をまるごと採用する(「半端な文を残さない」を優先)。"""
    out: list[str] = []
    total = 0
    for s in reversed(_split_sentences(text)):
        if out and total + len(s) > max_chars:
            break
        out.insert(0, s)
        total += len(s)
    return "".join(out)


def apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]:
    """各チャンクの先頭に、直前チャンク末尾の完全な文を重なりとして付ける。

    文字数で機械的に切ると前置きが文の途中から始まり、埋め込みにも表示にも半端な断片が
    混じる。重なりは必ず文単位で取る。
    """
    if not chunks:
        return []
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        ov = _trailing_sentences(chunks[i - 1], overlap)
        out.append(ov + chunks[i] if ov else chunks[i])
    return out
