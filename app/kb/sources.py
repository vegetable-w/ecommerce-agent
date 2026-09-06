"""ナレッジ構築に使う資料の一覧。

CLI(scripts/build_kb.py)と登録画面(/kb)がここを共通で読む。
片方だけが別の一覧を持つと「プレビューでは 3 ファイルだが実際の build では 4 ファイル」
というズレが起きるため、定義はこの 1 か所だけに置く。
"""

import pathlib

KB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "kb"

# ファイル名 → content_type
SOURCE_TYPES = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
    "product-spec-manual.md": "manual",
}

# mined は会話からの抽出で付く。資料ファイルには現れない
CONTENT_TYPES = ("faq", "policy", "manual", "mined")


def source_path(name: str) -> pathlib.Path:
    """資料名から実ファイルのパスを得る。一覧に無い名前は拒否する。

    登録画面から任意のファイル名を渡せてしまうと、`../` を含むパスで
    リポジトリ外を読めてしまうため、ここで一覧に対する完全一致だけを許す。
    """
    if name not in SOURCE_TYPES:
        raise ValueError(f"資料一覧にない名前: {name!r}")
    return KB_DIR / name
