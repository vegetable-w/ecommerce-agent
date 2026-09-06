"""質問文の正規化と重複排除。

抽出したQ&Aを staging から知識ベースへ昇格させる前に、表記ゆれだけが違う
同じ質問を落とす。
"""

import re
import unicodedata

# \W は日本語(かな・カタカナ・漢字・長音符)を含まないため、空白と記号だけが落ちる
_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_question(q: str) -> str:
    # NFKC: 全角数字 → 半角、半角カナ → 全角。日本語入力では全角数字が出やすく、
    # これを揃えないと「10日以内」と「１０日以内」が別物として重複排除をすり抜ける(実測済み)。
    return _STRIP_RE.sub("", unicodedata.normalize("NFKC", q).strip().lower())


def dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]:
    seen = {normalize_question(q) for q in existing_questions}
    kept, discarded = [], []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            discarded.append(item)
        else:
            seen.add(key)
            kept.append(item)
    return kept, discarded
