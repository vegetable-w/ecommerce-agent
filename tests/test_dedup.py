from dataclasses import dataclass

from app.kb import dedup


@dataclass
class Item:
    question: str


def test_normalize_strips_punct_keeps_japanese():
    assert dedup.normalize_question(" 送料は、いくらですか? ") == "送料はいくらですか"


def test_normalize_folds_fullwidth_digits():
    # 日本語入力では全角数字が出やすい。揃えないと重複排除をすり抜ける
    assert dedup.normalize_question("１０日以内") == dedup.normalize_question("10日以内")


def test_dedupe_within_batch_and_against_existing():
    items = [Item("送料はいくらですか"), Item("送料はいくらですか?"),
             Item("返品はできますか"), Item("送料はどう計算されますか")]
    kept, discarded = dedup.dedupe(items, existing_questions=["送料はどう計算されますか"])
    kept_q = [i.question for i in kept]
    assert "送料はいくらですか" in kept_q
    assert "返品はできますか" in kept_q
    assert "送料はどう計算されますか" not in kept_q
    assert len(kept) == 2 and len(discarded) == 2
