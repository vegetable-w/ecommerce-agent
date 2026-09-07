"""検索クエリ展開のラベル付き評価。件数・重複・語の捏造・重要語の保持を機械的に見る。上流の呼び出しが必要。

展開は純粋なプロンプトの出力なので、単体テストで固定できるのは契約(ちょうど 3 件、
重複除去、失敗時の縮退)だけで、**中身が使える展開かどうか**はここで測る。

見るのは 4 点。

- ちょうど 3 件返るか
- 3 件が互いに異なるか(同じクエリを 3 回投げても検索結果は 1 通りしか増えない)
- **元の質問に無い型番や数値を作っていないか**。数字と EC- 型番を機械的に照合する。
  「30日以内」「未開封」のような条件を勝手に足されると、それが元の質問に無い前提として
  検索へ入り、ユーザーに当てはまらない規約を引いてくる
- 重要な語(商品名・型番)が全クエリで保たれているか

実行: PYTHONUTF8=1 PYTHONPATH=. uv run --env-file .env python scripts/eval_expand.py
"""
import asyncio
import re

from app.core.query_understanding import expand_queries

_DIGITS = re.compile(r"[0-9]+")
_MODEL = re.compile(r"EC-[A-Za-z0-9]+", re.IGNORECASE)

# (質問, 各クエリに残っていてほしい重要語)
# 重要語は**対象を指す語**(商品名・型番・カテゴリ)にする。プロンプトの規則 1 が保持を
# 求めているのはそこで、「返品」「交換」のような動作の語まで 3 件全部に要求すると、
# 観点を変えた 3 件目(例:「注文違い 商品 返品期限」)を規則違反と数えてしまう。
# 観点が違うことこそ展開の目的なので、その数え方は測る対象を取り違えている。
# 返金・アフターサービス系だけを並べる。expand_queries を使うのはその経路だけで、
# 単純な FAQ では使わない(1 回で当たる質問に 3 回検索しても上流を 3 回叩くだけ)。
CASES = [
    ("Bluetoothイヤホンは返品できますか", ["イヤホン"]),
    ("注文1001の商品に品質問題があります。返金できますか", ["1001", "返金"]),
    ("スマート家電が壊れました。保証を使えますか", ["保証"]),
    ("EC-RV300 を開封してしまいましたが返品できますか", ["EC-RV300"]),
    ("ロボット掃除機の返金はいつ口座に入りますか", ["返金"]),
    ("届いた商品が注文と違いました。交換してもらえますか", ["商品"]),
    ("修理に出した商品の進捗はどこで確認できますか", ["修理"]),
    ("返品の送料は誰が負担しますか", ["返品"]),
    ("保証期間が切れた後でも修理を頼めますか", ["修理"]),
    ("セール品を返品して返金してもらうことはできますか", ["セール品"]),
]


def invented(query: str, expanded: list) -> list:
    """元の質問に無い数値・型番を拾う。展開が前提を増やしていないかの確認。"""
    src_nums = set(_DIGITS.findall(query))
    src_models = {m.upper() for m in _MODEL.findall(query)}
    bad = []
    for q in expanded:
        bad += [f"{q!r} の数値 {n}" for n in _DIGITS.findall(q) if n not in src_nums]
        bad += [f"{q!r} の型番 {m}" for m in _MODEL.findall(q) if m.upper() not in src_models]
    return bad


async def main():
    ok_count = 0
    three = dup_free = keyword_kept = clean = 0
    problems = []
    for query, keys in CASES:
        got = await expand_queries(query)
        is_three = len(got) == 3
        is_uniq = len(set(got)) == len(got)
        missing = [f"{q!r} に {keys} が無い" for q in got
                   if not any(k in q for k in keys)]
        bad = invented(query, got)

        three += is_three
        dup_free += is_uniq
        keyword_kept += not missing
        clean += not bad
        ok = is_three and is_uniq and not missing and not bad
        ok_count += ok
        print(f"{'OK' if ok else 'NG'} {query!r} -> {got}(件数={len(got)})")
        for line in bad + missing:
            print(f"     {line}")
        if not ok:
            problems.append((query, got, bad, missing, is_three, is_uniq))

    n = len(CASES)
    print(f"\n全条件を満たした質問 {ok_count}/{n}(model は非決定的。揺れは再実行して記録する)")
    print(f"  ちょうど 3 件: {three}/{n}")
    print(f"  重複なし: {dup_free}/{n}")
    print(f"  重要語を全クエリで保持: {keyword_kept}/{n}")
    print(f"  元の質問に無い数値・型番を作らなかった: {clean}/{n}")
    if problems:
        print("\n外した質問:")
        for query, got, bad, missing, is_three, is_uniq in problems:
            why = []
            if not is_three:
                why.append(f"件数 {len(got)}")
            if not is_uniq:
                why.append("重複あり")
            why += bad + missing
            print(f"  {query!r} -> {got}\n      " + " / ".join(why))


if __name__ == "__main__":
    asyncio.run(main())
