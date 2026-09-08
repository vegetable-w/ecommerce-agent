"""07 章の要約プロンプトを、正解を人が付けた 3 件で測る。

要約は純粋な Prompt の仕事で、単体テストでは品質を測れない。「注文番号が残ったか」
「書かれていない数字を足していないか」は上流を実際に呼んでみないと分からないので、
この章では TDD の代わりにこの eval を置く(04 章の eval_04.py と同じ考え方)。

見るのは 4 点。key fact(注文番号 / 電話番号 / 商品)が残ること、挨拶が残らないこと、
source に無い数字を作らないこと、長さが範囲に収まること。JSON の形は
structured output が保証するのでここでは測らない。

会話本文の話者名は summarizer._render_dialog の出力に合わせて「ユーザー」「担当」で
書く。実際に要約器へ渡る形と違う形で測っても、prompt の出来は分からない。

使い方: PYTHONUTF8=1 uv run --env-file .env python scripts/eval_07.py
"""

import asyncio
import re

from app.core.summarizer import summarize_dialog

# case1: 事実の保持。注文番号 / 電話番号 / 要望は要約に必ず残る
CASE1_DIALOG = """ユーザー:こんにちは、いますか
担当:こんにちは。私は小ニャーです。どのようなご用件でしょうか。
ユーザー:注文1001のキャットタワーがまだ届きません。いつ発送されますか
担当:注文1001は杭州倉庫から発送済みで、明後日到着予定です。
ユーザー:遅いですね。電話番号は13800138000です。届く前に配送員から電話してください
担当:承知しました。配送前に13800138000へ連絡するよう記録しました。
ユーザー:ちなみに、このキャットタワーの耐荷重はいくつですか
担当:このキャットタワーの最大耐荷重は15kgです。"""
CASE1_MUST = ["1001", "13800138000", "キャットタワー"]
CASE1_BAN = ["いますか", "こんにちは。私は小ニャーです"]  # 挨拶は残さない

# case2: 捏造しない。要約に出てくる数字はすべて会話本文に存在すること
CASE2_DIALOG = """ユーザー:注文2002は返品できますか
担当:注文2002はキャットフードで、受取後3日です。7日以内の自己都合返品条件を満たしているため返品可能です。
ユーザー:では返品したいです。理由は猫が食べないからです
担当:承知しました。注文2002、理由「猫が食べない」で返品希望を記録しました。"""

# case3: 繋ぎ込み。前回の要約にしか無い事実を落とさない
CASE3_OLD = "ユーザーは注文1001(キャットタワー)の配送状況を問い合わせ、電話番号13800138000へ配送前連絡を希望。耐荷重15kgも確認済み。"
CASE3_DIALOG = """ユーザー:キャットタワーは届きましたが、支柱が1本足りません
担当:申し訳ありません。支柱のみ再発送、または注文全体の返品が可能です。どちらをご希望ですか
ユーザー:再発送してください
担当:承知しました。注文1001の支柱再発送を登録しました。3日以内に発送予定です。"""
CASE3_MUST = ["1001", "13800138000", "支柱"]  # 前回の事実(電話番号)+ 今回の事実(部品の再発送)


def check(name: str, summary: str, must=(), ban=(), src_digits: str = "") -> bool:
    """1 件分の判定。落ちた理由を全部並べてから OK / NG を返す。"""
    problems = []
    if not 20 <= len(summary) <= 250:
        problems.append(f"長さ{len(summary)}が範囲外[20,250]")
    for kw in must:
        if kw not in summary:
            problems.append(f"事実が落ちた:{kw}")
    for kw in ban:
        if kw in summary:
            problems.append(f"挨拶が残った:{kw}")
    if src_digits:
        # 4 桁以上の数字だけを見る。注文番号と電話番号がこの形で、
        # 「3日」「15kg」のような会話の中の小さい数はここでは問わない。
        src_nums = set(re.findall(r"\d{4,}", src_digits))
        for n in sorted(set(re.findall(r"\d{4,}", summary))):
            if n not in src_nums:
                problems.append(f"会話に無い数字を作った:{n}")
    ok = not problems
    print(f"{'OK' if ok else 'NG'}  {name} len={len(summary)}")
    print(f"      要約:{summary}")
    if problems:
        print(f"      問題:{problems}")
    return ok


async def main() -> int:
    results = []

    s1 = await summarize_dialog("", CASE1_DIALOG)
    results.append(check("1 事実の保持と挨拶の除去", s1,
                         must=CASE1_MUST, ban=CASE1_BAN, src_digits=CASE1_DIALOG))

    s2 = await summarize_dialog("", CASE2_DIALOG)
    results.append(check("2 捏造しない(数字は会話本文にあるものだけ)", s2,
                         must=["2002"], src_digits=CASE2_DIALOG))

    s3 = await summarize_dialog(CASE3_OLD, CASE3_DIALOG)
    results.append(check("3 繋ぎ込みで前回の事実を落とさない", s3,
                         must=CASE3_MUST, src_digits=CASE3_OLD + CASE3_DIALOG))

    print(f"\n{sum(results)}/{len(results)} 件が期待どおり"
          "（モデルは非決定的。揺れたら再実行して記録する）")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
