"""正規化・重複判定プロンプトを、ラベル付きサンプル 12 件で測る(プロンプトの TDD の代わり)。

プロンプトは単体テストで品質を固定できない(モデルを差し替えたら測っているのは
差し替えた側になる)。そのため実際の上流を呼び、期待値を付けたサンプルで通過率を出す。
合格ラインは 80%。

1 件を「通過」とするには次の 3 つを**すべて**満たすこと:

  1. まとめ先の判定が期待どおり(expect_match と matched_question_id が一致)。
     null を期待するケースは「似ているが別の質問」を候補として渡してある。ここが緩いと、
     重複判定が全部 match に倒れても気づけない。
  2. expect_keywords が正規化後の質問にすべて含まれる(意味の核が落ちていない)。
  3. **expect_absent が正規化後の質問に 1 つも含まれない**(口語・感情・無関係な detail が
     実際に消えている)。含有チェックだけにすると、生の質問をそのまま返す実装/プロンプトでも
     通ってしまう。正規化は「足りているか」だけでなく「余計なものが落ちているか」で測る。

DB も Milvus も使わない(サンプルは JSON、突き合わせ相手の候補もサンプルに直書き)。
書き込みは一切しない。

実行:
    PYTHONUTF8=1 uv run --env-file .env python scripts/validate_flywheel_samples.py
    make flywheel-samples
"""

import asyncio
import json
import pathlib
import sys

from app.core.flywheel import normalize_and_match

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "flywheel_samples.json"
THRESHOLD = 0.8


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    passed = 0
    for s in samples:
        try:
            r = await normalize_and_match(s["raw_question"], s["candidates"])
        except Exception as exc:
            print(f"NG {s['name']}\n   上流の呼び出しに失敗: {type(exc).__name__}: {exc}")
            continue

        reasons = []
        if r.matched_question_id != s["expect_match"]:
            reasons.append(f"まとめ先が {r.matched_question_id}(期待 {s['expect_match']})")
        missing = [k for k in s["expect_keywords"] if k not in r.normalized_question]
        if missing:
            reasons.append(f"核が落ちている: {missing} が無い")
        left = [k for k in s["expect_absent"] if k in r.normalized_question]
        if left:
            reasons.append(f"ノイズが残っている: {left}")

        passed += not reasons
        print(f"{'OK' if not reasons else 'NG'} {s['name']}")
        print(f"   {s['raw_question']}")
        print(f"   → {r.normalized_question}  [match={r.matched_question_id}]")
        if r.ai_suggested_answer:
            print(f"   回答例: {r.ai_suggested_answer}")
        for reason in reasons:
            print(f"   × {reason}")

    rate = passed / len(samples)
    print(f"\n期待どおり {passed}/{len(samples)} = {rate:.0%}(合格ライン {THRESHOLD:.0%})")
    return 0 if rate >= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
