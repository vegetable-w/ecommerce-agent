"""ナレッジ抽出 eval: 抽出すべきものを抽出し、個別事例を無理に抽出しないかを見る。

チャット上流を使う。実行: uv run --env-file .env python scripts/eval_mining.py
"""

import asyncio
import json
import pathlib
import sys

from app.kb import mining

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "mining_samples.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        pairs = await mining.extract_qa([s["conversation"]])
        if s.get("expect_empty"):
            ok = len(pairs) == 0
            expect = "抽出しない"
        else:
            ok = any(s["expect_question_contains"] in p.question for p in pairs)
            expect = f"{s['expect_question_contains']!r} を含む質問"
        failures += not ok
        print(f"{'OK ' if ok else 'NG '}{expect}  → {len(pairs)} 件: {[p.question for p in pairs]}")
    passed = len(samples) - failures
    print(f"\n期待どおり {passed}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
