"""受け入れ1 の eval: 言い換えた質問でベクトル検索し、期待する内容が返るか照合する。

事前に make kb-build && make kb-vectorize が必要(埋め込み上流も使う)。
実行: uv run --env-file .env python scripts/eval_retrieval.py
"""

import asyncio
import json
import pathlib
import sys

from app.core import retrieval

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "retrieval_samples.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        hits = await retrieval.search_knowledge(s["query"])
        top = hits[0] if hits else None
        ok = bool(top) and s["expect_answer_contains"] in top["answer"]
        failures += not ok
        mark = "OK " if ok else "NG "
        if top:
            print(f"{mark}{s['query']}  → [{top['score']:.3f}] {top['question']}")
        else:
            print(f"{mark}{s['query']}  → (ヒットなし)")
        if not ok:
            print(f"      期待: {s['expect_answer_contains']!r} を含む回答")
    passed = len(samples) - failures
    print(f"\n召回成功 {passed}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
