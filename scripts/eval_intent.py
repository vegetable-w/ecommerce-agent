"""intent 分類のラベル付き評価。7 class の accuracy を確認する。上流の呼び出しが必要。

分類は純粋な prompt の出力なので unit test では正しさを固定できない(上流を差し替えれば
自分で書いた期待値をなぞるだけになる)。代わりにラベル付きサンプルを実際に流して
実測の accuracy を見る。境界例(アフターサービス ↔ 返金返品 など)の少数の揺れは許容し、
特定の class へ明確に偏るときだけ prompt を直す。

実行: PYTHONUTF8=1 uv run --env-file .env python scripts/eval_intent.py
"""
import asyncio

from app.core.intent import classify

SAMPLES = [
    ("注文1001の荷物は今どこ?", "配送"),
    ("買った商品はもう発送された?", "配送"),
    ("注文2002は今どんな状態?", "注文"),
    ("先週注文した商品の金額はいくらだった?", "注文"),
    ("この商品はいくら?", "商品相談"),
    ("このスマート家電の使い方を教えて", "商品相談"),
    ("返品したい", "返金返品"),
    ("返金は通常何日で反映される?", "返金返品"),
    ("買った機器が壊れた。保証できる?", "アフターサービス"),
    ("交換手続きの進捗は?", "アフターサービス"),
    ("このサービスひどすぎる。苦情を言いたい", "苦情"),
    ("最悪です。説明してください", "苦情"),
    ("こんにちは", "雑談"),
    ("今日はいい天気ですね", "雑談"),
]


async def main():
    passed = 0
    for q, expect in SAMPLES:
        got = await classify(q)
        ok = got == expect
        passed += ok
        print(f"{'OK' if ok else 'NG'} {q!r} -> {got} expected={expect}")
    print(f"\n正解 {passed}/{len(SAMPLES)}(model は非決定的。揺れは再実行して記録する)")


if __name__ == "__main__":
    asyncio.run(main())
