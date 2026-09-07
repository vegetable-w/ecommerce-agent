"""intent 分類のラベル付き評価。8 class の accuracy を確認する。上流の呼び出しが必要。

分類は純粋な prompt の出力なので unit test では正しさを固定できない(上流を差し替えれば
自分で書いた期待値をなぞるだけになる)。代わりにラベル付きサンプルを実際に流して
実測の accuracy を見る。境界例(アフターサービス ↔ 返金返品 など)の少数の揺れは許容し、
特定の class へ明確に偏るときだけ prompt を直す。

**サンプルの文面は prompt の few-shot と重ねないこと。** 重ねると few-shot をそのまま
なぞるだけになり、この評価が「prompt に書いた文を prompt が当てられるか」の確認に化ける。
境界の 4 組は狙って入れてあるが、言い回しは prompt 側と変えてある。

confidence も出す。分類を外した問いが「言い切って外した」のか「迷った末に外した」のかで、
prompt を直すべきか 09 章の段階的エスカレーションに任せるべきかが変わる。

実行: PYTHONUTF8=1 uv run --env-file .env python scripts/eval_intent.py
"""
import asyncio
from collections import Counter

from app.core.intent import classify

# 末尾の note は「なぜこの問いを入れたか」。境界の 4 組(返金返品 ↔ アフターサービス、
# 苦情 ↔ 配送、商品相談 ↔ 注文、雑談 ↔ その他)を必ず通す。
SAMPLES = [
    ("注文1001の荷物は今どこ?", "配送", ""),
    ("買った商品はもう発送された?", "配送", ""),
    ("一週間経っても荷物が来ません。どうなっているんでしょうか", "配送", "境界: 苦情 ↔ 配送"),
    ("注文2002は今どんな状態?", "注文", ""),
    ("先週注文した商品の金額はいくらだった?", "注文", ""),
    ("この前買ったやつ、支払った金額を教えて", "注文", "境界: 商品相談 ↔ 注文"),
    ("この商品はいくら?", "商品相談", ""),
    ("このスマート家電の使い方を教えて", "商品相談", ""),
    ("その掃除機の吸引力の仕様を知りたい", "商品相談", "境界: 商品相談 ↔ 注文"),
    ("返品したい", "返金返品", ""),
    ("返金は通常何日で反映される?", "返金返品", ""),
    ("届いた商品を返して代金を戻してもらえますか", "返金返品", "境界: 返金返品 ↔ アフターサービス"),
    ("買った機器が壊れた。保証できる?", "アフターサービス", ""),
    ("交換手続きの進捗は?", "アフターサービス", ""),
    ("故障したので点検に出したいです", "アフターサービス", "境界: 返金返品 ↔ アフターサービス"),
    ("このサービスひどすぎる。苦情を言いたい", "苦情", ""),
    ("最悪です。説明してください", "苦情", ""),
    ("前回も同じ対応でした。上の方から回答をください", "苦情", "境界: 苦情 ↔ 配送"),
    ("こんにちは", "雑談", ""),
    ("今日はいい天気ですね", "雑談", ""),
    ("いつもありがとうございます、助かってます", "雑談", "境界: 雑談 ↔ その他"),
    ("さっきの件、あれでお願いします", "その他", "境界: 雑談 ↔ その他"),
    ("うーん、どうしようかな", "その他", "境界: 雑談 ↔ その他"),
    ("とりあえず", "その他", "境界: 雑談 ↔ その他"),
]


async def main():
    passed = 0
    missed = []
    got_counts = Counter()
    for q, expect, note in SAMPLES:
        r = await classify(q)
        got, conf = r["intent"], r["confidence"]
        got_counts[got] += 1
        ok = got == expect
        passed += ok
        tail = f"  <- {note}" if note else ""
        print(f"{'OK' if ok else 'NG'} conf={conf:.2f} {q!r} -> {got} expected={expect}{tail}")
        if not ok:
            missed.append((q, expect, got, conf, note))

    print(f"\n正解 {passed}/{len(SAMPLES)}(model は非決定的。揺れは再実行して記録する)")
    print("分類の分布:", dict(got_counts))
    if missed:
        print("\n外した問い:")
        for q, expect, got, conf, note in missed:
            print(f"  {q!r} 期待={expect} 実際={got} conf={conf:.2f} {note or '(境界ではない)'}")


if __name__ == "__main__":
    asyncio.run(main())
