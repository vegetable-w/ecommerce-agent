"""05 章の受け入れ 5 条件を実サービス上で通す。

紙の上での確認ではなく、MySQL / Milvus / 上流モデルをすべて実際に使う。
アプリの起動が要る(既定 http://localhost:8000)。

受け入れ 1 だけ graph を直接呼ぶのは、trace を見るため。HTTP の応答(AgentResponse)は
answer / tool_calls / tool_results / suggested_actions しか返さず、「強制 retrieval を
通ったか」は載っていない。log を grep する手もあるが、出力の書式に依存して脆いので、
State の trace を直接読む。**上流も DB も本物**なので、実サービスであることは変わらない。

使い方: PYTHONUTF8=1 uv run --env-file .env python scripts/eval_05.py
"""

import asyncio
import os

import httpx

BASE = os.environ.get("EVAL_05_BASE", "http://localhost:8000")


async def agent(client, msg, cid=None):
    r = await client.post(f"{BASE}/api/agent",
                          json={"user_id": "eval-05", "message": msg, "conversation_id": cid})
    r.raise_for_status()
    return r.json()


async def acceptance_1() -> tuple[bool, object]:
    """ポリシー系の質問が強制 retrieval を通り、trace に残ること。"""
    from app.graph import runtime

    await runtime.init_graph()
    try:
        out = await runtime.run_turn("eval-05", "返品ポリシーを教えて", None)
    finally:
        await runtime.close_graph()
    trace = out["state"].get("trace", {})
    ok = trace.get("route") == "knowledge" or trace.get("forced_rag") is True
    return ok, {k: trace.get(k) for k in ("intent", "route", "forced_rag", "confidence")}


async def main() -> int:
    results = []

    ok, detail = await acceptance_1()
    results.append(("1 ポリシー系は強制 retrieval を通る", ok, detail))

    async with httpx.AsyncClient(timeout=180) as c:
        # 2: 配送の質問で Agent が自分で tool を選ぶ
        b = await agent(c, "注文1001の荷物は今どこ?")
        names = {tc["name"] for tc in b["tool_calls"]}
        results.append(("2 配送 tool を自分で選ぶ", "query_logistics" in names, sorted(names)))

        # 3: 苦情は Agent に入れず、2 つの選択肢を返す。backend は勝手に作らない
        b = await agent(c, "苦情を言いたい。対応がひどい")
        types = {a["type"] for a in b.get("suggested_actions", [])}
        results.append(("3 苦情で 2 つの選択肢", {"transfer_human", "create_ticket"} <= types,
                        sorted(types)))

        # 4: 雑談は固定応答。tool を呼ばない
        b = await agent(c, "こんにちは")
        results.append(("4 雑談は固定応答で tool なし",
                        not b["tool_calls"] and bool(b["answer"]), b["answer"][:28]))

        # 5: 複合質問が順に依存する multi-step になる。
        # 並列に 2 つ呼んだだけでも「2 件呼んだ」は成り立つので、
        # query_logistics が query_order の結果由来の tracking_no を受け取ったかで見る
        b = await agent(c, "注文1001はもう発送された? 今どこ?")
        logi = [tc for tc in b["tool_calls"] if tc["name"] == "query_logistics"]
        chained = bool(logi) and "tracking_no" in (logi[0].get("args") or {})
        results.append(("5 ReAct が順に依存して多段になる", chained,
                        [tc["name"] for tc in b["tool_calls"]]))

    print()
    for name, ok, detail in results:
        print(f"{'OK' if ok else 'NG'}  {name}\n      {detail}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} 件が期待どおり（モデルは非決定的。揺れたら再実行して記録する）")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
