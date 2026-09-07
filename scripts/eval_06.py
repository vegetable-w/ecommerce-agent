"""06 章の受け入れ 4 条件を実サービス上で通す。

MySQL / Milvus / 上流モデルをすべて実際に使う。判定は log を目で読むのではなく
**trace を読んで機械的に**行う(plan は acceptance 1 を「log で確認」としていたが、
それでは通ったかどうかが記録に残らない)。

trace は HTTP の応答に載っていないので graph を直接呼ぶ。上流も DB も本物なので
実サービスであることは変わらない(05 章の eval_05.py と同じ考え方)。

使い方: PYTHONUTF8=1 uv run --env-file .env python scripts/eval_06.py
"""

import asyncio
from datetime import datetime

from app.graph import runtime
from app.graph.nodes import resolve_answer
from app.tools.business import list_user_orders, order_snapshot


def _recent_order(user_id: str) -> tuple[str, int] | tuple[None, None]:
    """規約の期限内に収まる注文を 1 件選ぶ。無ければ (None, None)。"""
    for o in list_user_orders(user_id):
        snap = order_snapshot(o["order_id"])
        ordered = datetime.strptime(snap["created_at"], "%Y-%m-%d %H:%M")
        days = (datetime.now().date() - ordered.date()).days
        if days <= 7:
            return o["order_id"], days
    return None, None


async def acceptance_1() -> tuple[bool, object]:
    """会話をまたいで intent が動くこと。配送 → 返金返品 → 配送。

    2 turn 目は指示語だけで注文を指すので、書き下しが効いていないと分類も外れる。
    """
    a = await runtime.run_turn("eval-06", "注文1001は今どこですか", None)
    cid = a["conversation_id"]
    b = await runtime.run_turn("eval-06", "ではそれを返品したいです", cid)
    c = await runtime.run_turn("eval-06", "やっぱり今どこにありますか", cid)

    got = [t["state"].get("trace", {}) for t in (a, b, c)]
    routes = [t.get("route") for t in got]
    ok = routes[0] == "business" and routes[1] == "refund_flow" and routes[2] == "business"
    return ok, {"routes": routes,
                "intents": [t.get("intent") for t in got],
                "coref": [t.get("coref") for t in got]}


async def acceptance_3() -> tuple[bool, object]:
    """返金の subflow が順に通ること。注文の特定 → 規約検索 → 可否の判断。"""
    order_id, days = _recent_order("eval-06-refund")
    if order_id is None:
        return False, "期限内の注文が見つからない(注文日の生成を確認すること)"
    out = await runtime.run_turn(
        "eval-06-refund", f"注文{order_id}を返品して返金してほしいです", None)
    st = out["state"]
    trace = st.get("trace", {})
    acts = [a["type"] for a in st.get("suggested_actions", [])]
    ok = (trace.get("route") == "refund_flow"
          and trace.get("fetch_order", {}).get("order_id") == order_id
          and trace.get("retrieve_policy", {}).get("hits", 0) > 0
          and "refund_form" in acts)
    return ok, {"order": f"{order_id}({days}日前)",
                "fetch_order": trace.get("fetch_order"),
                "policy_hits": trace.get("retrieve_policy", {}).get("hits"),
                "actions": acts}


async def acceptance_3b() -> tuple[bool, object]:
    """期限を過ぎた注文では申請フォームを出さないこと(判断が効いていることの裏)。"""
    old = None
    for o in list_user_orders("eval-06-old"):
        snap = order_snapshot(o["order_id"])
        days = (datetime.now().date()
                - datetime.strptime(snap["created_at"], "%Y-%m-%d %H:%M").date()).days
        if days > 30:
            old = (o["order_id"], days)
            break
    if old is None:
        return False, "期限切れの注文が見つからない"
    out = await runtime.run_turn("eval-06-old", f"注文{old[0]}を返品したいです", None)
    acts = [a["type"] for a in out["state"].get("suggested_actions", [])]
    return "refund_form" not in acts, {"order": f"{old[0]}({old[1]}日前)", "actions": acts}


async def acceptance_4() -> tuple[bool, object]:
    """注文番号が無いとき、推測せずに一覧を出して止まること。"""
    out = await runtime.run_turn("eval-06-pick", "返金したいです", None)
    itr = out.get("interrupt") or {}
    orders = itr.get("orders") or []
    ok = itr.get("type") == "select_order" and len(orders) >= 2
    return ok, {"kind": itr.get("type"), "件数": len(orders),
                "先頭": orders[0] if orders else None}


async def acceptance_4b() -> tuple[bool, object]:
    """選ばれた注文で再開し、その注文について続きが進むこと。"""
    out = await runtime.run_turn("eval-06-resume", "返金したいです", None)
    cid = out["conversation_id"]
    orders = (out.get("interrupt") or {}).get("orders") or []
    if not orders:
        return False, "中断しなかった"
    picked = orders[0]["order_id"]
    done = await runtime.resume_turn(cid, picked)
    st = done["state"]
    trace = st.get("trace", {})
    ok = (trace.get("fetch_order", {}).get("order_id") == picked
          and trace.get("retrieve_policy", {}).get("hits", 0) > 0
          and bool(resolve_answer(st)))
    return ok, {"選んだ注文": picked, "fetch_order": trace.get("fetch_order"),
                "policy_hits": trace.get("retrieve_policy", {}).get("hits"),
                "回答": resolve_answer(st)[:44]}


async def main() -> int:
    await runtime.init_graph()
    results = []
    try:
        for name, fn in [
            ("1 会話をまたいで intent が動く(配送→返金→配送)", acceptance_1),
            ("3 返金 subflow が順に通る(注文→規約→判断)", acceptance_3),
            ("3b 期限切れでは申請フォームを出さない", acceptance_3b),
            ("4 注文番号が無ければ一覧を出して止まる", acceptance_4),
            ("4b 選ばれた注文で続きが進む", acceptance_4b),
        ]:
            ok, detail = await fn()
            results.append((name, ok, detail))
    finally:
        await runtime.close_graph()

    print()
    for name, ok, detail in results:
        print(f"{'OK' if ok else 'NG'}  {name}\n      {detail}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} 件が期待どおり（モデルは非決定的。揺れたら再実行して記録する）")
    print("\n2 の intent の安定性と「その他」への退避は scripts/eval_intent.py、")
    print("書き下しは scripts/eval_coref.py、クエリ展開は scripts/eval_expand.py で見る。")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
