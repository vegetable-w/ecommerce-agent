"""06 red-line smoke: いま入っている version で interrupt / resume の形を実測する。

06 章では「注文番号が分からないとき、モデルに推測させず、画面で選ばせて続きを再開する」
という流れを作る。その土台が LangGraph の interrupt で、確かめたいのは 3 点。

  A) ainvoke が interrupt に当たったとき、戻り値のどこに情報が出るか
  B) Command(resume=v) で再開したとき、node の interrupt() が v を返すか。
     そのとき node は**先頭から再実行される**のか
  C) astream(stream_mode=["messages","updates"]) で interrupt がどの chunk に出るか
     （出ないなら aget_state で拾う必要がある）

上流は呼ばない。interrupt する node だけの小さな graph で見る。

実行: PYTHONUTF8=1 uv run --env-file .env python scripts/smoke_interrupt.py
"""

import asyncio
from importlib.metadata import version
from typing import Annotated

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class S(TypedDict):
    messages: Annotated[list, add_messages]
    picked: str
    ran: int


_calls: list[str] = []


async def ask_order(state: S):
    # 再実行されるかを見るため、interrupt の前に足跡を残す
    _calls.append("ask_order:enter")
    picked = interrupt({"type": "select_order",
                        "orders": [{"order_id": "1001"}, {"order_id": "2002"}]})
    _calls.append(f"ask_order:resumed({picked})")
    return {"picked": picked}


async def confirm(state: S):
    _calls.append("confirm")
    return {"messages": [("ai", f"注文 {state['picked']} を選びました")]}


def _build(cp):
    b = StateGraph(S)
    b.add_node("ask_order", ask_order)
    b.add_node("confirm", confirm)
    b.add_edge(START, "ask_order")
    b.add_edge("ask_order", "confirm")
    b.add_edge("confirm", END)
    return b.compile(checkpointer=cp)


async def main() -> int:
    print("langgraph", version("langgraph"))
    ok = True

    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        graph = _build(cp)

        # --- A) 非ストリーミングで interrupt に当たる
        cfg = {"configurable": {"thread_id": "smoke-int-1"}}
        _calls.clear()
        out = await graph.ainvoke({"messages": [("human", "返金したい")]}, cfg)
        print("\nA) ainvoke の戻り値のキー:", sorted(out.keys()))
        itr = out.get("__interrupt__")
        print("   __interrupt__ =", itr)
        if itr:
            first = itr[0]
            print("   要素の型:", type(first).__name__,
                  "/ .value =", getattr(first, "value", "(value 属性なし)"))
        else:
            ok = False
            print("   !! __interrupt__ が無い。aget_state で拾う必要がある")
            st = await graph.aget_state(cfg)
            print("   aget_state().next =", st.next, "/ .interrupts =",
                  getattr(st, "interrupts", "(属性なし)"))

        # --- B) resume
        out2 = await graph.ainvoke(Command(resume="1001"), cfg)
        print("\nB) resume 後 picked =", out2.get("picked"))
        print("   messages =", [getattr(m, "content", m) for m in out2["messages"]])
        print("   node の足跡 =", _calls)
        print("   → node は先頭から再実行されるか:",
              _calls.count("ask_order:enter") >= 2)
        if out2.get("picked") != "1001":
            ok = False
            print("   !! interrupt() が resume の値を返していない")

        # --- C) streaming で interrupt がどこに出るか
        cfg2 = {"configurable": {"thread_id": "smoke-int-2"}}
        _calls.clear()
        print("\nC) astream(['messages','updates']) の chunk:")
        seen_interrupt = False
        async for mode, chunk in graph.astream(
            {"messages": [("human", "返金したい")]}, cfg2,
            stream_mode=["messages", "updates"],
        ):
            keys = list(chunk) if isinstance(chunk, dict) else type(chunk).__name__
            print(f"   {mode}: {keys}")
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                seen_interrupt = True
                print("      __interrupt__ =", chunk["__interrupt__"])
        print("   → updates に __interrupt__ が出るか:", seen_interrupt)
        if not seen_interrupt:
            st = await graph.aget_state(cfg2)
            print("   出ないので aget_state を見る: .next =", st.next)
            print("      .tasks の interrupts =",
                  [getattr(t, "interrupts", None) for t in getattr(st, "tasks", ())])

        # --- C') streaming で resume できるか
        print("\nC') astream で resume:")
        async for mode, chunk in graph.astream(
            Command(resume="2002"), cfg2, stream_mode=["messages", "updates"],
        ):
            if mode == "updates":
                print("   updates:", list(chunk))
        st2 = await graph.aget_state(cfg2)
        print("   最終 picked =", st2.values.get("picked"))
        if st2.values.get("picked") != "2002":
            ok = False

    print(f"\n{'OK' if ok else 'NG'}: interrupt / resume の形を確認")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
