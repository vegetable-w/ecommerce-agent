"""05 red-line smoke: いま入っている version で骨格の 3 経路が動くことを確かめる。

確かめるのは 3 つ。

1. StateGraph が組めて走ること
2. AsyncSqliteSaver が checkpointer として使えること(turn をまたいで history が続くこと)
3. streaming が **どの形** で返ってくるか

3 が本題。LangGraph は途中で stream の出力形式を増やしており、旧来の
`(mode, chunk)` tuple と、新しい `version="v2"` の StreamPart dict
({"type": ..., "data": ...}) がある。どちらで来るかで frontend へ token を
振り分ける parser が変わるので、コードを書く前にここで実測して固定する。

実行: PYTHONUTF8=1 uv run --env-file .env python scripts/smoke_langgraph.py
"""

import asyncio
from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.core.llm import get_chat_model


class S(TypedDict):
    messages: Annotated[list, add_messages]


async def call_model(state: S):
    ai = await get_chat_model(streaming=True).ainvoke(state["messages"])
    return {"messages": [ai]}


def _describe(part) -> str:
    """1 件の stream 出力が何者かを、形を仮定せずに書き出す。"""
    if isinstance(part, tuple) and len(part) == 2:
        mode, chunk = part
        if mode == "messages" and isinstance(chunk, tuple) and len(chunk) == 2:
            msg, meta = chunk
            node = meta.get("langgraph_node") if isinstance(meta, dict) else "?"
            return f"tuple/messages node={node} text={(getattr(msg, 'content', '') or '')[:16]!r}"
        if mode == "updates":
            keys = list(chunk) if isinstance(chunk, dict) else type(chunk).__name__
            return f"tuple/updates keys={keys}"
        return f"tuple/{mode} {type(chunk).__name__}"
    if isinstance(part, dict) and "type" in part:
        return f"StreamPart type={part['type']} data={type(part.get('data')).__name__}"
    return f"不明な形: {type(part).__name__} {part!r}"[:120]


async def main() -> int:
    from importlib.metadata import version
    # langgraph は __version__ を公開していないので、配布メタデータから読む
    print("  ".join(f"{p} {version(p)}" for p in
                    ("langgraph", "langgraph-checkpoint-sqlite", "aiosqlite")))

    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        b = StateGraph(S)
        b.add_node("call_model", call_model)
        b.add_edge(START, "call_model")
        b.add_edge("call_model", END)
        graph = b.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": "smoke-1"}}

        print("\n--- astream(stream_mode=['messages','updates']) の実際の形")
        seen, shapes = set(), set()
        async for part in graph.astream(
            {"messages": [HumanMessage("一文で自己紹介してください")]},
            config, stream_mode=["messages", "updates"],
        ):
            d = _describe(part)
            shapes.add(d.split(" ")[0])
            if isinstance(part, tuple):
                seen.add(part[0])
            elif isinstance(part, dict):
                seen.add(part.get("type"))
            if len(shapes) <= 6 or d.startswith("tuple/updates"):
                print("  " + d)

        print(f"\n  出た mode: {sorted(m for m in seen if m)}")
        print(f"  出力の形 : {sorted(shapes)}")

        # checkpointer が本当に turn をまたぐか(名乗った内容を次の turn で覚えているか)
        print("\n--- checkpointer が history を継続するか")
        state = await graph.aget_state(config)
        print(f"  1 turn 目の後の messages 数: {len(state.values['messages'])}")
        await graph.ainvoke({"messages": [HumanMessage("いま何と言いましたか")]}, config)
        state = await graph.aget_state(config)
        n = len(state.values["messages"])
        print(f"  2 turn 目の後の messages 数: {n}")
        ok_cp = n >= 4
        print(f"  継続している: {ok_cp}")

        ok = {"messages", "updates"} <= seen and ok_cp
        print(f"\n{'OK' if ok else 'NG'}: 3 経路とも動作")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
