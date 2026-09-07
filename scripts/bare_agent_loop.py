"""framework を使わずに Agent loop を手で書く。

LangGraph を入れる前に、Agent の中身が何なのかを目で見て確かめるための script。
結論を先に書くと、**Agent の本体は「tool 一覧を持った for loop」**である。

    model.bind_tools(tools) を呼ぶ
      → tool_calls が返ってきたら実行して、結果を messages に足してもう一度呼ぶ
      → tool_calls が返ってこなくなったら、それが最終回答

LangGraph はこの loop を消すものではなく、この loop を**より大きな graph の 1 ノード**
として管理し、その周りに決定的な経路(intent 分類・強制 retrieval・fallback・logging)を
固定するためのもの。だから「LangGraph が Agent を作る」のではなく、
「Agent は元からこの形で、LangGraph はその外側を作る」と理解するのが正しい。

使い方:
    PYTHONUTF8=1 uv run --env-file .env python scripts/bare_agent_loop.py "注文1001の配送状況は?"
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_chat_model
from app.core.prompts import AGENT_SYSTEM
from app.tools.infra import execute_tool_call
from app.tools.registry import get_all_tools


async def run_agent(query: str, max_turns: int = 6) -> str:
    model = get_chat_model().bind_tools(get_all_tools())
    messages = [SystemMessage(AGENT_SYSTEM), HumanMessage(query)]

    for step in range(1, max_turns + 1):
        ai = await model.ainvoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            print(f"[step {step}] tool call なし → 収束")
            return ai.text

        for tc in ai.tool_calls:
            print(f"[step {step}] tool {tc['name']} を実行 args={tc['args']}")
            run = await execute_tool_call(tc, conversation_id=0)
            print(f"[step {step}]   → {run.tool_message.content[:90]}")
            messages.append(run.tool_message)

    # 上限に当たったら、tool の生の結果をそのまま答えにしてはいけない。
    # production ではここが fallback への分岐になる(05 章の should_continue)。
    print(f"[上限到達] max_turns={max_turns} を使い切っても収束しなかった")
    return "(未収束)"


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "注文1001の配送状況は?"
    print("質問:", query)
    print("回答:", await run_agent(query))


if __name__ == "__main__":
    asyncio.run(main())
