"""現在の上流モデルが構造化された tool_calls を返せるか実環境で確認する。go/no-go リスクゲート。"""

import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model


class AddInput(BaseModel):
    a: int = Field(description="1つ目の加数")
    b: int = Field(description="2つ目の加数")


@tool(args_schema=AddInput)
def add(a: int, b: int) -> int:
    """2つの整数を加算する。"""
    return a + b


async def main() -> None:
    model = get_chat_model()  # 非ストリーミング、settings.chat_base_url へ直接接続
    bound = model.bind_tools([add])
    ai = await bound.ainvoke("ツールを使って 23 + 19 を計算してください")
    print("content:", repr(ai.content))
    print("tool_calls:", ai.tool_calls)
    if ai.tool_calls and ai.tool_calls[0]["name"] == "add":
        print("GO: 上流モデルは tool calling に対応。add を選択、args=", ai.tool_calls[0]["args"])
    else:
        print("NO-GO: 期待した tool_calls が返らない。ユーザーへ確認し、独自判断で方式を変更しない")


if __name__ == "__main__":
    asyncio.run(main())
