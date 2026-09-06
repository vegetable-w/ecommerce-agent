"""/api/agent と /api/agent/stream の入出力スキーマ。"""

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    user_id: str = Field(min_length=1, description="ユーザー識別子。会話の作成・帰属に使用")
    message: str = Field(min_length=1, description="ユーザーの今回のメッセージ")
    conversation_id: int | None = Field(
        default=None, description="会話を継続する場合に指定。空なら新規作成し、done フレームで返す"
    )


class ToolCallView(BaseModel):
    id: str
    name: str
    args: dict


class ToolResultView(BaseModel):
    tool_call_id: str
    name: str
    ok: bool
    content: str


class AgentResponse(BaseModel):
    conversation_id: int
    answer: str
    tool_calls: list[ToolCallView]
    tool_results: list[ToolResultView]
