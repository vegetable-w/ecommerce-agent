"""/api/agent と /api/agent/stream の入出力スキーマ。"""

from pydantic import BaseModel, Field, field_validator


class AgentRequest(BaseModel):
    user_id: str = Field(min_length=1, description="ユーザー識別子。会話の作成・帰属に使用")
    message: str = Field(min_length=1, description="ユーザーの今回のメッセージ")
    conversation_id: int | None = Field(
        default=None, description="会話を継続する場合に指定。空なら新規作成し、done フレームで返す"
    )

    # min_length は OpenAPI スキーマの minLength として表出させるために残し、
    # 空白のみの値(min_length を通過してしまう)はこの validator で拒否する
    # (app/schemas/chat.py の ChatRequest と同じ規約。新しい書き方を発明しない)。
    #
    # chapter 1 より重い理由: あちらの SessionStore はインメモリでプロセスと共に消えるが、
    # こちらの履歴は MySQL に残る。空白のみの user 行を一度作ると、
    # app/core/agent.py の _build_history が user 行を無条件に
    # HumanMessage(m.content or "") として拾うため、その会話の以後すべてのターンで
    # プロンプトに混入し続け、トークン予算を永久に食う。
    @field_validator("user_id", "message")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("空白のみの値は許可されない")
        return v


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
