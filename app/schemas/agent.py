"""/api/agent と /api/chat の入出力スキーマ。

05 章で入口は 2 つになった(spec D2)。どちらも同じ graph を通し、同じ形の入力を取る。
"""

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
    # 既定を空リストにするのは、選択肢を出さないターン(大半)でも必ずこの key が
    # 存在するようにするため。呼び出し側が key の有無で分岐せずに済む。
    suggested_actions: list = Field(default_factory=list)
    # 06 章。注文が特定できないと fetch_order は interrupt で止まり、答えの無いまま
    # ここへ戻ってくる。中断の payload({"type": "select_order", "orders": [...]})を
    # そのまま載せて、何を選ばせればよいかを呼び出し側へ伝える。
    #
    # 既定を None にするのは suggested_actions と同じ理由で、中断していないターンでも
    # key を必ず出すため。**中身を型で縛らない**のは、中断の種類が増えたときに
    # payload の形も変わり、ここが変更の追随点になってしまうため
    # (何を描くかは "type" を見て画面が決める)。
    interrupt: dict | None = Field(
        default=None, description="ユーザーの選択待ちで停止した場合の payload。通常は null"
    )
