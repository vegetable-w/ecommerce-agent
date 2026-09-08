"""会話の一覧と履歴(読み取り専用)の出力スキーマ。

07 章の受け入れ検証のために足した。会話を切り替えて「前の会話の文脈が保たれて
いるか」を人が確かめるには、一覧と履歴の 2 つが要る(spec §「Observability」)。

response_model を置くのは、項目が OpenAPI にそのまま出て画面側との契約になるため
(app/schemas/actions.py と同じ理由)。
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ConversationItem(BaseModel):
    id: int = Field(description="会話 ID。画面はこれを conversation_id として送り返す")
    # **日本語の表示ラベル。** 英語の識別子ではない。画面側に対応表を持たせると、
    # app/core/labels.py を直したときにここと画面で別の日本語が出る。
    status: str = Field(description="会話の状態(日本語の表示ラベル)")
    preview: str = Field(description="最初のユーザー発話の抜粋。発話が無ければ空")
    # 要約の本文は載せない(数百文字あり、一覧の応答が要約で埋まる)。
    has_summary: bool = Field(description="古いターンが要約済みかどうか")
    updated_at: datetime = Field(description="最終更新時刻")


class ConversationListResponse(BaseModel):
    items: list[ConversationItem]


class MessageItem(BaseModel):
    # role は英語の識別子のまま。画面がどちら側の吹き出しにするかを決める鍵で、
    # 表示用の文字列ではない。
    role: str = Field(description="user / assistant")
    content: str = Field(description="発話の本文")
    created_at: datetime = Field(description="発話の時刻")


class MessageListResponse(BaseModel):
    items: list[MessageItem]
