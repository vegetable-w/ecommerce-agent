from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会話ID。同一会話では複数ターンで再利用する")
    message: str = Field(min_length=1, description="ユーザーの今回のメッセージ")
