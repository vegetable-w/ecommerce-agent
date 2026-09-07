"""/api/chat の入力スキーマ。

05 章で /api/chat は graph のストリーミング入口になり、/api/agent とまったく同じ入力
(user_id / message / conversation_id)を取るようになった(spec §7 / D2)。同じ検証を
2 か所へ書き写すと片方だけ直す事故が起きるので、AgentRequest をそのまま継承する。
1 章の session_id はここで役目を終えた。会話の同一性は conversation_id が持つ。
"""

from app.schemas.agent import AgentRequest


class ChatRequest(AgentRequest):
    """/api/chat のリクエストボディ。中身は AgentRequest と同一。"""
