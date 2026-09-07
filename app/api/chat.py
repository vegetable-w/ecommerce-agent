"""チャット画面のストリーミング出口(SSE)。

- POST /api/chat : フロントエンドの主入口

05 章でこのエンドポイントは graph の astream 入口になった(spec §7 / D2)。
1 章の SessionStore による履歴保持はもう使わない。会話の State は checkpointer が
thread_id(= conversation_id)ごとに持つので、履歴を組み立てる責務はここには無い。

イベント → SSE フレームの変換と例外の対応付けは app/api/sse.py が持つ。06 章で
SSE の入口が 2 つになった(こちらと POST /api/actions/resume)ので、同じ変換を
2 か所に書かないための分離。何を流して何を流さないかの判断は
app/graph/runtime.py にある。
"""

from fastapi import APIRouter

from app.api import sse
from app.graph import runtime
from app.schemas.chat import ChatRequest

router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest):
    # runtime.stream_turn は属性経由で呼ぶこと(from ... import stream_turn に
    # しない)。テストの monkeypatch.setattr(runtime, "stream_turn", ...) が
    # 効かなくなり、本物の上流 LLM と本番相当の DB へ流れ落ちる。
    events = runtime.stream_turn(req.user_id, req.message, req.conversation_id)
    return sse.stream_response(events, context=f"user_id={req.user_id}")
