"""ツール実行基盤。

registry からツールを引き、会話主キーを注入し、タイムアウトを掛け、
一時的な失敗を retry する。例外は一切外へ漏らさず、モデルが読める
ToolMessage(status="error") を含む ToolRun として返す。

タイムアウトは attempt(試行)ごとに掛かる。デフォルトの
timeout=5.0, max_retries=2 では、全て失敗した場合の最悪ケースの壁時計時間は
おおよそ (max_retries+1) * timeout + backoff sleep の合計
= 3 * 5.0 + (0.2*1 + 0.2*2) = 15.0 + 0.6 = 15.6 秒。
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.tools import registry

logger = logging.getLogger(__name__)

# モデルに見せるメッセージには例外クラス名など内部詳細を含めない。
# (詳細は logger.exception が記録するので、運用側の調査には困らない)
_UNKNOWN_TOOL_MSG = "不明なツール {name}"
_VALIDATION_ERROR_MSG = "入力内容を確認できませんでした"
_EXECUTION_ERROR_MSG = "一時的なエラーが発生しました"


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage


def _error_run(tc_id: str, name: str, msg: str) -> ToolRun:
    return ToolRun(
        tool_call_id=tc_id,
        name=name,
        ok=False,
        tool_message=ToolMessage(
            content=f"ツール実行失敗: {msg}", tool_call_id=tc_id, name=name, status="error"
        ),
    )


async def execute_tool_call(
    tool_call: dict, conversation_id: int, timeout: float = 5.0, max_retries: int = 2
) -> ToolRun:
    # name/id が欠けた、または id が None の壊れた tool_call でも例外を外へ漏らさない
    # (このモジュールの契約: 常に ToolRun を返す)。実測で確認済み:
    # - dict の素朴な添字アクセスは KeyError で契約を破っていた(キー欠落)。
    # - `.get("id", "unknown")` だけでは id=None を防げない: langchain_core の
    #   ToolCall/ToolCallChunk は id: str | None を許容しており(OpenAI互換ゲートウェイが
    #   id を省略した tool_call チャンクをマージするとこの形になりうる)、キーは存在するが
    #   値が None なので既定値は使われず、後段の ToolMessage(tool_call_id=None) が
    #   ValidationError で落ちる。`or "unknown"` で falsy(None/空文字含む)を弾く。
    name = tool_call.get("name", "")
    tc_id = tool_call.get("id") or "unknown"
    args = dict(tool_call.get("args") or {})

    tool = registry.get_tool(name)
    if tool is None:
        return _error_run(tc_id, name, _UNKNOWN_TOOL_MSG.format(name=name))

    if name in registry.INJECT_CONVERSATION:
        args["conversation_id"] = conversation_id

    retries = 0 if name in registry.NO_RETRY else max_retries
    attempt = 0
    while True:
        try:
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
            content = json.dumps(result, ensure_ascii=False, default=str)
            return ToolRun(
                tool_call_id=tc_id,
                name=name,
                ok=True,
                tool_message=ToolMessage(content=content, tool_call_id=tc_id, name=name),
            )
        except ValidationError:
            # 引数が不正なケースは再試行しても結果が変わらない(同じ引数で同じ検証に
            # 必ず失敗する)ため、retry 予算を消費せずに即エラー確定する。
            logger.exception("ツール引数検証失敗 name=%s", name)
            return _error_run(tc_id, name, _VALIDATION_ERROR_MSG)
        except Exception:  # noqa: BLE001 - すべて捕捉してエラー結果としてモデルへ返す
            attempt += 1
            if attempt > retries:
                logger.exception("ツール実行失敗 name=%s", name)
                return _error_run(tc_id, name, _EXECUTION_ERROR_MSG)
            await asyncio.sleep(0.2 * attempt)
