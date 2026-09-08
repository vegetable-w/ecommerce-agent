"""08 章の統一実行エンジン。すべてのツール呼び出しが通る唯一の実行経路。

パイプライン:
ツールの引き当て → JSON Schema 検証 → 権限ゲート
→ 実行(timeout / retry) → 切り分け → 整形 + 監査。

失敗や「見つからなかった」をモデルへ隠さずに返す。
監査の書き込みに失敗しても log に残すだけで、ツールの実行は絶対に止めない。
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from langchain_core.messages import ToolMessage

from app.config import settings
from app.db import repository
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

# 監査ログの status。tool_audit_logs.status の
# ENUM('success','failed','timeout','validation_blocked','permission_denied') と 1 対 1。
# 日本語の表示名は app/core/labels.py の TOOL_AUDIT_STATUS だけが持つ
# (DB / ORM / ツール引数は英語識別子、という 02 章からの取り決め)。
# 文字列を各所へ散らすと綴り違いが ENUM 違反として実行時にしか出ないので、ここへ集める。
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_VALIDATION_BLOCKED = "validation_blocked"
STATUS_PERMISSION_DENIED = "permission_denied"
ALL_STATUSES: frozenset[str] = frozenset(
    {STATUS_SUCCESS, STATUS_FAILED, STATUS_TIMEOUT, STATUS_VALIDATION_BLOCKED, STATUS_PERMISSION_DENIED}
)

# 再試行する価値のある一過性の失敗: timeout とネットワークの揺れ。
# 業務エラーや ToolException は同じ引数で何度呼んでも同じ結果なので含めない。
# TimeoutError は Python 3.11 以降 asyncio.TimeoutError と同一のオブジェクトなので
# 片方だけ書けば asyncio.wait_for の timeout も捕まる(両方並べても重複するだけ)。
# httpx の例外は MCP ツール(Streamable HTTP)の経路で届く。
_TRANSIENT = (TimeoutError, ConnectionError, httpx.TransportError)
# httpx.TimeoutException は TransportError の系統で、組み込みの TimeoutError は継承しない。
# 「timeout だったのか、それ以外の通信断だったのか」を監査で区別するために別途見る。
_TIMEOUTS = (TimeoutError, httpx.TimeoutException)
_SUMMARY_LIMIT = 500


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage
    status: str               # ALL_STATUSES のいずれか(DB の ENUM と同じ英語識別子)
    retry_count: int = 0
    duration_ms: int = 0


def validate_args(spec: ToolSpec, args: dict) -> str | None:
    """モデルが生成した引数を JSON Schema で検証し、人が読めるエラー文を返す。None は合格。

    agent_tools も、create_ticket の引数が揃っていて確認の interrupt を出すべきか
    判断するときにこの関数を再利用する。
    """
    err = best_match(Draft202012Validator(spec.json_schema).iter_errors(args))
    if err is None:
        return None
    where = f"（フィールド {err.json_path}）" if err.json_path != "$" else ""
    return f"{err.message}{where}"


def _timeout_of(spec: ToolSpec) -> float:
    if spec.timeout is not None:
        return spec.timeout
    return settings.mcp_tool_timeout if spec.source == "mcp" else settings.tool_default_timeout


def _summarize(content: str) -> str:
    return content if len(content) <= _SUMMARY_LIMIT else content[:_SUMMARY_LIMIT] + "…（省略）"


def _format_content(spec: ToolSpec, result) -> str:
    """結果の整形。

    MCP のツールは JSON の文字列を返すことがあるので、まず dict へ戻す。
    format_result があれば、必要なフィールドの選択と内部 enum の翻訳をそこへ任せる。
    最後に ensure_ascii=False で直列化し、日本語を escape しない。
    """
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result                      # ただのテキストはそのままモデルへ返す
    if spec.format_result is not None and isinstance(result, dict):
        result = spec.format_result(result)
    return json.dumps(result, ensure_ascii=False, default=str)


async def _audit(
    conversation_id,
    tool_call_id,
    name,
    spec: ToolSpec | None,
    args,
    result_summary,
    status,
    error_message,
    retry_count,
    duration_ms,
) -> None:
    try:
        await repository.insert_tool_audit(
            conversation_id=conversation_id or None,
            tool_call_id=tool_call_id or None,
            tool_name=name,
            tool_source=(spec.source if spec else "builtin"),   # 未知のツールは出所が分からないので builtin 扱い
            mcp_server=(spec.mcp_server if spec else None),
            arguments=args or None,
            result_summary=result_summary,
            status=status,
            error_message=error_message,
            retry_count=retry_count,
            duration_ms=duration_ms,
        )
    except Exception:  # noqa: BLE001 - 監査の失敗でツールの実行を止めてはいけない
        logger.exception("監査の書き込みに失敗(ツールの実行には影響なし) tool=%s status=%s", name, status)


async def execute_tool_call(
    tool_call: dict,
    conversation_id: int,
    specs: dict[str, ToolSpec],
    *,
    confirmed: bool = False,
    deny_note: str | None = None,
) -> ToolRun:
    name = tool_call.get("name") or ""
    tc_id = tool_call.get("id") or ""
    args = dict(tool_call.get("args") or {})
    started = time.monotonic()

    def _run(ok: bool, content: str, status: str, retry_count: int = 0, *, msg_status=None) -> ToolRun:
        return ToolRun(
            tool_call_id=tc_id,
            name=name,
            ok=ok,
            status=status,
            retry_count=retry_count,
            duration_ms=int((time.monotonic() - started) * 1000),
            tool_message=ToolMessage(
                content=content,
                tool_call_id=tc_id,
                name=name or "unknown",
                # ToolMessage.status は LangChain 側の "success" / "error" で、
                # 上の監査用 status とは別物。混ぜないこと。
                status=msg_status or ("success" if ok else "error"),
            ),
        )

    spec = specs.get(name)

    # ① ツールの引き当て
    if spec is None:
        run = _run(False, f"ツール実行に失敗しました：未知のツール {name}", STATUS_FAILED)
        await _audit(
            conversation_id, tc_id, name or "unknown", None, args, None,
            STATUS_FAILED, f"未知のツール {name}", 0, run.duration_ms,
        )
        return run

    # ② JSON Schema 検証。
    # ここで止めても例外は投げず、何が足りないかをモデルへ返す。
    # モデルは引数を直すか、足りない情報をユーザーへ聞き直せる。
    verr = validate_args(spec, args)
    if verr is not None:
        run = _run(
            False,
            f"パラメータ検証に失敗しました：{verr}。"
            "パラメータを修正して再度呼び出してください。"
            "不足情報がある場合は先にユーザーへ確認し、推測しないでください。",
            STATUS_VALIDATION_BLOCKED,
        )
        await _audit(
            conversation_id, tc_id, name, spec, args, None,
            STATUS_VALIDATION_BLOCKED, verr, 0, run.duration_ms,
        )
        return run

    # ③ 権限ゲート。
    # 書き込み操作には確認済みの印が要る。この印は interrupt の確認を経た
    # agent_tools だけが渡せるので、モデルの側からは迂回できない。
    if spec.permission == "write" and not confirmed:
        note = deny_note or (
            "この書き込み操作にはユーザー確認が必要です。"
            "未確認のため実行を拒否しました。ユーザーが再度明示的に要求しない限り、再実行しないでください。"
        )
        run = _run(False, f"チケット作成は実行されませんでした：{note}", STATUS_PERMISSION_DENIED)
        await _audit(
            conversation_id, tc_id, name, spec, args, None,
            STATUS_PERMISSION_DENIED, note, 0, run.duration_ms,
        )
        return run

    # ④ 実行。timeout を掛け、一過性の失敗だけ再試行する。
    # 書き込み操作は常に retry=0。timeout でも上流では実行済みかもしれず、
    # 二重に実行される方が危険だから。
    if spec.inject_conversation:
        args["conversation_id"] = conversation_id      # 検証の後に注入する(注入する引数はスキーマに含めない)

    retries = 0 if spec.permission == "write" else settings.tool_max_retries
    tool_timeout = _timeout_of(spec)
    attempt = 0

    while True:
        try:
            result = await asyncio.wait_for(spec.tool.ainvoke(args), timeout=tool_timeout)
            content = _format_content(spec, result)
            run = _run(True, content, STATUS_SUCCESS, attempt)
            await _audit(
                conversation_id, tc_id, name, spec, args, _summarize(content),
                STATUS_SUCCESS, None, attempt, run.duration_ms,
            )
            return run

        except _TRANSIENT as e:
            if attempt < retries:
                attempt += 1
                logger.warning(
                    "ツールが一過性の失敗。再試行 #%s name=%s err=%s",
                    attempt, name, type(e).__name__,
                )
                await asyncio.sleep(0.2 * attempt)
                continue

            is_timeout = isinstance(e, _TIMEOUTS)
            status = STATUS_TIMEOUT if is_timeout else STATUS_FAILED

            # ⑤ 切り分け。実際の失敗を隠さずモデルへ返す
            run = _run(
                False,
                f"ツールは一時的に利用できません："
                f"{'実行タイムアウト' if is_timeout else type(e).__name__}。"
                "しばらくしてから再試行するか、ユーザーへ正直に状況を説明してください。",
                status,
                attempt,
            )
            await _audit(
                conversation_id, tc_id, name, spec, args, None,
                status, type(e).__name__, attempt, run.duration_ms,
            )
            return run

        except Exception as e:  # noqa: BLE001 - 業務エラーや未知の例外(ToolException を含む)は再試行しない
            logger.exception("ツールの実行に失敗 name=%s", name)
            run = _run(
                False,
                f"ツールは一時的に利用できません：{type(e).__name__}。"
                "結果を捏造せず、ユーザーへ正直に説明してください。",
                STATUS_FAILED,
                attempt,
            )
            await _audit(
                conversation_id, tc_id, name, spec, args, None,
                STATUS_FAILED, f"{type(e).__name__}: {e}"[:500], attempt, run.duration_ms,
            )
            return run
