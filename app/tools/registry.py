"""08 章のツールレジストリ。built-in(起動時に builtin/ package を scan)と
MCP(mcp_client がその都度取得)を ToolSpec という 1 つの型へ揃える。

ツールが名乗るべき 3 要素は、名前・使い方の説明・引数の JSON Schema。
権限はこちら側の WRITE_TOOLS だけを信用し、サーバの自己申告は信用しない。

**このモジュールはツールの名前を 1 つも持たない。** 名前を書いた瞬間、
「ファイルを足すだけでツールが増える」という 08 章の受け入れ条件が崩れ、
新しいツールのたびにここを直す作りへ戻る。
"""

import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 権限はこちら側のルールが唯一の根拠。書き込み系はこの名前の許可リストで決め、
# ここに無いもの(MCP のツールは全部ここに無い)は read として扱う。
# 信用できないサーバへ繋ぐ本番では、未知の書き込み操作は既定で拒否する側に倒すべき。
# 本章で自作する 2 台は照会専用なので read 扱いで足りる。
WRITE_TOOLS: set[str] = {"create_ticket"}


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict            # 引数の JSON Schema(検証とモデルへ見せる定義。注入する引数は含まない)
    tool: BaseTool
    permission: str              # "read" | "write"
    source: str                  # "builtin" | "mcp"
    mcp_server: str | None = None
    timeout: float | None = None
    inject_conversation: bool = False
    format_result: Callable[[dict], dict] | None = None


_BUILTIN: dict[str, ToolSpec] = {}
_scanned = False


def permission_for(name: str) -> str:
    return "write" if name in WRITE_TOOLS else "read"


def _json_schema_of(tool: BaseTool) -> dict:
    """スキーマの取り出し口を 1 か所に揃える。

    MCP のツールの args_schema は素の dict で届く。
    built-in は Pydantic のモデルなので、tool_call_schema(InjectedToolArg を
    除いた「モデルに見せる引数」)を JSON Schema にする。args_schema をそのまま
    使うと、注入するはずの conversation_id がモデルへ見えてしまう。
    """
    raw = getattr(tool, "args_schema", None)
    if isinstance(raw, dict):
        return raw
    tcs = getattr(tool, "tool_call_schema", None) or raw
    return tcs.model_json_schema()


def spec_from_langchain_tool(
    tool: BaseTool,
    *,
    source: str,
    mcp_server: str | None = None,
    timeout: float | None = None,
    inject_conversation: bool = False,
    format_result: Callable[[dict], dict] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        json_schema=_json_schema_of(tool),
        tool=tool,
        permission=permission_for(tool.name),
        source=source,
        mcp_server=mcp_server,
        timeout=timeout,
        inject_conversation=inject_conversation,
        format_result=format_result,
    )


def register(spec: ToolSpec) -> None:
    """先に登録した方を残す。

    後勝ちにすると、名前がぶつかったときにどちらが生きているかが import の順序で
    決まってしまい、画面から見て原因の分からない挙動になる。黙って捨てずに警告を出す。
    """
    if spec.name in _BUILTIN:
        logger.warning(
            "ツール名が重複したため後から登録された方を破棄 name=%s(先に登録された方を残す)", spec.name
        )
        return
    _BUILTIN[spec.name] = spec


def scan_builtin() -> None:
    """builtin/ package の全 module を import し、import 時の自己登録を走らせる。

    サービス起動(lifespan)から呼ぶ。何度呼んでも 1 回しか走らない。
    """
    global _scanned
    if _scanned:
        return
    _scanned = True
    from app.tools import builtin as pkg

    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{m.name}")
    logger.info("built-in ツールの登録完了:%s", sorted(_BUILTIN))


def builtin_specs() -> list[ToolSpec]:
    scan_builtin()
    return list(_BUILTIN.values())


def get_builtin_spec(name: str) -> ToolSpec | None:
    scan_builtin()
    return _BUILTIN.get(name)


# ---- 移行用の互換 API(旧 infra.py と main_agent がまだ使う。Task 5 の MCP 配線後に削除)----

NO_RETRY: set[str] = {"create_ticket", "submit_refund"}  # 書き込み系ツールは自動 retry しない
# submit_refund は agent_tools が横取りするので本来ここへ来ない。それでも入れるのは、
# 横取りを外したときに申請の意思表示が黙って 2 回流れるのを防ぐため
INJECT_CONVERSATION: set[str] = {"create_ticket"}  # 会話主キーを注入するツール
# ツールごとの timeout。**まだ誰も読んでいない**(旧 infra.execute_tool_call は引数の
# 既定値 5.0 を使う)。Task 3 の実行エンジンが ToolSpec.timeout を見るようになるまでの
# 置き場所で、ここに値を足しても現在の挙動は変わらない。
TOOL_TIMEOUTS: dict[str, float] = {"query_faq": 30.0}


def get_all_tools() -> list[BaseTool]:
    return [s.tool for s in builtin_specs()]


def get_tool(name: str) -> BaseTool | None:
    spec = get_builtin_spec(name)
    return spec.tool if spec else None
