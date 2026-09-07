"""ツールの一覧・名前引き・retry/注入ポリシーを一元管理するレジストリ。"""

from langchain_core.tools import BaseTool

from app.tools.business import (
    create_ticket,
    query_faq,
    query_logistics,
    query_order,
    query_product,
    submit_refund,
)

_ALL: list[BaseTool] = [query_order, query_product, query_logistics, query_faq,
                        create_ticket, submit_refund]
_BY_NAME: dict[str, BaseTool] = {t.name: t for t in _ALL}

NO_RETRY: set[str] = {"create_ticket", "submit_refund"}  # 書き込み系ツールは自動 retry しない
# submit_refund は agent_tools が横取りするので本来ここへ来ない。それでも入れるのは、
# 横取りを外したときに申請の意思表示が黙って 2 回流れるのを防ぐため
INJECT_CONVERSATION: set[str] = {"create_ticket"}  # 会話主キーを注入するツール


def get_all_tools() -> list[BaseTool]:
    return _ALL


def get_tool(name: str) -> BaseTool | None:
    return _BY_NAME.get(name)
