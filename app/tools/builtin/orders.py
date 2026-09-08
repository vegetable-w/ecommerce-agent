"""注文・商品の照会ツール。

戻り値の中身は app/tools/business.py(mock のデータ源)が作る。ここは
ツールとしての名前・説明・引数の形と、レジストリへの登録だけを持つ。
"""
import random

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools import registry
from app.tools.business import order_snapshot


class OrderInput(BaseModel):
    order_id: str = Field(description="注文番号。例: 1001")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名またはキーワード。例: キャットフード")


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """注文のステータス、金額、注文日時、商品名を確認する。ユーザーが特定の注文について質問した場合に使用する。"""
    return order_snapshot(order_id)


# 取り扱い商品カタログ。正は data/kb/product-spec-manual.md(この店が説明している商品)。
# ここに写しを置くのは、query_product を deterministic かつ依存なしに保つため
# (ナレッジベースを引くと mock の単体テストに Milvus と埋め込み API が必要になる)。
# 写しが原本とずれないことは tests/test_tools_mock.py が原本を parse して機械的に検証する。
PRODUCT_CATALOGUE = {
    "EC-RV100": "ロボット掃除機 Standard",
    "EC-RV200": "ロボット掃除機 Pro",
    "EC-RV300": "ロボット掃除機 Max",
    "EC-AP20": "空気清浄機 Compact",
    "EC-AP40": "空気清浄機 Plus",
    "EC-EA10": "ワイヤレスイヤホン Lite",
    "EC-EA30": "ワイヤレスイヤホン Pro",
    "EC-KB20": "メカニカルキーボード",
    "EC-CM10": "ドリップコーヒーメーカー Basic",
    "EC-CM30": "全自動コーヒーメーカー Pro",
    "EC-HD20": "ヘアドライヤー Ionic",
    "EC-CH10": "オフィスチェア Standard",
    "EC-CH30": "オフィスチェア Ergo",
    "EC-PW20": "自動ペット給水器",
    "EC-PCAM1": "ペット見守りカメラ",
}
_MATCH_MIN = 3  # 「キーボード」のような品目名で当てるための最小一致長


def _find_product(query: str) -> tuple[str, str] | None:
    """問い合わせ文からカタログの商品を 1 件特定する。見つからなければ None。

    完全一致ではなく部分一致にするのは、ユーザーが「静かなキーボード」のように
    品目名だけで尋ねるため。型番が含まれていればそれを最優先する。
    """
    q = query.upper()
    for code in PRODUCT_CATALOGUE:
        if code in q:
            return code, PRODUCT_CATALOGUE[code]
    best = None
    for code, name in PRODUCT_CATALOGUE.items():
        for size in range(len(name), _MATCH_MIN - 1, -1):
            if best and size <= best[0]:
                break
            for i in range(len(name) - size + 1):
                if name[i:i + size] in query:
                    best = (size, code, name)
                    break
            if best and best[0] == size:
                break
    return (best[1], best[2]) if best else None


@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """商品の価格と在庫数を確認する。ユーザーが在庫の有無や価格を尋ねた場合に使用する。
    仕様・型番・機能そのものを聞かれた場合は query_faq を使う。"""
    # 取り扱いのない商品にも値段と在庫を答えてしまうのは mock の粗さではなく、
    # ユーザーへ嘘をつく挙動。カタログに無ければ在庫も価格も出さない。
    found = _find_product(product_name)
    if found is None:
        return {
            "product_name": product_name,
            "found": False,
            "message": f"「{product_name}」の取り扱いを確認できませんでした",
        }
    code, name = found
    rng = random.Random(f"product:{code}")
    return {
        "product_name": name,
        "model_number": code,
        "found": True,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
    }


registry.register(registry.spec_from_langchain_tool(query_order, source="builtin"))
registry.register(registry.spec_from_langchain_tool(query_product, source="builtin"))
