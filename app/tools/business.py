"""注文の mock データ源(純粋関数)。ツールの実装は app/tools/builtin/ にある。

ここに残すのは「注文の中身を決める」関数だけ。tool として公開せず、
query_order(app/tools/builtin/orders.py)と fetch_order node の両方が
同じ関数から中身を取ることで、経路によって注文が食い違わないようにする。
"""
import random
from datetime import datetime, timedelta


# 注文日の幅(何日前か)。上限は規約の 7 日をまたぐように広く取り、
# 一覧に「まだ返品できる注文」と「期限切れの注文」の両方が並ぶようにする。
_ORDER_AGE_DAYS = (0, 45)


def _days_ago(days: int) -> str:
    """今日から days 日前を注文日の書式で返す。"""
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 10:00")


def order_snapshot(order_id: str) -> dict:
    """注文番号から決定的な注文の中身を作る。**注文の内容の唯一の出所。**

    tool(query_order)と node(app/graph/nodes.py の fetch_order)で別々に組み立てると、
    同じ注文の中身が経路によって食い違う。画面に出した一覧と、選ばれた後に後段が読む
    注文が別物になると、返品可否の判断がユーザーの見ていない注文に対して下される。

    **draw の順序が値そのもの**なので、既存のキーの間に新しい draw を挟まないこと
    (02 章の golden value テストがこの順序に乗っている)。
    """
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["支払い待ち", "支払い済み", "発送済み", "配達完了"]),
        "amount": rng.randint(50, 2000),
        # 注文日は「何日前か」で決める。固定の日付にすると、時間が経つほど全注文が
        # 古くなり、規約の「受取後 7 日以内」を満たす注文が 1 件も作れなくなる。
        # 実測: 2026-07 固定にしていたため、返品可能と判断される経路にどうやっても
        # 到達できず、返金フローの主シナリオが試せなかった。
        # 幅を _ORDER_AGE_DAYS にしてあるので、一覧には新しい注文と古い注文が混ざる。
        "created_at": _days_ago(rng.randint(*_ORDER_AGE_DAYS)),
        "product": rng.choice(["自動猫トイレ", "キャットフード 5kg", "キャットタワー", "自動給水器"]),
        # 配送伝票番号。query_logistics の唯一の入口で、注文番号からは導けない
        # (導けると「注文を引いてから配送を引く」という順序が要らなくなり、
        # モデルが 2 つのツールを同時に呼べてしまう)。同じ注文には常に同じ番号を返す。
        # 既存キーの値を変えないよう、この draw は必ず末尾に置くこと。
        "tracking_no": f"JP{rng.randint(10**11, 10**12 - 1)}",
    }


# 一覧に出す注文の件数の幅。0 件だと画面に選ぶものが無く、多すぎると選ばせる意味が薄れる。
_USER_ORDER_COUNT = (2, 5)
# 注文番号は 4 桁で採番する。app/graph/nodes.py の _ORDER_ID_MIN_DIGITS と揃えてあり、
# 片方だけ変えると「一覧には出るが、ユーザーが打った文からは拾えない番号」が生まれる。
_ORDER_ID_RANGE = (1000, 9999)


def list_user_orders(user_id: str) -> list[dict]:
    """そのユーザーの注文一覧。注文番号が分からないときに画面へ出して選ばせるためのもの。

    **@tool にしていない。** モデルへ渡すと「一覧から自分で 1 件選ぶ」ができてしまい、
    ユーザーに選ばせるという 06 章の要件が崩れる。呼ぶのは fetch_order node だけ。

    同じ user_id なら常に同じ一覧を返す。各要素は order_snapshot から作るので、
    画面に出した product / status / amount と、選ばれた後に後段が読む注文の中身は
    必ず一致する。
    """
    rng = random.Random(f"orders:{user_id}")
    count = rng.randint(*_USER_ORDER_COUNT)
    # sample にするのは、同じ注文番号が一覧に 2 度出ると画面で見分けが付かないため
    numbers = rng.sample(range(_ORDER_ID_RANGE[0], _ORDER_ID_RANGE[1] + 1), count)
    orders = []
    for n in numbers:
        snap = order_snapshot(str(n))
        orders.append({k: snap[k] for k in ("order_id", "product", "status", "amount")})
    return orders
