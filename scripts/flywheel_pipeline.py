"""CLI(再実行可能な job): 未処理の低信頼質問 → 正規化と重複判定 → レビュー待ち。

カーソルが `low_confidence_questions.matched_review_id IS NULL` なので、途中で落ちても
次回は続きから再開でき、何度回しても同じ質問が二重に積まれることはない
(app/core/flywheel.py の docstring 参照)。上流のチャットモデルを 1 行につき 1 回呼ぶので
課金される。

実行:
    PYTHONUTF8=1 uv run --env-file .env python scripts/flywheel_pipeline.py
    make flywheel

定期実行の例(30 分ごと。出力は追記して残す):
    */30 * * * * cd /path/to/ecommerce-agent && make flywheel >> log/flywheel.log 2>&1

skipped が 0 でない場合、その行はカーソルに残っている(次回の実行でやり直される)。
同じ件数が何度も skipped として出続けるなら、上流の障害かプロンプトの問題なので
log/ の warning を確認すること。
"""

import asyncio
import sys

from app.core.flywheel import process_pending


async def main() -> int:
    stats = await process_pending(limit=200)
    print(
        f"今回処理 {stats['processed']} 件: 新しい穴 {stats['created']}, "
        f"まとめた {stats['merged']}, 見送り(次回やり直す) {stats['skipped']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
