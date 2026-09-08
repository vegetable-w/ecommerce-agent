"""アプリのログ設定。**呼ばれるまでアプリのログは 1 行も出ない。**

FastAPI も uvicorn も、アプリ自身の logger には何もしてくれない。uvicorn が
設定するのは uvicorn.* の logger だけで、`app.*` は root(既定 WARNING)に落ちる。
つまり configure() を呼ばない限り、コードのあちこちにある logger.info は
どこにも出ない。07 章の model_ctx(モデルへ渡した文脈)と要約の進行ログが
まさにそれで、単体テストは caplog が明示的に水準を下げるので緑のまま通る。

出力先はコンソールとファイルの両方。受け入れ検証では `grep model_ctx log/app.log`
のようにファイルを後から読む必要があり、コンソールだけだと流れて消える。
"""

import logging
import logging.handlers
from pathlib import Path

from app.config import settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# 1 ファイル 10MB を 3 世代。長い会話を回すと model_ctx だけで 1 ターン十数行になる。
_MAX_BYTES = 10 * 1024 * 1024
_BACKUPS = 3

_configured = False


def configure() -> None:
    """`app.*` の logger を INFO でコンソールとファイルへ流す。二度目以降は何もしない。

    root ではなく `app` にだけ handler を付けるのは、root へ付けると
    依存ライブラリ(httpx, asyncmy, pymilvus)の INFO まで一緒に溢れ出し、
    自分のログが埋まるため。

    **ファイルを開けなくてもアプリは起動させる。** ログの置き場所が無いことは
    サービスを止める理由にならない(コンソールには出続ける)。逆にここで落とすと、
    書き込めないディレクトリを渡しただけでチャットが上がらなくなる。
    """
    global _configured
    if _configured:
        return

    logger = logging.getLogger("app")
    logger.setLevel(settings.log_level.upper())
    # **propagate は切らない。** 二重出力を嫌って切りたくなるが、uvicorn が handler を
    # 付けるのは uvicorn.* だけで root には付けないので、渡しても出力は増えない。
    # 逆に切ると、root 側の handler で記録を拾う仕組み(pytest の caplog がそれ)が
    # このプロセス全体で効かなくなり、一度でも configure が走ったテストより後ろの
    # caplog テストが黙って空になる。

    fmt = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        path = Path(settings.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # encoding を明示する。この環境のコンソールは cp932 で、既定のままだと
        # 日本語の要約を書いた瞬間に UnicodeEncodeError でログが落ちる。
        f = logging.handlers.RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8")
        f.setFormatter(fmt)
        logger.addHandler(f)
    except OSError:
        logger.warning("ログファイルを開けないのでコンソールにだけ出力します path=%s",
                       settings.log_file)

    _configured = True
