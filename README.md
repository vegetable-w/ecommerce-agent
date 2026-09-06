# EC カスタマーサポート バックエンド

日本語 EC サイトのアフターサービス向けバックエンド。チャット応答(ストリーミング)と、問い合わせ文からのチケット情報抽出の2機能を提供する。

## 構成

- `POST /api/chat` — SSE ストリーミング応答。フレームは `data: {"delta": "..."}` 、終端は `data: [DONE]`。上流エラー時は `event: error` の後に `data: [DONE]` が続く。セッション単位で会話履歴を保持する(`session_id` で識別)。
- `POST /api/extract` — 問い合わせ本文から `AfterSalesTicket` を JSON で返す。`order_id`(nullable)、`request_type`(返金/交換/修理/苦情/その他)、`expected_solution`(必須の一文要約)。上流障害時は 502、モデル応答の解析に失敗した場合は 500(`抽出結果の解析に失敗しました`)を返す。

## 起動

上流(OpenAI 互換 API)の接続情報は `.env` に設定する(`.env.example` を参照)。`.env` は絶対にコミットしないこと。

```bash
make dev
```

`make` が使えない環境(PATH に無い等)では、同等のコマンドを直接実行する。

```bash
uv run --env-file .env uvicorn app.main:app --port 8000
```

**Windows 利用者への注意:** このプロジェクトのシェルスクリプト・評価スクリプトは Git Bash から実行することを前提にしている。PowerShell / cmd.exe からは動作を保証しない。

## テスト(ユニットテスト、モック使用)

```bash
make test
```

`make` が無い場合:

```bash
uv run pytest -v
```

## 受け入れ確認(実モデル呼び出し)

以下はいずれも実際の上流モデルを呼び出すため、事前に `make dev`(または上記の直接コマンド)でサーバーを起動しておくこと。

### 1. ラベル付きサンプルによる抽出精度の確認

`tests/data/extract_samples.json` に定義した5件のラベル付きサンプルを `/api/extract` に投入し、`order_id` と `request_type` を期待値と照合する(`expected_solution` は自然さを目視確認する)。

```bash
make eval
```

`make` が無い場合:

```bash
uv run --env-file .env python scripts/eval_extract.py
```

期待結果: `5/5 合格`。基準に届かない場合は `app/core/prompts.py` の `EXTRACT_SYSTEM` を調整して再実行する(プロンプトの調整が主な対処手段であり、ラベル付きサンプル自体は書き換えない)。

### 2. 2ターン対話によるコンテキスト保持の確認

`/api/chat` に対して同一 `session_id` で2ターン連続して送信し、2ターン目の応答が1ターン目で伝えた名前と購入商品を覚えているかを確認する。

```bash
bash scripts/demo_chat.sh
```

(実行権限が反映されない場合は `bash scripts/demo_chat.sh` のように明示的に bash 経由で実行する。)

期待結果: 1ターン目の応答が `data: {"delta": ...}` フレームとして少しずつ届き、2ターン目の応答で1ターン目に伝えた名前と購入商品(自動猫トイレ)を正しく参照していること。

**Windows/Git Bash 上の既知の注意点:** 日本語を含む JSON ペイロードを `curl` のコマンドライン引数にそのまま渡すと、MSYS がネイティブの `curl.exe` を起動する際にコンソールのコードページ(cp932)経由で引数が変換され、UTF-8 バイト列が壊れてサーバー側で `There was an error parsing the body` になることがある。`scripts/demo_chat.sh` はこれを避けるため、ペイロードを一旦 UTF-8 のテンポラリファイルに書き出し `curl -d @file` で読み込ませている。自前で curl を叩く場合も同様の方式を推奨する。

## ディレクトリ

```
app/            アプリ本体(api / core / schemas)
tests/          ユニットテスト、ラベル付きサンプル(tests/data/)
scripts/        評価・デモスクリプト(eval_extract.py, demo_chat.sh)
```
