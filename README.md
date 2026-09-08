# EC カスタマーサポート バックエンド

日本語 EC サイトのアフターサービス向けバックエンド。チャット応答(ストリーミング)と、問い合わせ文からのチケット情報抽出の2機能を提供する。

## 構成

- `POST /api/chat` — フロントエンドの入口。SSE ストリーミング応答。body は `{user_id, message, conversation_id?}`。フレームは `data: {"delta": "..."}` のほか、`{"event":"tool"}` / `{"event":"citations"}` / `{"event":"actions"}` / `{"event":"done","conversation_id":N}`。終端は `data: [DONE]`。上流エラー時は `event: error`。
- `POST /api/agent` — 非ストリーミング。評価とテスト用。`answer` / `tool_calls` / `tool_results` / `suggested_actions` を返す。
- `POST /api/actions/create-ticket` — チケット作成ボタンの受け口。**ユーザーが押したときだけ** `tickets` へ書く。

どちらの入口も同じ LangGraph の graph を通る(`app/graph/`)。会話履歴は checkpointer(sqlite)が turn をまたいで保持するので、`session_id` によるインメモリ保持は 05 章で廃止した。
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

`/api/chat` に対して、1ターン目の応答で返る `conversation_id` を2ターン目に付けて送信し、2ターン目の応答が1ターン目で伝えた名前と購入商品を覚えているかを確認する。

```bash
bash scripts/demo_chat.sh
```

(実行権限が反映されない場合は `bash scripts/demo_chat.sh` のように明示的に bash 経由で実行する。)

期待結果: 1ターン目の応答が `data: {"delta": ...}` フレームとして少しずつ届き、2ターン目の応答で1ターン目に伝えた名前と購入商品(自動猫トイレ)を正しく参照していること。

**Windows/Git Bash 上の既知の注意点:** 日本語を含む JSON ペイロードを `curl` のコマンドライン引数にそのまま渡すと、MSYS がネイティブの `curl.exe` を起動する際にコンソールのコードページ(cp932)経由で引数が変換され、UTF-8 バイト列が壊れてサーバー側で `There was an error parsing the body` になることがある。`scripts/demo_chat.sh` はこれを避けるため、ペイロードを一旦 UTF-8 のテンポラリファイルに書き出し `curl -d @file` で読み込ませている。自前で curl を叩く場合も同様の方式を推奨する。

### 3. 05 章の受け入れ 5 条件（実サービス）

MySQL / Milvus / アプリをすべて起動したうえで:

```bash
make eval-05
```

5 条件をこの順で確認する。

1. ポリシー系の質問が**強制 retrieval** を通ること（trace に `route=knowledge` / `forced_rag=True`）
2. 配送の質問で Agent が自分で `query_logistics` を選ぶこと
3. 苦情で「有人対応」「チケット作成」の**選択肢だけ**が返り、backend がチケットを作らないこと
4. 雑談が固定応答で、tool を 1 つも呼ばないこと
5. 複合質問が**順に依存する** ReAct 多段になること（`query_order` の `tracking_no` を `query_logistics` が受け取る）

画面側（`http://localhost:8000`）では、苦情に対して 2 つのボタンが出ること、「チケット作成」でフォームが開き、送信すると `tickets` に 1 行入ることを確認する。

### 4. 06 章の受け入れ（実サービス）

DB のマイグレーションを 1 度だけ流す（返金申請は `tickets` を再利用し、`ticket_type` に `refund` を足す）:

```bash
docker exec -i support-mysql mysql -uroot -proot support < sql/06-ticket-type.sql
```

MySQL / Milvus / アプリを起動したうえで:

```bash
make smoke-interrupt   # interrupt / resume が動くこと(上流を呼ばない)
make eval-06           # 受け入れ 5 条件
```

確認する内容:

1. 会話をまたいで intent が動くこと（配送 → 返金返品 → 配送。2 turn 目は指示語だけ）
2. 返金の subflow が**順に**通ること（注文の特定 → 規約検索 → 可否の判断）
3. 期限を過ぎた注文では申請フォームを**出さない**こと
4. 注文番号が無ければ推測せず、一覧を出して**止まる**こと
5. 選ばれた注文で**続きから**進むこと

プロンプト側は評価セットで見る:

```bash
uv run --env-file .env python scripts/eval_intent.py   # 8 分類 + 「その他」への退避
uv run --env-file .env python scripts/eval_coref.py    # 指示対象の解決と素通し
uv run --env-file .env python scripts/eval_expand.py   # 検索クエリの展開
```

画面（`http://localhost:8000`）では、「返金したいです」で**注文カード**が並び、選ぶと同じ会話の続きとして判断が流れ、対象なら「返金申請を送信」ボタン → フォーム → `tickets` に `ticket_type='refund'` の行が入ることを確認する。

### 5. 07 章の受け入れ（実サービス）

会話の要約の列とテーブルを 1 度だけ流す:

```bash
docker compose exec -T mysql mysql --default-character-set=utf8mb4 -uroot -proot support < sql/07-ddl.sql
docker compose exec -T mysql mysql --default-character-set=utf8mb4 -uroot -proot support < sql/07-layers.sql
```

要約プロンプトは正解付きのサンプルで見る:

```bash
make eval-07
```

画面では、左の切り替え欄で会話を行き来できることと、長い会話でも序盤の事実（注文番号など）を
覚えていることを確認する。そのターンでモデルへ実際に渡した文脈は log から読める:

```bash
grep model_ctx log/app.log
```

### 6. 08 章の受け入れ（実サービス）

監査ログのテーブルを 1 度だけ流す:

```bash
docker compose exec -T mysql mysql --default-character-set=utf8mb4 -uroot -proot support < sql/08-ddl.sql
```

**このチャプターからは MCP Server 2 台が要る。** `make dev` が起動時に立ち上げるが、単体でも操作できる:

```bash
make mcp-up     # logistics=:8101 / after-sales=:8102
make mcp-down
```

受け入れのサンプルはラベル付きで機械判定する:

```bash
make eval-08
```

確認する内容:

1. **ファイルを 1 つ置くだけでツールが増える** — `cp scripts/demo_08_promotions.py.txt app/tools/builtin/promotions.py`
   してアプリを再起動すると、core code を 1 行も変えずに registry へ載る
2. **MCP のツールが built-in と同じ一覧に並ぶ** — 配送状況の照会は別プロセスの MCP Server が担当し、
   internal な状態コードは client 側の formatter が日本語へ訳す
3. **MCP Server にツールを足すと、本体を再起動せずに次のターンから見える**（tool list は毎回取得する）
4. **チケット作成は必ず確認カードを挟む** — 「送信を確認」を押すまで `tickets` には 1 行も入らない
5. **キャンセルすると作成されず、監査に権限拒否が残る**
6. **タイムアウトと retry** — `MOCK_DELAY_SECONDS=8` で MCP Server を起動すると再試行の記録が残り、
   書き込み系（チケット作成）は二重実行を避けるため retry しない

すべてのツール呼び出しは `tool_audit_logs` に 1 行ずつ残る:

```sql
SELECT tool_name, tool_source, mcp_server, status, retry_count, duration_ms
  FROM tool_audit_logs ORDER BY id DESC LIMIT 20;
```

画面（`http://localhost:8000`）では、「猫用トイレのコンセントから火花が出ました。担当の方に対応して
ほしいです」と伝えると**チケット内容の確認カード**が出て、「送信を確認」でチケット番号が返り、
「キャンセル」では作成されないことを確認する。

## ディレクトリ

```
app/            アプリ本体(api / core / schemas)
tests/          ユニットテスト、ラベル付きサンプル(tests/data/)
scripts/        評価・デモスクリプト(eval_extract.py, demo_chat.sh)
```
