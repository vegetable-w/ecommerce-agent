# EC カスタマーサポート エージェント

EC サイトのカスタマーサポートを自動化するバックエンド。RAG で社内ナレッジを引きながら、注文・配送・返品などの問い合わせに答える。

設計の軸は **「自由に答えさせない」** こと。ポリシーに関する質問は必ず検索を通し、根拠が弱ければ答えずに断る。注文番号が分からなければ推測せず画面で選ばせる。チケット作成のような書き込みは、利用者が確認するまで実行しない。そして **答えられなかった質問は捨てずに集め、担当者が確認したうえでナレッジへ還す。**

## 主な機能

| | |
|---|---|
| **対話** | SSE ストリーミング、マルチターン。長い会話はスライディングウィンドウ + 非同期の要約で保持 |
| **ルーティング** | 意図を 9 分類し、知識系は強制検索、苦情は固定フロー、返金は多段の subflow へ |
| **RAG** | BM25 / Dense のハイブリッド検索 + RRF + リランカー。根拠の確信度が低いターンは回答を拒否 |
| **ツール** | 注文・商品・FAQ・チケットの内製ツールと、MCP 経由の外部ツールを同一の枠組みで実行 |
| **ユーザー確認** | 注文が特定できないときと、チケットを作るときは処理を中断し、画面の選択を待って再開 |
| **改善ループ** | 答えられなかった質問を収集 → 正規化・重複排除 → レビュー → ナレッジへ書き戻し |
| **可観測性** | Langfuse で 1 リクエストの trace を追跡。意図別の token コストと評価指標のトレンドを画面で確認 |

## アーキテクチャ

1 ターンの流れ（`app/graph/`）:

```
発話
 └─ 指示語の解決 ─ 意図分類 ─┬─ knowledge  ─ 強制検索 ─ 確信度ゲート ─ 回答 / 拒否
                             ├─ business   ─ ReAct（ツールを選んで多段で呼ぶ）
                             ├─ refund     ─ 注文特定 → 規約照会 → 可否判断
                             ├─ escalate   ─ 苦情の固定応答 + 選択肢
                             └─ fallback   ─ 聞き直し
```

- **Workflow と Agent のハイブリッド。** 確実性が要る経路（強制検索・苦情・返金）は固定のグラフで縛り、柔軟性が要る経路（注文・配送などの business）だけ Agent に道具を選ばせる。
- **中断と再開。** 注文が特定できない場合とチケット作成の確認は、LangGraph の `interrupt` でグラフを止め、画面の選択を `resume` で受けて同じターンの続きから走る。
- **会話の保存。** checkpointer（sqlite）がターンをまたいで状態を持つ。発話と回答は MySQL にも残る。

## セットアップ

### 必要なもの

- Python（`uv` で管理）、Docker Desktop
- 上流 API（OpenAI 互換のチャット / 埋め込み / リランク）

### 手順

```bash
cp .env.example .env      # 上流の接続情報を記入する。.env は絶対にコミットしない
docker compose up -d      # MySQL + Milvus
make dev                  # MCP Server 2 台 + アプリ（http://localhost:8000）
```

ナレッジベースは初回だけ構築する:

```bash
make kb-build       # data/kb/ の資料を chunk 化して DB へ
make kb-vectorize   # 埋め込んで Milvus へ
```

DB のスキーマは `sql/` に章ごとの DDL がある。まとめて流す場合:

```bash
for f in sql/02-ddl.sql sql/03-ddl.sql sql/04-ddl.sql sql/06-ticket-type.sql \
         sql/07-ddl.sql sql/07-layers.sql sql/08-ddl.sql sql/09-ddl.sql; do
  docker compose exec -T mysql mysql --default-character-set=utf8mb4 -uroot -proot support < "$f"
done
```

**Windows の場合:** シェルスクリプトと評価スクリプトは Git Bash から実行すること。PowerShell / cmd.exe は動作保証外。

### 起動するもの

| | ポート | 起動 |
|---|---|---|
| アプリ | 8000 | `make dev` |
| MySQL / Milvus | 3306 / 19530 | `docker compose up -d` |
| MCP Server（物流 / アフターサービス） | 8101 / 8102 | `make dev` が自動起動（単体は `make mcp-up`） |
| Langfuse（任意） | 3000 | `make langfuse-up` |

Langfuse は任意。`.env` に 3 つの変数が揃ったときだけ trace を送り、未設定なら何も送らずに通常動作する。

## API

### 対話

| | |
|---|---|
| `POST /api/chat` | 画面の入口。SSE。フレームは `delta` / `tool` / `citations` / `actions` / `interrupt` / `done`、終端は `[DONE]` |
| `POST /api/agent` | 非ストリーミング。評価とテスト用 |
| `POST /api/actions/resume` | 中断したターンの再開。注文の選択（`order_id`）とチケットの確認（`confirmed`）が同じ入口を使う |
| `POST /api/feedback` | 👎 を改善ループへ流す。👍 はログのみ |
| `GET /api/conversations` | 会話の一覧と履歴（画面の切り替え欄） |

### 書き込み

| | |
|---|---|
| `POST /api/actions/create-ticket` | 有人対応チケット。**利用者がボタンを押したときだけ** `tickets` へ書く |
| `POST /api/actions/create-refund` | 返金申請フォームの送信 |

### 運用

`/api/kb`（ナレッジの取り込み・検索）、`/api/review`（レビューキュー）、`/api/observability`（コストと評価のトレンド）、`/api/jobs`（バッチの起動と進捗）、`/api/rag-eval`（評価レポート）、`/api/admin`（各画面の要約）。

### その他

`POST /api/extract` — 問い合わせ本文から `AfterSalesTicket` を JSON で返す。`order_id`（nullable）、`request_type`、`expected_solution`。

## 画面

| | |
|---|---|
| `/` | チャット。会話の切り替え、注文の選択カード、チケット確認カード、引用の表示 |
| `/admin` | 管理コンソールの入口 |
| `/kb` | ナレッジの取り込みと検索の確認 |
| `/rag-eval` | 検索戦略の比較レポート、幻覚ケースの台帳 |
| `/review` | レビューキュー。承認するとナレッジへ書き戻る |
| `/observability` | 意図別コスト、評価トレンド、確信度しきい値の校正 |

## テスト

```bash
make test        # ユニットテスト。上流にも Milvus にも触らない
```

DB を使うテストは `support_test` を使う（本番相当の `support` には書かない）。

## 評価

品質は感覚ではなく指標で見る。いずれも実際の上流を呼ぶため課金される。

| | 内容 |
|---|---|
| `make eval-rag` | 検索戦略 4 種を 300 問の評価セットで比較。Recall@5 / MRR / Evidence Coverage / Answer Coverage / Faithfulness（**最も重い**） |
| `make eval-retrieval` | 検索のみ。生成も judge も呼ばない |
| `make calibrate-confidence` | 確信度ゲートのしきい値を評価セットの分布から校正 |
| `make eval-flywheel` | 評価を実行して `eval_runs` に残し、前回と比較。`LIMIT=40` で件数を絞れる |
| `make cost-report` | 意図別の token コスト（Langfuse が要る） |
| `scripts/eval_intent.py` 他 | 意図分類・指示語解決・クエリ展開のラベル付きサンプル（`uv run --env-file .env python scripts/eval_intent.py`） |

最新の結果（300 問、Hybrid+Rerank）:

| 指標 | 値 |
|---|---|
| Recall@5 | 98.1% |
| MRR | 86.0% |
| Faithfulness | 99.3% |
| 幻覚率 | 0.7%（2/300） |

ハイブリッド検索は単独では純 Dense（96.5%）を下回る（90.1%）。リランカーを足して 98.1% に戻る。詳細は `data/04/reports/`。

## 改善ループを回す

答えられなかった質問は `low_confidence_questions` に溜まる。入口は 3 つ（根拠の確信度が低い / 生成側の自己点検で不十分 / 利用者の 👎）で、そのときの検索結果も一緒に保存する。「ナレッジに無い」のか「有るのに引けていない」のかを、レビュー時に見分けられるようにするため。

```bash
make flywheel     # 溜まった質問を正規化・重複排除してレビューキューへ
```

`/review` で内容を確認し、承認するとナレッジベースへ書き戻り、ベクトル化まで走る。**次に同じ質問が来たときは引用付きで答えられる。**

## ディレクトリ

```
app/
  api/       HTTP の入口
  graph/     LangGraph のノードと配線（対話の本体）
  core/      検索・意図分類・要約・確信度・可観測性など
  tools/     ツールの registry / 実行エンジン / built-in ツール / MCP client
  kb/        ナレッジの取り込み、チャンク分割、Milvus
  db/        ORM と repository
mcp_servers/ 自作の MCP Server 2 台（物流 / アフターサービス）
data/kb/     ナレッジの原稿（Markdown）
sql/         DDL
scripts/     評価・校正・バッチ・smoke
static/      画面
tests/       ユニットテストとラベル付きサンプル（tests/data/）
```
