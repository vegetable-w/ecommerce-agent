.PHONY: dev test eval seed seed-conv kb-preview kb-build kb-repatch kb-vectorize kb-mine kb-reset eval-retrieval eval-mining eval-rag eval-check judge-check eval-05 eval-06 smoke-interrupt calibrate-confidence eval-07 eval-08 mcp-up mcp-down langfuse-up langfuse-down

# --reload は付けない。この環境では watchfiles が入っていても変更を検知せず
# (実測: 起動後の app/tools/business.py の更新で reload されなかった)、
# 「直したのに反映されない」まま気づかない事故になる。さらに reloader の親を
# 止めても子プロセスがポートを掴んだまま残る。コードを変えたら手で起動し直す。
dev:
	$(MAKE) mcp-up
	uv run --env-file .env uvicorn app.main:app --port 8000

test:
	uv run pytest -v

eval:
	uv run --env-file .env python scripts/eval_extract.py

seed:
	docker compose exec -T mysql mysql -uroot -proot support < sql/02-seed.sql

kb-preview:
	uv run --env-file .env python scripts/preview_kb.py

kb-build:
	uv run --env-file .env python scripts/build_kb.py

# data/kb を書き換えたときの差分反映。変わっていない chunk は再度埋め込まず、
# id も変えない(Milvus の PK が chunk id なので、同じ id で upsert すれば置き換わる)。
kb-repatch:
	uv run --env-file .env python scripts/repatch_kb.py

kb-vectorize:
	uv run --env-file .env python scripts/vectorize_kb.py

kb-mine:
	uv run --env-file .env python scripts/mine_knowledge.py

# 破壊的: MySQL の 2 テーブルを空にし、Milvus の collection を削除する
kb-reset:
	uv run --env-file .env python scripts/reset_kb.py

eval-retrieval:
	uv run --env-file .env python scripts/eval_retrieval.py

seed-conv:
	docker compose exec -T mysql mysql --default-character-set=utf8mb4 -uroot -proot support < sql/03-seed.sql

eval-mining:
	uv run --env-file .env python scripts/eval_mining.py

# service 名は docker-compose.yml の定義に合わせる(コンテナ名は milvus-* だがサービス名は etcd/minio/milvus)
milvus-up:
	docker compose up -d etcd minio milvus
	@echo "Milvus の起動を待機中 (healthz)..."; 	for i in $$(seq 1 60); do 	  curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && echo "Milvus OK" && exit 0; 	  sleep 3; done; echo "Milvus が ready になりません" && exit 1

milvus-down:
	docker compose stop etcd minio milvus

smoke-rag:
	uv run --env-file .env python scripts/smoke_milvus_bm25.py
	uv run --env-file .env python scripts/smoke_rerank.py

# 4 戦略の RAG 評価。埋め込み / リランク / チャットの上流をすべて呼ぶため課金される。
# 生成段を省いて決定的な Stage 1/2 だけ回すなら --no-generation を付ける。
eval-rag:
	uv run --env-file .env python scripts/eval_04.py

# 評価セットの自己検証。MySQL の knowledge_chunks を読むだけで、上流も Milvus も使わない。
eval-check:
	uv run --env-file .env python scripts/validate_eval_04.py

# 06 章の受け入れ 4 条件を実サービス上で通す。全 service の起動が要る。
eval-06:
	uv run --env-file .env python scripts/eval_06.py

# 06 章の red line。interrupt / resume の形を実測する。上流は呼ばない。
smoke-interrupt:
	uv run --env-file .env python scripts/smoke_interrupt.py

# 05 章の受け入れ 5 条件を実サービス上で通す。アプリと MySQL / Milvus の起動が要る。
eval-05:
	uv run --env-file .env python scripts/eval_05.py

# judge の回帰チェック。幻覚ケース台帳に人が付けた印を正解として judge だけを測り直す。
# 検索も生成もやり直さない(台帳のスナップショットを judge へ渡すだけ)。DB は読み取りのみ。
# 対象 1 件につき judge を 1 回呼ぶので、eval-rag よりずっと軽いが課金はされる。
judge-check:
	uv run --env-file .env python scripts/judge_check.py

# 07 章の要約プロンプトを正解付きの 3 件で測る。上流のチャットモデルを呼ぶ。
eval-07:
	uv run --env-file .env python scripts/eval_07.py

# 08 章のラベル付きサンプル 5 件を実サービス上で通す。MySQL と MCP Server 2 台
# (mcp-up)の起動が要る。tickets と tool_audit_logs に行が増える(それが測っているもの)。
eval-08:
	uv run --env-file .env python scripts/eval_08.py

# 08:2 台の business MCP Server の起動/停止(independent process、Streamable HTTP :8101/:8102)。
#
# 停止に pid file を使わないのは実測の結果:
#  - `nohup uv run python ... &` の $! は uv の wrapper を指す。それを kill しても
#    子の python は生き残り、:8101 を LISTENING のまま掴み続けた。
#  - uv を外して .venv の python を直接叩いても同じ。Windows の venv python は
#    trampoline → 本体 の 2 段で、bash の job PID を kill しても本体が残る。
# どちらも「古い Server が残ったまま新しいコードが反映されない」事故になるので、
# 停止は port から LISTENING しているプロセスの Windows PID を引いて木ごと落とす。
# 前回の起動が残した孤児もこれで一緒に片付く。
# recipe を 1 物理行に畳んであるのは milvus-up と同じ理由(継続行を使わない)。
MCP_LISTENER = netstat -ano | awk -v pat=":$$p$$" '$$1=="TCP" && $$4=="LISTENING" && $$2 ~ pat {print $$5}'

mcp-up:
	@mkdir -p log data
	@nohup uv run python mcp_servers/logistics_server.py  > log/mcp-logistics.log 2>&1 &
	@nohup uv run python mcp_servers/aftersales_server.py > log/mcp-aftersales.log 2>&1 &
	@for i in $$(seq 1 60); do ok=1; for np in logistics:8101 aftersales:8102; do n=$${np%%:*}; p=$${np##*:}; pid=$$($(MCP_LISTENER) | head -1); if [ -z "$$pid" ]; then ok=0; else echo $$pid > data/mcp-$$n.pid; fi; done; if [ $$ok = 1 ]; then echo "MCP Servers started: logistics=:8101 aftersales=:8102(pid: data/mcp-*.pid)"; exit 0; fi; sleep 1; done; echo "MCP Server が起動しません。log/mcp-*.log を確認してください" && exit 1

mcp-down:
	@for p in 8101 8102; do for pid in $$($(MCP_LISTENER) | sort -u); do taskkill //F //T //PID $$pid >/dev/null 2>&1 || kill -9 $$pid 2>/dev/null || true; done; done
	@rm -f data/mcp-logistics.pid data/mcp-aftersales.pid
	@echo "MCP Servers stopped"

# 09 Langfuse self-hosted。**既存の docker-compose.yml とは別 stack**で、
# project 名は docker-compose.langfuse.yml の `name: langfuse` が固定している
# (既定の project 名は directory 名になり、mysql / milvus を巻き込む)。
# 初回は image の pull と migration で 2〜3 分かかる。固定の sleep ではなく
# health endpoint が通るまで待つ(pull にかかる時間はマシンによって桁が違う)。
# recipe を 1 物理行に畳んであるのは milvus-up と同じ理由(継続行を使わない)。
langfuse-up:
	docker compose -f docker-compose.langfuse.yml up -d
	@echo "Langfuse の起動を待機中 (http://localhost:3000/api/public/health)..."; 	for i in $$(seq 1 120); do 	  curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1 && echo "Langfuse OK  http://localhost:3000  (admin@support.local / support123)" && exit 0; 	  sleep 3; done; echo "Langfuse が ready になりません。docker compose -f docker-compose.langfuse.yml logs langfuse-web を確認してください" && exit 1

langfuse-down:
	docker compose -f docker-compose.langfuse.yml down

# 09 章の確信度のしきい値を 04 章の評価セットで校正する。
# 検索とリランクだけを 300 件走らせる(生成も judge も呼ばない)。
# Milvus の起動とナレッジベースの構築が要る。
calibrate-confidence:
	PYTHONUTF8=1 uv run --env-file .env python scripts/calibrate_confidence.py
