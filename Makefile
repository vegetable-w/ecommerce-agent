.PHONY: dev test eval seed seed-conv kb-preview kb-build kb-repatch kb-vectorize kb-mine kb-reset eval-retrieval eval-mining eval-rag eval-check judge-check eval-05 eval-06 smoke-interrupt

# --reload は付けない。この環境では watchfiles が入っていても変更を検知せず
# (実測: 起動後の app/tools/business.py の更新で reload されなかった)、
# 「直したのに反映されない」まま気づかない事故になる。さらに reloader の親を
# 止めても子プロセスがポートを掴んだまま残る。コードを変えたら手で起動し直す。
dev:
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
