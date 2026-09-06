.PHONY: dev test eval seed seed-conv kb-preview kb-build kb-vectorize kb-mine kb-reset eval-retrieval eval-mining

dev:
	uv run --env-file .env uvicorn app.main:app --port 8000 --reload

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
