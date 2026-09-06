.PHONY: dev test eval seed seed-conv kb-build kb-vectorize kb-mine eval-retrieval eval-mining

dev:
	uv run --env-file .env uvicorn app.main:app --port 8000 --reload

test:
	uv run pytest -v

eval:
	uv run --env-file .env python scripts/eval_extract.py

seed:
	docker compose exec -T mysql mysql -uroot -proot support < sql/02-seed.sql

kb-build:
	uv run --env-file .env python scripts/build_kb.py

kb-vectorize:
	uv run --env-file .env python scripts/vectorize_kb.py

kb-mine:
	uv run --env-file .env python scripts/mine_knowledge.py

eval-retrieval:
	uv run --env-file .env python scripts/eval_retrieval.py

seed-conv:
	docker compose exec -T mysql mysql --default-character-set=utf8mb4 -uroot -proot support < sql/03-seed.sql

eval-mining:
	uv run --env-file .env python scripts/eval_mining.py
