.PHONY: dev test eval seed kb-build kb-vectorize kb-mine eval-retrieval

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
