.PHONY: dev test eval seed

dev:
	uv run --env-file .env uvicorn app.main:app --port 8000 --reload

test:
	uv run pytest -v

eval:
	uv run --env-file .env python scripts/eval_extract.py

seed:
	docker compose exec -T mysql mysql -uroot -proot support < sql/02-seed.sql
