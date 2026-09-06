.PHONY: dev test eval

dev:
	uv run --env-file .env uvicorn app.main:app --port 8000

test:
	uv run pytest -v

eval:
	uv run --env-file .env python scripts/eval_extract.py
