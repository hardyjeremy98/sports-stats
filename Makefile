.PHONY: setup dev api worker web demo test lint

setup:
	uv sync
	cd web && npm install

api:
	uv run uvicorn matchlab_server.app:app --reload --port 8000

worker:
	uv run matchlab-worker

web:
	cd web && npm run dev

# Run api + worker + web together for local dev (SQLite, stub-friendly)
dev:
	$(MAKE) -j3 api worker web

demo:
	uv run matchlab-demo

test:
	uv run pytest packages -q

lint:
	uv run ruff check packages
	cd web && npm run lint --if-present
