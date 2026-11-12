.PHONY: test lint format theory paper

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

theory:
	uv run python theory/stage1_ar1/verify.py

paper:
	cd paper && tectonic -X compile --synctex --keep-logs main.tex
