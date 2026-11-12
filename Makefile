.PHONY: test lint format theory paper e02-predictions e02-evaluate

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

theory:
	uv run python theory/check_all.py

paper:
	cd paper && tectonic -X compile --synctex --keep-logs main.tex

e02-predictions:
	uv run python experiments/e02_lgssm/predictions.py > experiments/e02_lgssm/results/predictions.txt

e02-evaluate:
	uv run python -u experiments/e02_lgssm/evaluate.py | tee logs/e02_evaluate.log
