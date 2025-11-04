# Contributing

Thanks for considering contributing! This repo focuses on the backend (FastAPI + Celery + scikit-learn).

## Dev setup
- Install Python 3.12 (3.10+ supported).
- Install uv: `pip install uv`
- Sync deps (app + dev): `uv sync --dev`
- Activate the virtualenv if needed: `.venv` is created in the project root by uv.

## Running tests
- Quick subset (fast, backend-only): `uv run pytest -q tests/test_training_tasks.py tests/test_hyperparameter_tuning_optuna.py tests/test_hyperparameter_tuning_strategies.py`
- Full suite (may require services): `uv run pytest -q`

## Lint & typecheck
- Black/Isort: `uv run black . && uv run isort .`
- Flake8: `uv run flake8`
- MyPy: `uv run mypy .`

## Commit style
- Keep diffs focused.
- Add/adjust tests when changing behavior.
- Prefer small PRs with context in the description.

## Issue triage
- Include environment, steps to reproduce, expected vs actual.
- Label runtime issues vs. feature requests.

## Security
See `SECURITY.md`. Please do not open public issues for sensitive reports.

## Contributor License Agreement (CLA)

By opening a Pull Request, you agree to the terms in `CLA.md` and grant the project the right to relicense your contribution.

Please read `CLA.md` for the short-form agreement that contributors accept when submitting changes.
