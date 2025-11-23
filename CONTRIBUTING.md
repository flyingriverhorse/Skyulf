# Contributing

Thanks for considering contributing! This repo focuses on the backend (FastAPI + Celery + scikit-learn).

## Dev setup
- Install Python 3.12 (3.10+ supported).
- Install uv: `pip install uv`
- Sync deps (app + dev): `uv sync --dev`
- Activate the virtualenv if needed: `.venv` is created in the project root by uv.

Alternatively, without uv:
- `python -m venv .venv && .venv/Scripts/Activate.ps1` (Windows PowerShell)
- `pip install -r requirements-fastapi.txt -r requirements-ci.txt`

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

We recommend Conventional Commits (optional):
- `feat: add data source preview for parquet`
- `fix: handle unicode column names in EDA`
- `docs: update quick start for Windows`

Branch names:
- `feature/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`

Open a Draft PR early if helpful; CI runs on PRs for Python 3.10 and 3.11.

## Issue triage
- Include environment, steps to reproduce, expected vs actual.
- Label runtime issues vs. feature requests.

Labels used:
- `bug`, `enhancement`, `deps`, `chore`, `documentation`, `good first issue`

## Security
See `SECURITY.md`. Please do not open public issues for sensitive reports.

## Contributor License Agreement (CLA)

By opening a Pull Request, you agree to the terms in `CLA.md` and grant the project the right to relicense your contribution.

Please read `CLA.md` for the short-form agreement that contributors accept when submitting changes.

© 2025 Murat Unsal — Skyulf Project
