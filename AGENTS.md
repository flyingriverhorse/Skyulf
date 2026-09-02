# Working instructions — Skyulf

## Repo map (load first)

When locating, fixing, or building anything in this repo, load the
`skyulf-codebase-map` skill first. It maps the three layers
(skyulf-core library / FastAPI+Celery backend / frontend/ml-canvas),
key flows (job lifecycle, tuning, drift, threshold tuning), and known
traps.

Three layers, strict dependency direction: `frontend` → (HTTP) → `backend` → (import) → `skyulf-core`.

- **`skyulf-core/`** — standalone ML library. Stateless `Calculator`/`Applier` node pairs wrapping pandas/numpy/scikit-learn. No FastAPI, no Celery, no DB, no filesystem access. pandas-only (never polars).
- **`backend/`** — FastAPI + Celery API server. User requests, file uploads, DB, async job execution. Uses `polars` for ingestion/ETL; the polars↔pandas boundary is `backend/services/data_service.py` — never pass a polars DataFrame into a core node.
- **`frontend/ml-canvas/`** — React + TypeScript + React Flow canvas. Talks to backend via REST. New node types must be registered in `src/core/registry/init.ts`.

Key flows: job lifecycle (upload → ETL → pipeline run → results), hyperparameter tuning, drift detection, threshold tuning.

## Related skills to reach for

- `brainstorming` — before any new feature or behavior change, explore intent and requirements first.
- `context-map` — before any multi-file change, map the relevant files.
- `systematic-debugging` — on any bug, test failure, or unexpected behavior.
- `test-driven-development` — when implementing features or bugfixes.
- `zen-coder` — Python work in this repo: simple, readable, effective solutions; verify by actually running tests; delegate routine/heavy work to the local llama.cpp server.
- `verification-before-completion` — run the gates before claiming done (this repo's gates: pytest suites, `ruff check`, `ty check`, vitest/tsc/eslint).
- `refactor-plan` — before any multi-file refactor.
- `finishing-a-development-branch` — when work on a branch is complete.

## TOON MCP (token-efficient JSON ingestion)

The `toon` MCP server is registered in user scope (`~/.copilot/mcp-config.json`),
installed at `~/.local/share/toon-mcp` (own venv, stdio). Tools:
`encode_toon`, `convert_file_to_toon`, `estimate_token_savings`.

**Trip-wire rule:** before reading any JSON larger than ~5 KB / ~100 lines
that is an array of objects sharing the same keys, convert it with
`convert_file_to_toon(file_path=..., output_path=...)` and read the written
file instead of the raw JSON. For command output (`gh api`, coverage reports,
lockfiles), write it to a temp file first, then convert — never paste large
JSON into `encode_toon`. Always pass `output_path` for large files so the
payload doesn't round-trip through context.

**Do NOT use TOON for:** human-facing output (Markdown tables are only ~12%
larger and more readable), small payloads (< ~5 KB), irregular/nested
structures, or anything another program parses (configs, fixtures, API
bodies). Savings are ~35% vs compact JSON / ~44% vs pretty JSON, only on
uniform record lists.

## Repo conventions (short form)

- Python deps: `uv pip` only, never plain pip; keep `requirements-*.txt` in sync with `pyproject.toml`.
- Commits need DCO sign-off; pre-commit hooks run ruff/ty/eslint.
- Full-stack features usually touch skyulf-core + backend + frontend — check all three layers.
- After working with the frontend always rebuild it (`npm run build` in `frontend/ml-canvas/`).
- Docs: docstrings are the source of truth for `docs/reference/`; run `mkdocs build --strict` after doc changes.
- Changelog: entries go in `changelog/<major>.<minor>.x.md` (root `CHANGELOG.md` is an index only); version lives in root `pyproject.toml`, sync frontend via `npm run sync-version`.
