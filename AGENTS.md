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
- Commits need pre-commit hooks run ruff/ty/eslint.
- Full-stack features usually touch skyulf-core + backend + frontend — check all three layers.
- After working with the frontend always rebuild it (`npm run build` in `frontend/ml-canvas/`).
- Docs: docstrings are the source of truth for `docs/reference/`; run `mkdocs build --strict` after doc changes.
- Changelog: entries go in `changelog/<major>.<minor>.x.md` (root `CHANGELOG.md` is an index only); version lives in root `pyproject.toml`, sync frontend via `npm run sync-version`.
- Do not run mkdocs, CI pipelines are running this for you. Run `mkdocs build --strict` only to check your own changes before committing, if needed!

## Lint scope & test hygiene

Ruff scope (see the comments in `pyproject.toml` for the reasoning): `F401`,
`F841` and the pydocstyle `D` family are enforced on `skyulf-core/skyulf/` and
`backend/`. `D` is **waived for `tests/**`, `skyulf-core/tests/**`, examples and
benchmarks** — those hold **2,528** sites (2,484 tests, 33 examples, 11
benchmarks; re-measured 2026-09-06 by running `ruff check . --select D` with the
per-file-ignores removed, which leaves **0** in `backend/` or
`skyulf-core/skyulf/`) and writing them by hand would have found no defects. The
waiver is a linting decision, not a change
to the standard: `coding_standards.instructions.md` §4 still asks for a docstring
on every function, tests included. Nothing will remind you, so:

- **New tests still get a one-line docstring** stating the behaviour being
  pinned — what would break if the test failed. The test name carries the *what*,
  the docstring carries the *why it matters*.
- **A test body must end in a real assertion.** `F841` is enabled precisely
  because an unused local in a test is a missing-assertion signal first and a
  lint problem second: enabling it exposed three tests whose bodies stopped
  before any assertion and had therefore always passed trivially. Read the body
  before you strip the binding.
- Mock `assert_called()` / `assert_awaited_once()` and a `pytest.raises` block
  **are** assertions — a grep for `assert ` misses them. Don't "fix" a test that
  already checks something.
- **Never rename an identifier to silence a linter.** Prefixing an unused
  parameter (`config` → `_config`) breaks keyword callers and surfaces as a pile
  of type errors, not as a lint win. For a genuinely unused argument, leave the
  signature alone. The unused-argument rules (`ARG`) are deliberately **not**
  enabled: the 121 in-scope sites are mostly `Calculator`/`Applier` and framework
  contract signatures that cannot be renamed without breaking keyword callers, so
  enforcing them meant carrying ~90 permanent waivers to surface ~4 real defects.
  Those defects were fixed on their own merits instead.
- For unused imports and locals, check whether the binding is load-bearing
  before deleting: availability probes (`import polars` inside a `try` whose
  `except ImportError` sets a skip flag) and side-effecting calls keep their
  line and take a `# noqa: F401 - reason` waiver instead, following the existing
  `BLE001` precedent.
- `D` fixes are all marked unsafe, so pre-commit's `ruff --fix` will **not**
  write docstrings for you — add them by hand, Google style, and run
  `ruff check` before committing.
- **A clean `ruff check .` does not prove every function has a docstring.** The
  `D1xx` missing-docstring rules fire on *public* definitions only, and ruff
  derives publicity from the whole dotted module path — a leading underscore on
  the module, or on **any enclosing package**, exempts everything beneath it.
  That silently hid 82 more sites across 112 of 327 in-scope modules
  (`backend/ml_pipeline/_internal`, `_execution`, `_services`;
  `skyulf/modeling/_tuning` and the underscore-prefixed modeling/preprocessing
  modules), including the job routers and the whole tuning engine. All 82 have
  been written by hand. To audit a private module, copy it to a public filename
  *outside its package* and lint the copy — the reported line numbers carry over
  unchanged, because the copy is byte-identical. Renaming the real file is not an
  option; the underscore is deliberate encapsulation. The `D2xx`/`D4xx`
  formatting rules are *not* privacy-gated, so those did reach every file.
  Matching blind spot for length: `E501` is not selected and `ruff format` will
  not reflow a docstring or comment, so a 105-character prose line passes every
  gate this repo runs — count your characters.
- **A pydantic model's docstring is part of the wire format.** It becomes the
  `description` in `model_json_schema()`, so it reaches clients through the
  OpenAPI document. Adding one to a request/response model will fail
  `tests/integration/test_pipeline_config_snapshots.py` until the snapshot is
  updated intentionally with `pytest <that file> --snapshot-update`. Check the
  resulting diff is descriptions-only (insertions, no deletions) before keeping
  it — that test exists to catch renamed fields and new required keys. Never
  hand-edit an `.ambr`: a multi-line value is written as an indented `'''` block,
  where a blank line *inside the value* becomes an indentation-only line, so the
  whitespace looks wrong but is load-bearing. That is why
  `__snapshots__/*.ambr` is excluded from the auto-fix hooks in
  `.pre-commit-config.yaml`; if a snapshot looks malformed, regenerate it.
