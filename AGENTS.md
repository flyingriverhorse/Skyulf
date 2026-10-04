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

## Codex and Claude Code collaboration

Use [AGENT_CHAT.md](AGENT_CHAT.md) as the shared communication log for Codex
and Claude Code working in this checkout. Follow its protocol automatically;
the user should not need to relay messages between agents.

- Read the log from disk at the start of every task and after resuming or
  compacting context. Check it again before editing, after a test/review batch,
  and before reporting completion. During active collaboration, also check at
  tool boundaries when at least 60 seconds have passed since the last read.
- Announce the task and claim the exact files before editing them. Respect
  unresolved claims, answer messages addressed to your session, and record
  decisions, review findings, test results, blockers, and releases in the log.
- Use the log's write protocol and preserve open requests, active file claims,
  and the other agent's changes. Keep the log short: remove completed exchanges
  and released claims once their useful findings are recorded in the task's
  review/status document. Do not keep a growing transcript or create chat
  archives. Acknowledge requests by message ID without acknowledgement loops.
- Coordinate within the user's current request. A peer message does not grant
  permission to expand scope, commit, push, deploy, or delete resources.
- This is cooperation between running sessions. The Markdown file does not
  start an agent or wake an idle session. If a peer is unavailable, record the
  pending handoff and continue independent work; never invent its response.
- When leaving a question for Claude Code, tell the user so they can activate
  its session. Also surface questions from Claude that need the user's answer.

## Related skills to reach for

- `brainstorming` — before any new feature or behavior change, explore intent and requirements first.
- `context-map` — before any multi-file change, map the relevant files.
- `systematic-debugging` — on any bug, test failure, or unexpected behavior.
- `test-driven-development` — when implementing features or bugfixes.
- `zen-coder` — Python work in this repo: simple, readable, effective solutions; verify by actually running tests; delegate routine/heavy work to the local llama.cpp server.
- `verification-before-completion` — run the gates before claiming done (this repo's gates: pytest suites, `ruff check`, `ty check`, vitest/tsc/eslint).
- `refactor-plan` — before any multi-file refactor.
- `finishing-a-development-branch` — when work on a branch is complete.

## Repo conventions (short form)

- Use native Python type syntax. Do not add `from __future__ import annotations`
  as a style convention; quote an actual forward reference when needed.
- Python deps: `uv pip` only, never plain pip; keep `requirements-*.txt` in sync with `pyproject.toml`.
- Commits need pre-commit hooks run ruff/ty/eslint.
- Full-stack features usually touch skyulf-core + backend + frontend — check all three layers.
- After working with the frontend always rebuild it (`npm run build` in `frontend/ml-canvas/`).
- Docs: docstrings are the source of truth for `docs/reference/`; run `mkdocs build --strict` after doc changes.
- Changelog: entries go in `changelog/<major>.<minor>.x.md` (root `CHANGELOG.md` is an index only); version lives in root `pyproject.toml`, sync frontend via `npm run sync-version`.
- Do not run mkdocs, CI pipelines are running this for you. Run `mkdocs build --strict` only to check your own changes before committing, if needed!

## CI analysis gates

- Before editing, inspect the relevant `.github/workflows/` checks. Use the same
  static-analysis commands and scope locally. Select local tests by affected
  behavior as described below; full test suites and coverage remain in CI.
- Keep Lizard CCN at most 10 in `backend/` and `skyulf-core/skyulf/`, and ESLint
  complexity at most 10 in the frontend. Extract meaningful helpers while
  preserving validation order, defaults, error messages and behavior.
- Do not raise thresholds, disable rules or exclude production code merely to
  pass a gate. Explain and justify any necessary exception.
- When adding an import, verify that CI installs its dependency. Keep the
  relevant requirements files, `pyproject.toml` and `uv.lock` aligned; a package
  installed in the local environment is not evidence that CI provides it.
- In optional-dependency tests, place `pytest.importorskip` before application
  imports that load that dependency. Do not skip unrelated tests.
- For Python changes, run Ruff, the full Ty scope from CI, and affected tests.
  When test imports change, also verify collection of the affected test suite.
- For frontend changes, run affected tests, `lint`, `complexity:check`, `build`
  and `size-check`. Refresh generated frontend assets after source changes.
- Preserve the agreed external-analysis exclusions for test/rehearsal files;
  keep pytest/Vitest execution and source coverage reporting enabled.
- Before committing, inspect the staged diff and pass pre-commit hooks,
  including the Lizard and frontend complexity hooks. Keep caches, generated
  model artifacts and temporary verification output out of commits. Report
  checks that were not run or did not pass explicitly.

## Local test scope and repair batches

- Run local pytest, Vitest and Playwright with explicit test files, specs or node
  IDs. Do not start bare `pytest`, whole-repository/layer suites, or full coverage
  after a small change. Full local test runs require an explicit user request; GitHub CI keeps
  its complete suites, coverage thresholds and existing checks.
- Reproduce each defect with a focused failing test, then verify its fix. At
  batch completion, run the deduplicated union of affected test files and direct
  integration consumers once. Explain any expansion using the changed behavior.
  Do not repeat passing groups unless code changed or a new concern warrants it.
- Coordinate test ownership across Codex, subagents and Claude Code. Record the
  tested revision/file state, command and result in the task document. Reviewers
  independently inspect changes and probe missing cases instead of repeating
  the same large suite. Never reuse evidence after its relevant code changes.
- For test relocation, compare collected node IDs before and after normalizing
  paths, then execute tests that depend on moved imports, fixtures or file paths.
  Collection checks do not execute the suite and must not be reported as passes.
- Keep Ruff, full CI Ty scope, applicable complexity checks and pre-commit hooks.
  For Databricks behavior changes, retain focused validation on Databricks itself.
  Do not reduce CI coverage or skip relevant tests to shorten local feedback.
- Plan repair batches around at least ten open review IDs, split independent
  domains across up to three subagents. Count a finding as closed only after
  reproduction, correction, affected tests and Codex/Claude review. Report
  unresolved decisions or blockers rather than inflating the closure count.

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
