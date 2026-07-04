# Version Update Log

## v0.6.5

### 🔧 skyulf-core

- **Pylint false-positive suppressions (`arguments-differ`, 89 issues):** `BaseCalculator.fit`/`BaseApplier.apply` are declared with a 3-parameter signature, but the `@fit_method`/`@apply_method` decorators (in `preprocessing/base.py`) wrap a 4-parameter inner implementation `(self, X, y, config)` before exposing the base-compatible signature at runtime. Pylint's static AST analysis only sees the pre-decoration signature, so every decorated `fit`/`apply` override across `skyulf-core/skyulf/preprocessing/**` (48 files) was flagged. Verified against Pylint's own docs that the `signature-mutators` option does not cover `arguments-differ`; added inline `# pylint: disable=arguments-differ` to each flagged definition instead.
- **Pylint false-positive suppressions (`no-value-for-parameter`, 14 issues):** Same decorator pattern as above also causes Pylint to misread call sites. Added targeted `# pylint: disable=no-value-for-parameter` at each affected call (`bucketing.py`, `feature_selection/facade.py`, and a file-level comment covering 12 call sites in `examples/06_text_nlp_vectorization.py`).
- **Pylint false-positive suppressions (`assignment-from-no-return`, 8 issues):** `profiling/_analyzer/_utils.py` defines `_AnalyzerState(Protocol)` with stub methods (`...` bodies, no return) that mixins inherit from. Pylint's resolver picks up the stub instead of the real mixin implementation. Added `# pylint: disable=assignment-from-no-return` at the 8 affected call sites in `column.py`, `analyzer.py`, and `rules.py`.
- **Bug fix:** `modeling/hyperparameters/__init__.py` — `MULTINOMIAL_NB_PARAMS` and `BERNOULLI_NB_PARAMS` were imported but missing from `__all__`, so they weren't actually re-exported despite the module docstring promising it. Added both alongside `GAUSSIAN_NB_PARAMS`.
- **Cleanup:** removed a fully dead `scale_range` nested function from `profiling/drift.py` (defined, never called) instead of just silencing its shadowed-builtin warning.
- **Cleanup:** `preprocessing/vectorization/count_vectorizer.py` — removed an unused `unpack_pipeline_input` import.
- **Security review (Bandit/Semgrep):** reviewed all remaining `try/except: pass`/`continue` fallback blocks (17 total across `bucketing.py`, `drift.py`, `_analyzer/dates.py`, `feature_generation/_pandas_ops.py`, `feature_generation/_polars_ops.py`, `modeling/_evaluation/classification.py`, `modeling/_evaluation/metrics.py`, `utils.py`) — all are legitimate best-effort/optional-metric fallbacks; annotated each with `# nosec B110`/`# nosec B112` and a short rationale rather than changing behavior.
- **Security review (Bandit/Semgrep):** `preprocessing/encoding/one_hot.py`'s `_MISSING_TOKEN` sentinel was flagged as a hardcoded credential by Bandit's naming heuristic — confirmed false positive, annotated `# nosec B105`.
- **Security review (Semgrep pickle):** `pipeline.py`'s `save`/`load`/artifact-hashing use of `pickle` only ever operates on artifacts produced and consumed by the same trusted process — added `# nosemgrep: avoid-pickle` alongside the existing `# nosec B301` annotations, with rationale.

### 🐛 Backend

- **Bug fix:** `backend/database/async_init_db.py` — `DatabaseType` was imported inside a `try:` block but referenced in the corresponding `except Exception:` fallback; if the import itself failed this would raise `NameError` instead of falling back cleanly. Moved the import to module level.
- **Cleanup:** `backend/data_ingestion/schemas/ingestion.py` — removed an unused `List` import.
- **Security review:** `backend/main.py` — SQL table name is f-string interpolated into an `UPDATE` statement, but the value only ever comes from a hardcoded 2-tuple (`"basic_training_jobs"`, `"advanced_tuning_jobs"`), never user input. Annotated with `# nosec B608` / `# nosemgrep: hardcoded-sql-expression, avoid-sqlalchemy-text` plus an explanatory comment.
- **Security review:** `backend/config/environments.py` — `HOST: "0.0.0.0"` dev-only default annotated `# nosec B104` (never used for production defaults).
- **Security review:** `run_fastapi.py` — `subprocess.Popen` call uses a static list of literals with no shell and no untrusted input; import and call site both annotated (`# nosec B403/B603`, `# nosemgrep: dangerous-subprocess-use-audit`).
- **Security review:** reviewed remaining `try/except: pass` fallbacks in `backend/ml_pipeline/_execution/advanced_tuning_manager.py`, `backend/ml_pipeline/_execution/basic_training_manager.py`, `backend/ml_pipeline/artifacts/s3.py`, `backend/data_ingestion/connectors/s3.py`, `backend/realtime/manager.py`, `backend/database/engine.py`, `backend/database/async_connection_manager.py`, `backend/monitoring/router.py` — all legitimate best-effort patterns; annotated with `# nosec B110` and a short rationale.
- **Docs:** fixed `docs/examples/scripts/data_drift_check.py` f-strings that had no interpolation placeholders (dropped the unnecessary `f` prefix on 3 `print()` calls).

### 🎨 Frontend

- **Lint config:** `frontend/ml-canvas/.eslintrc.cjs` was missing a Node environment override for build/config scripts (`scripts/*.mjs`, `tailwind.config.js`), causing false `no-undef` errors on `process`/`require`. Added a scoped `overrides` entry instead of enabling Node globals for the whole (browser) app.
- **Security review:** reviewed Semgrep "node-ssrf" findings in `src/core/api/datasets.ts` and `src/core/api/eda.ts` — all calls target our own fixed `API_BASE` with encoded/numeric path segments, never an attacker-supplied host or URL; annotated as false positives (`// nosemgrep: node-ssrf`) with rationale.
- **Security review:** reviewed Semgrep "insecure-random-generator" findings in `recentPipelines.ts`, `CausalGraph.tsx`, and `useNotificationsStore.ts` — all uses of `Math.random()` are for non-cryptographic local UI ids or cosmetic layout fallback, not security tokens; annotated as false positives with rationale.

### 🔒 CI / Supply chain

- **GitHub Actions pinning:** pinned all third-party (non-`actions/*`) workflow steps to full-length commit SHAs instead of mutable tags/branches, per Semgrep's "not pinned to commit SHA" finding: `astral-sh/setup-uv` (`ci.yml` ×2, `security.yml`), `codecov/codecov-action` (`ci.yml`), `peaceiris/actions-gh-pages` (`docs.yml`), `pypa/gh-action-pypi-publish` (`release.yml`), and `codacy/codacy-analysis-cli-action@master` (`codacy.yml`, now pinned off the mutable `master` branch).
- **Known accepted risk (no fix available):** Trivy flags two transitive dependency CVEs in `uv.lock` with no upstream fix yet — `ecdsa@0.19.2` (CVE-2024-23342, Minerva attack) and `torch@2.12.0` (CVE-2025-3000). Tracked for future re-scan once upstream patches land; no code change possible today.

### 📦 Versioning

- Bumped root app version `0.6.4` → `0.6.5` (`pyproject.toml`).
- Bumped `skyulf-core` package version `0.1.14` → `0.1.15` (`skyulf-core/skyulf/__init__.py`).
- Synced `frontend/ml-canvas/package.json` to `0.6.5` via `npm run sync-version`.
