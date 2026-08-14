# Testing and CI/CD depth audit

**Audit date:** 2026-08-11  
**Scope:** `backend/`, `skyulf-core/`, `frontend/ml-canvas/`, and `.github/workflows/`. This is a source review and collection audit, not a statement of line coverage.

## Executive assessment

Skyulf has a materially better test suite than raw file ratios suggest: the two Python suites collect **1,177 backend/root tests** and **3,000 core tests**, and the frontend has **92 Vitest files plus four Playwright specs**. There are real engine, tuning, API, and UI behavior tests. However, the safety net is uneven at the boundaries most likely to break during DL/Ray work: no backend coverage threshold, no frontend coverage run/threshold, a skipped machine-specific full-inference test, no real browser-to-FastAPI/Celery/Postgres path, and E2E deliberately avoids React Flow drag/connection behavior.

**Overall risk: High** — ordinary unit regressions are likely to be caught, but a contract, orchestration, deployment, or actual-canvas-interaction failure can reach production.

## 1. Coverage reality check

### Inventory

Counts were generated with `find` (excluding frontend `node_modules` and excluding test/spec files from source counts), then collection was checked with pytest:

| Area | Production source files | Test files | Ratio | Collected tests |
|---|---:|---:|---:|---:|
| Backend (`backend/**/*.py`) | 132 | 115 root `tests/test_*.py` files | 0.87 | 1,177 (`pytest tests --collect-only -q`) |
| Core (`skyulf-core/skyulf/**/*.py`) | 171 | 128 `skyulf-core/tests/test_*.py` files | 0.75 | 3,000 (`pytest skyulf-core/tests --collect-only -q`) |
| Frontend (`frontend/ml-canvas/src/**/*.{ts,tsx}`) | 288 | 92 Vitest files, plus 4 Playwright specs | 0.32 unit-test files/source file | Vitest count not collected in this audit |

File counts are only a triangulation aid: they do **not** demonstrate line/branch coverage. The core CI explicitly requires only 45% coverage (`.github/workflows/skyulf-core-tests.yml:83-86`); backend produces XML but has no `--cov-fail-under` (`.github/workflows/backend-tests.yml:90-99`); frontend configures a reporter but CI runs `npm test`, not `vitest --coverage` (`frontend/ml-canvas/vitest.config.ts:22-27`, `.github/workflows/frontend-tests.yml:82-84`).

### Directly untested or weakly evidenced production areas

The following are conservative findings: a search of all respective test contents found no filename/module-stem reference, and there is no dedicated test file. This proves no *direct* test coverage, not necessarily no incidental execution through a higher-level test.

* **Backend — High for service/engine paths.** `backend/middleware/error_handler.py`, `backend/database/async_registry.py`, `backend/database/async_init_db.py`, `backend/config/environments.py`, `backend/utils/logging_utils.py`, `backend/database/data_sources/async_postgres_queries.py`, `backend/database/data_sources/async_sqlite_queries.py`, `backend/ml_pipeline/_services/job_service.py`, `backend/ml_pipeline/_services/pipeline_versions_service.py`, `backend/ml_pipeline/_execution/engine/_warning_capture.py`, and `backend/config/mixins/llm.py` have no direct test reference. The two untested pipeline service modules are particularly important because the API/job orchestration layer is where training execution meets persistence.
* **Core — Medium.** `skyulf-core/skyulf/modeling/_sklearn_compat.py`, `skyulf-core/skyulf/modeling/hyperparameters/_svm.py`, and `skyulf-core/skyulf/preprocessing/transformations/_power_common.py` have no direct test reference. This is a small gap relative to the large core suite, but compatibility and shared helper code have broad blast radius.
* **Frontend — High for core canvas/model workflows.** 115 source files have no direct test reference. Entirely untested-by-module groups include the 15-file `src/pages/drift/` component directory, the modeling node UI (`modules/nodes/modeling/RegressionNode.tsx`, `TextClassificationNode.tsx`, `EnsembleSettings.tsx`, `SegmentationSettings.tsx`, and the model-editor components), and actual React Flow presentation components (`components/canvas/FlowCanvas.tsx`, `CustomNodeWrapper.tsx`, `CustomEdge.tsx`, `TemplatesGalleryModal.tsx`). `DatasetNode.tsx` also has no direct test. Existing converter tests do not replace tests that a user can configure these controls correctly.

## 2. Test quality: substantive tests exist, but not consistently

### Strong behavioral examples

* The tuning integration test runs imputation, scaling, one-hot encoding, split, tuned fitting, best-parameter retrieval, and held-out prediction on a 300-row fixture (`skyulf-core/tests/test_pipeline_integration_tuning.py:1-10`, `:100-142`). Its grid-search assertion independently recomputes K-fold scores rather than merely trusting the tuner’s result (`:169-210`). This is a high-quality oracle.
* The backend engine test creates a CSV, executes data loading, splitting, training, and tuned training through `PipelineEngine`, then asserts artifacts, metrics, node results, and single-write behavior (`tests/test_execution.py:24-155`). This is a genuine in-process integration test.
* The frontend converter suite checks graph topology, backend step type, parameter forwarding, training mode conversion, and tuning payloads (`frontend/ml-canvas/src/core/utils/pipelineConverter.test.ts:36-60`, `:131-225`). It is important because converter configuration is the frontend/backend boundary.
* API registry tests assert meaningful response properties — known registry items and no duplicate IDs — rather than only HTTP 200 (`tests/test_api_integration.py:16-39`).

### Weak or misleading examples

* `tests/test_core_pipeline.py` is a useful smoke chain, but its oracle is shallow: it only asserts no null ages, one encoded column, near-zero training mean, and equal column counts (`tests/test_core_pipeline.py:12-68`). It would not detect many incorrect values, category handling errors, or leakage defects.
* The nominal “full inference” test is skipped on CI because it requires a hard-coded local Windows workspace (`tests/test_full_inference_pipeline.py:19-25`). Worse, several failure branches print and `return` rather than fail the test (`:165-196`), so it is not a reliable regression test even when manually run.
* Many API tests are stronger than a status-only check, but their service behavior is frequently mocked. For example, ingestion task testing mocks database lookup, connector, and profiler (`tests/test_data_ingestion.py:22-59`). This verifies task wiring and metadata but cannot catch file parsing, connector, DB transaction, or broker integration defects.
* Canvas page tests explicitly mock `MainLayout` and therefore do not render React Flow, sidebar, toolbar, autosave, or real canvas interaction (`frontend/ml-canvas/src/pages/CanvasPage.test.tsx:10-16`). This is a legitimate focused unit test, but it must not be represented as canvas end-to-end coverage.

## 3. Critical-path coverage

| Critical path | Assessment | Severity / effort |
|---|---|---|
| Core preprocessing/training/tuning | **Good in-process coverage.** Dedicated pipeline, engine-parity, modeling, tuning-engine, and tuning-integration test files exist. Tuning validates NaN/Inf and real outcomes (`skyulf-core/tests/test_tuning_engine.py:153-240`). | Medium residual risk / medium effort |
| Backend pipeline execution | **Partial.** `tests/test_execution.py` exercises a real local engine and artifacts, but there is no required full service path using production async DB, Celery worker, Redis, and API request. Test configuration forces Celery memory transport and disables Celery (`tests/conftest.py:9-17`), even though CI starts Redis (`.github/workflows/backend-tests.yml:33-42`). | **High** / high effort |
| Model training/tuning API orchestration | **Partial.** Engine-level tuned flow is present (`tests/test_execution.py:104-155`), but `job_service.py` and `pipeline_versions_service.py` lack direct tests. | **High** / medium effort |
| Authentication/authorization | **Insufficiently evidenced.** There are no dedicated auth-named test files in the collected root suite, while auth dependencies are optional (`pyproject.toml:75-79`). This audit found no authenticated/unauthenticated endpoint matrix or role/ownership boundary test. | **High** / medium effort |
| File upload/dataset ingestion | **Partial.** There are ingestion task, router/security, local-file, S3, and service tests, but the representative task test mocks all external collaborators (`tests/test_data_ingestion.py:22-59`). Add multipart upload → persisted source → background ingestion → schema/preview tests against real temporary storage and DB. | **High** / high effort |
| Frontend canvas run-preview | **Partial mock E2E.** Playwright clicks Run Preview and asserts a mocked POST and rendered mocked rows (`frontend/ml-canvas/e2e/preview.spec.ts:24-124`). It does not test the live API contract. | High / medium effort |

## 4. CI/CD pipeline review

### What runs on pull requests

* **Python static checks:** Ruff lint, Ruff format, and Ty type check (`.github/workflows/pr_check.yml:53-67`).
* **Backend:** pytest with branch coverage report, Redis service, and a Docker test-image smoke job (`.github/workflows/backend-tests.yml:33-42`, `:84-119`).
* **Core:** pytest with a 45% branch-coverage floor, wheel build/install check, and standalone Docker smoke job (`.github/workflows/skyulf-core-tests.yml:75-97`, `:105-126`).
* **Frontend:** ESLint, TypeScript, version check, Vitest, build, bundle budget, and Playwright (`.github/workflows/frontend-tests.yml:48-58`, `:82-127`).
* **Security:** OSV/pyscan dependency scan on PRs and weekly (`.github/workflows/security.yml:3-45`); CodeQL for Python and JavaScript/TypeScript on PRs (`.github/workflows/codeql.yml:21-60`); dependency review only when manifests/locks change, failing at high severity (`.github/workflows/dependency-review.yml:10-34`).

### Gate weaknesses

* Both full-tree Lizard reports are intentionally soft-failing via `|| true`; only selected core directories are hard-gated (`.github/workflows/backend-tests.yml:73-82`, `.github/workflows/skyulf-core-tests.yml:63-73`). This does not let failing tests merge, but it lets complexity regressions outside selected paths merge.
* The Playwright report upload uses `if: always()` (`.github/workflows/frontend-tests.yml:130-136`); this is correctly diagnostic, not a soft test failure. No `continue-on-error` was found in workflow YAML.
* A11y E2E intentionally fails only **critical** Axe violations and merely logs serious violations (`frontend/ml-canvas/e2e/a11y.spec.ts:5-10`, `:50-69`). This is a conscious quality soft gate.
* Workflow files cannot make checks required. Comments say workflows “should be required independently” (`.github/workflows/backend-tests.yml:3-11`), but branch protection/rulesets are GitHub server configuration. GitHub API verification was unavailable in this audit because the configured public-GitHub CLI token is invalid. Therefore, whether a maintainer can merge a failed or skipped workflow is **unverified**, not proven safe.
* No CI test sharding/flaky-test reporting, mutation testing, test-impact selection, coverage diff gate, or release/deployment smoke against a running production-like stack was found.

## 5. Disabled, skipped, and flaky tests

Static search found **55 `pytest.mark.skip/skipif` declarations, one strict xfail, 11 runtime `pytest.skip` calls, and 58 `pytest.importorskip` calls** across the Python test trees; no `it.skip`, `test.skip`, or `describe.skip` was found in frontend tests/E2E. These are declaration/call counts, not a unique skipped-test count (parameterization can expand one marker).

Important cases:

* `tests/compliance_suite.py` accounts for 32 `skipif` uses, mostly Polars availability, but is not discovered by the configured `pytest tests` command because its filename does not match `test_*.py`. It is therefore a misleading apparent test suite unless separately invoked.
* The skipped full-inference test is machine-specific and explicitly “skipped on CI” (`tests/test_full_inference_pipeline.py:22-25`) — **High severity** because it removes the broadest real pipeline/inference safety net.
* Core benchmarks are deliberately opt-in: pytest globally adds `--benchmark-skip` (`pyproject.toml:120-129`), and large benchmark cases require environment flags (`skyulf-core/tests/test_benchmarks.py:82-90`, `:154-157`). This is reasonable for PR speed but means performance regressions have no CI gate.
* Optional dependency skips are mostly intentional and documented: `pytest.importorskip("optuna")` in tuning integration (`skyulf-core/tests/test_pipeline_integration_tuning.py:213-240`), Polars conditionals, and a single **strict** xfail documenting pyarrow integer truncation (`skyulf-core/tests/test_pyarrow_dtypes.py:104-129`). CI does install Optuna, Polars, and PyArrow (`requirements-ci.txt:22-39`), so optionality is less defensible for those CI paths and should be measured in CI output.
* The Playwright configuration retries failed tests twice in CI (`frontend/ml-canvas/playwright.config.ts:15-18`). Retries can hide intermittent behavior; publish retry counts and treat a pass-on-retry as a flaky-test signal.

## 6. Frontend depth and the canvas interaction gap

There is substantial React Testing Library usage, not merely pure utilities: pages, dialogs, results panels, data/EDA components, stores, hooks, API clients, accessibility, and node settings have co-located tests. `CanvasPage.test.tsx` tests accessibility and deep-link behavior with RTL (`frontend/ml-canvas/src/pages/CanvasPage.test.tsx:37-93`), while converter tests provide valuable payload-contract coverage.

There are four Playwright specs: smoke, routes, preview, and accessibility. They boot the actual Vite app and mock every backend request (`frontend/ml-canvas/playwright.config.ts:4-11`, `frontend/ml-canvas/e2e/fixtures/mockApi.ts:3-31`). The smoke spec proves sidebar clicks add nodes, not that dragging works (`frontend/ml-canvas/e2e/smoke.spec.ts:27-44`).

The key gap is explicit: the preview spec seeds Zustand graph state through a development test hook because React Flow connection dragging is considered unreliable in headless Chromium (`frontend/ml-canvas/e2e/preview.spec.ts:11-17`, `:61-109`). Consequently, CI does **not** cover drag-and-drop placement, source-handle-to-target-handle connection, invalid connection prevention, node selection/deletion, or graph persistence using real user gestures. This is **High severity / Medium-high effort** for the product’s central interaction.

## 7. Test data and fixtures

The suite uses a mix of useful and weak data:

* Good: pipeline tuning uses a named 300-row fixture with numeric/categorical values and missingness (`skyulf-core/tests/test_pipeline_integration_tuning.py:3-10`, `:79-94`); pyarrow parity tests compare independent numpy and Arrow-backed paths, including nulls (`skyulf-core/tests/test_pyarrow_dtypes.py:45-101`); API recommendation tests include missing values, duplicates, high cardinality, and an outlier (`tests/test_api_integration.py:98-237`).
* Weak: common core fixtures are deterministic but tiny/simple — 100 random rows and one categorical column (`skyulf-core/tests/conftest.py:20-51`); cross-validation commonly uses 50 rows split to 40/10 (`tests/test_cross_validation_all_methods.py:37-90`); the core pipeline smoke uses four training rows and two test rows (`tests/test_core_pipeline.py:12-28`).
* Operational concern: tests use local files and `tmp_path` well in some places (`tests/test_execution.py:24-35`), but legacy API tests create/delete fixed relative `temp_test_data*` directories (`tests/test_api_integration.py:42-95`). This is less parallel-safe and can leak artifacts after interruption.

Add shared, versioned fixtures for empty datasets, all-null columns, mixed/invalid encodings, duplicate headers, huge/sparse/wide datasets, extreme numeric values, schema drift, multiclass/imbalanced targets, and adversarial upload filenames. Use property-based tests for graph/config validation and preprocessing invariants, not just random happy paths.

## 8. DL/Ray regression safety net

**Conclusion: not strong enough at the integration boundary.** Core calculator/pipeline logic has a credible safety net for existing pandas/Polars preprocessing and classical-model tuning. That will catch many pure-engine regressions. It will not reliably catch Ray scheduling/serialization, distributed artifact writes, worker failure/retry/cancellation, resource limits, process isolation, model serialization, live API job state transitions, or frontend configuration-to-backend contract drift.

DL/Ray nodes will plug into the exact areas with weak end-to-end coverage: `PipelineEngine`, job services, Celery/Redis orchestration, artifacts, and canvas node configuration. Before enabling them, introduce a hermetic integration profile that runs FastAPI + a real worker/broker + temporary database/object store, and a small Ray local-cluster suite. Include one real multi-node graph from canvas payload through API submission to trained artifact/inference, with deterministic tiny DL data and explicit timeout/cancellation assertions.

## Prioritized top five gaps threatening release quality

1. **High / High effort — No production-like end-to-end execution test.** Replace the machine-specific skipped test with required API → DB → broker/worker → artifact → inference coverage; test upload ingestion as part of it.
2. **High / Medium-high effort — No actual canvas drag/connect E2E.** Cover React Flow drag/drop, handle connection, invalid links, deletion, persistence, and Run Preview/Training. Keep the existing state-seeded test as a fast complement, not the only path.
3. **High / Medium effort — No coverage gate for backend or frontend.** Add baseline branch thresholds and changed-lines coverage reporting; raise core’s 45% floor incrementally. Run frontend coverage in CI and fail under a ratchet.
4. **High / Medium effort — Auth and pipeline job-service gaps.** Add authentication/authorization matrices and direct tests for `job_service.py`, `pipeline_versions_service.py`, retry/cancel/ownership, and failure-state persistence.
5. **High / Medium effort — DL/Ray-ready contract and resilience suite absent.** Define node-contract fixtures shared across frontend/backend, then add Ray-local failure/retry/serialization/artifact tests and one real contract E2E before introducing distributed nodes.

