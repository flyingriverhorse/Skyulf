# Skyulf Dual-Track Versioned Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sequence every finding from the `initiatives/` research (enterprise-readiness Phases 0–18, deep-learning, ray-migration, training-visualization, code-escape-hatch) into an ordered ladder of releases, each with an exact, independently-derived semver bump for `backend`, `frontend`, and `skyulf-core`, split into a mandatory **Core/OSS track** and an explicitly optional, trigger-gated **Enterprise track**.

**Architecture:** Skyulf is a monorepo with three independently-versioned artifacts: the FastAPI backend (`pyproject.toml`, name `skyulf`), the React canvas (`frontend/ml-canvas/package.json`), and the standalone PyPI library (`skyulf-core/setup.py`). Backend + frontend usually ship together as one user-facing "app release" but each carries its own version reflecting only what changed inside it; `skyulf-core` releases on its own `core-vX.Y.Z` tag cadence. Track A releases in a strict dependency order that puts cheap, visible, credibility-building work first and defers XL architectural work (partitionable calculators, Ray, DL) until the foundations they need exist. Track B is a separately-packaged, separately-versioned add-on (`skyulf-enterprise`) that never blocks Track A and is not started until a concrete demand trigger fires.

**Tech Stack:** Python 3.12 (backend, runtime floor per `README.md`) / Python 3.11+ idioms (skyulf-core), FastAPI, SQLAlchemy (async + sync), Celery → Ray, Redis, PostgreSQL, S3-compatible storage, pandas + Polars (+ Narwhals later), scikit-learn / XGBoost / LightGBM (+ CatBoost later), PyTorch (optional `dl` extra, later), TypeScript + React + React Flow + Zustand + Vite + Vitest + Playwright, Ruff + `ty` + ESLint + `tsc`.

## Global Constraints

- **Three independent version lines. Starting values, verified in-repo on branch `080`:**
  - `backend` — `pyproject.toml:5` → `version = "0.7.9"`
  - `frontend` — `frontend/ml-canvas/package.json:4` → `"version": "0.7.9"`
  - `skyulf-core` — `skyulf-core/setup.py:10` → `version="0.5.8"`
- **Semver rules applied throughout this plan, no exceptions:**
  - **PATCH** (`x.y.Z+1`) — bug fixes, docs, tests, internal refactors, dependency-pin bumps, and any change with zero new public surface.
  - **MINOR** (`x.Y+1.0`) — new backward-compatible capability: a new node, a new endpoint, a new public function/class, a new UI feature.
  - **MAJOR** (`X+1.0.0`) — any breaking change to a public contract: removed/renamed endpoint or route prefix, newly-required authentication, removed runtime support (SQLite in production), changed calculator/applier base-class signature, removed execution backend (Celery).
  - **Only bump a component that actually changed in that release.** A frontend-only release leaves `pyproject.toml` and `setup.py` untouched. This is explicitly exercised in R5 (frontend-only) and R3 (core + backend, frontend untouched).
  - `skyulf-core` version numbers never mirror backend/frontend numbers. They diverge permanently from R2 onward.
- **OSS-first, Enterprise-optional philosophy.** Track A is the product. Nothing in Track A is gated, license-keyed, or crippled to create upsell pressure. Track B exists only as an additive add-on package for organizations that need org-level identity/governance, and is not started until the trigger in "Track B — Gate" fires. If Track B never starts, Track A is still a complete, coherent product.
- **Licensing stays as it is today** (`COMMERCIAL-LICENSE.md`): backend + frontend AGPLv3, `skyulf-core` Apache-2.0. Track B's `skyulf-enterprise` package is the only new license surface, and only if Track B starts.
- Every Python change: `ruff check .`, `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`, `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`.
- Every frontend change: from `frontend/ml-canvas/` run `npm run lint`, `npx tsc --project tsconfig.json --noEmit`, `npm run build`.
- Every release that adds/renames/removes a node param, an enum/allow-list value, or a node output shape must satisfy the repo-wide **Backend/Core ↔ Frontend Sync Rule** before the release is cut — the frontend node component's hardcoded option arrays are the second half of every such change.
- Every commit includes `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.
- Every release follows `RELEASING.md`: app releases derive their GitHub Release body from `changelog/X.Y.x.md`, so the changelog entry must exist **before** the version bump is pushed; `skyulf-core` publishes on a `core-vX.Y.Z` tag.

---

## Version Ledger (authoritative — no other section may contradict this)

Track A, in release order. `—` means the component is not touched in that release and keeps its previous value.

| # | Release theme | backend | frontend | skyulf-core |
|---|---|---|---|---|
| — | **starting point (branch `080` today)** | 0.7.9 | 0.7.9 | 0.5.9 |
| R1 | Correctness & Honest Positioning | **0.7.10** | **0.7.10** | **0.5.9** |
| R2 | Five-Minute First Run | **0.8.0** | **0.8.0** | **0.6.0** |
| R3 | skyulf-core Stands Alone | **0.8.1** | — (0.8.0) | **0.7.0** |
| R4 | Leakage-Safe by Construction | **0.9.0** | **0.9.0** | **0.8.0** |
| R5 | Shared Frontend Infrastructure | — (0.9.0) | **0.10.0** | — (0.8.0) |
| R6 | Node Coverage Parity & Paper-Backed Nodes | **0.10.0** | **0.11.0** | **0.9.0** |
| R7 | Engine-Agnostic Core, Data Contracts & Forecasting | **0.11.0** | **0.12.0** | **0.10.0** |
| R8 | Transparency & Diagnostics | **0.12.0** | **0.13.0** | — (0.10.0) |
| R9 | Hardening, Contracts & Accessibility | **0.13.0** | **0.14.0** | — (0.10.0) |
| R10 | **v1.0 — Self-Hostable & Authenticated** | **1.0.0** | **1.0.0** | — (0.10.0) |
| R11 | **skyulf-core v1.0 — Partitionable Calculator Contract** | **1.1.0** | — (1.0.0) | **1.0.0** |
| R12 | Ray Migration I (execution backend + attempt lifecycle) | **1.2.0** | **1.1.0** | — (1.0.0) |
| R13 | Ray Migration II (Ray Jobs runtime + distributed compute) | **1.3.0** | — (1.1.0) | — (1.0.0) |
| R14 | Ray Migration III (operations + Celery removal) | **2.0.0** | **1.2.0** | — (1.0.0) |
| R15 | Deep Learning I (infra + tabular MLP) | **2.1.0** | **1.3.0** | **1.1.0** |
| R16 | Deep Learning II (text + time-series) | **2.1.1** | **1.4.0** | **1.2.0** |
| R17 | Deep Learning III (image, GPU via Ray, live curves) | **2.2.0** | **1.5.0** | **1.3.0** |
| R18 | Deployment & Registry Maturity | **2.3.0** | **1.6.0** | **1.4.0** |
| R19 | Code-First Loop & Conditional i18n | **2.4.0** | **1.7.0** | **1.5.0** |

Track B (optional) uses a **fourth, separate version line** — see "Track B — Packaging & Versioning Model". It never consumes a number from the table above.

---

## Global Procedure: Release Cut

Every Track A release ends with this procedure. It is written once here with real commands; each release's "Release cut" task supplies the literal parameter values from its own table, so nothing below is a placeholder.

**Parameters per release:** `BACKEND_VER`, `FRONTEND_VER`, `CORE_VER`, `SERIES_FILE` (= `changelog/<backend major>.<backend minor>.x.md`), `TITLE`.

- [ ] **Step A: Write the changelog entry**

Create the series file if it does not exist, then add the new block at the top of `SERIES_FILE`:

```markdown
## v0.7.10 — Correctness & Honest Positioning

### 🐛 Bug Fixes
- **Lag/Rolling target alignment:** `y` is now reordered and filtered in lockstep with `X`.

### 📚 Docs
- **README:** leakage-safe-by-construction positioning.
```

(Substitute the release's own `TITLE`, version header, and bullets. The header must match `pyproject.toml`'s version exactly or `release-drafter.yml` fails — see `RELEASING.md` troubleshooting table.)

- [ ] **Step B: Bump only the components that changed**

Backend (`pyproject.toml`, line 5):

```toml
[project]
version = "0.7.10"
```

Frontend (`frontend/ml-canvas/package.json`, line 4):

```json
  "version": "0.7.10",
```

skyulf-core (`skyulf-core/setup.py`, line 10):

```python
    version="0.5.9",
```

- [ ] **Step C: Verify the three versions agree with this plan's Version Ledger**

```bash
cd /Users/BH7043/Skyulf
grep -m1 '^version' pyproject.toml
grep -m1 '"version"' frontend/ml-canvas/package.json
grep -m1 'version=' skyulf-core/setup.py
```

Expected: exactly the row for this release in the Version Ledger; any component marked `—` must still print its previous value.

- [ ] **Step D: Run the full gate**

```bash
cd /Users/BH7043/Skyulf
ruff check . && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py
ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
python -m pytest tests skyulf-core/tests -q
cd frontend/ml-canvas && npm run lint && npx tsc --project tsconfig.json --noEmit && npm run build
```

Expected: all green. A release is not cut while any of these fail.

- [ ] **Step E: Commit and push**

```bash
git add pyproject.toml frontend/ml-canvas/package.json skyulf-core/setup.py changelog/
git commit -m "chore: release backend v0.7.10, frontend v0.7.10, skyulf-core v0.5.9

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push
```

- [ ] **Step F: Tag skyulf-core (only when `CORE_VER` changed)**

```bash
git tag core-v0.5.9
git push origin core-v0.5.9
```

Expected: `.github/workflows/release.yml` builds and publishes to PyPI via Trusted Publishing.

- [ ] **Step G: Publish the drafted GitHub Release**

The push in Step E touched `pyproject.toml`, so `release-drafter.yml` produces a draft within ~30s. Review the body, then publish.

---

# TRACK A — Core/OSS Roadmap (the product)

Track A is not optional and is not gated behind any commercial consideration. Its ordering principle, in priority order:

1. **Fix what is provably broken** (silent training-result corruption ships today).
2. **Make the first five minutes work** (a 47-star project's scarcest resource is a visitor's attention).
3. **Make `skyulf-core` genuinely good standalone** (the PyPI library is the cheapest distribution channel available and is Apache-2.0, so it has no adoption friction).
4. **Turn the one unclaimed positioning asset into a real feature** (leakage-safe by construction — Phase 17b confirms no competitor markets this).
5. **Then** breadth (nodes), **then** architecture (engine-agnostic, partitionable), **then** the XL bets (Ray, DL).

---

## Release R1 — Correctness & Honest Positioning

**Versions:** backend `0.7.10` (patch) · frontend `0.7.10` (patch) · skyulf-core `0.5.9` (patch)
**Effort:** ~5–7 days
**Draws from:** master-fix-list Phase 12 (all 9 confirmed bugs), Phase 16a (`.env.example` gap), Phase 16d (core packaging metadata, docs discoverability), Phase 17b (positioning/messaging items 1 and 3).
**Why first:** Phase 12 items 1–3 can silently corrupt training results. Nothing else on this roadmap is worth shipping on top of a platform that misaligns `X` and `y`. Everything else in R1 is free (docs/messaging), which is exactly the right shape of work for a zero-budget, credibility-building first release.
**Dependencies:** none. This release blocks nothing and is blocked by nothing.

### Task R1.1: Execute the existing Phase 12 bug-fix plan

**Files:**
- Follow: `docs/superpowers/plans/2026-08-11-phase12-confirmed-bugs.md` (already written, 9 tasks, failing-test-first)
- Modify: `skyulf-core/skyulf/preprocessing/time_series/lag.py`, `skyulf-core/skyulf/preprocessing/time_series/` rolling aggregate module, `skyulf-core/skyulf/preprocessing/feature_selection/`, `skyulf-core/skyulf/preprocessing/bucketing.py`, `skyulf-core/skyulf/preprocessing/feature_generation/generation.py`
- Modify: `backend/ml_pipeline/_internal/_routers/run_pipeline.py` (cross-process idempotency), pipeline graph validation (cycle rejection)
- Modify: `frontend/ml-canvas/src/pages/Jobs.tsx` + the job-polling client (out-of-order responses), upload size-limit copy
- Test: `skyulf-core/tests/test_time_series_nodes.py`, plus the per-task test files named in that plan

**Interfaces:**
- Consumes: nothing new.
- Produces: aligned `(X, y)` return contract from `LagFeaturesApplier.apply()` and the rolling-aggregate applier; a cycle-rejecting validation error raised at pipeline-validation time rather than execution time.

**Effort:** 3–4 days. **Version impact:** backend patch, frontend patch, core patch.

- [ ] **Step 1: Execute `2026-08-11-phase12-confirmed-bugs.md` tasks 1–9 in order**, each with its failing test first, as written in that document.
- [ ] **Step 2: Confirm no node param schema changed** — none of the 9 fixes adds/renames a param, enum value, or output column, so the Backend/Core ↔ Frontend Sync Rule requires no frontend node-component change. Verify with:

```bash
cd /Users/BH7043/Skyulf && git diff --stat master -- frontend/ml-canvas/src/modules/nodes/
```

Expected: no changes under `modules/nodes/` (only `pages/Jobs.tsx` and upload copy changed).
- [ ] **Step 3: Run the core and backend suites**

```bash
cd /Users/BH7043/Skyulf && python -m pytest skyulf-core/tests tests -q
```

Expected: PASS, with the 9 new regression tests present.

### Task R1.2: Complete `.env.example`

**Files:**
- Modify: `.env.example`
- Reference: `backend/config/mixins/aws.py` (the AWS/S3 settings class the app actually reads but the example file omits)

**Interfaces:**
- Consumes: the field names declared on every mixin under `backend/config/mixins/`.
- Produces: an `.env.example` that is a complete superset of every environment variable the app reads.

**Effort:** 2 hours. **Version impact:** backend patch (part of 0.7.10).

- [ ] **Step 1: Enumerate every settings field the app reads**

```bash
cd /Users/BH7043/Skyulf && grep -rn "class .*Mixin" backend/config/mixins/ && grep -rn ": str\|: int\|: bool" backend/config/mixins/aws.py
```

- [ ] **Step 2: Add every missing variable to `.env.example`** with a commented default and a one-line description, grouped by mixin (`# --- AWS / S3 ---`, etc.).
- [ ] **Step 3: Verify completeness** — every field name printed in Step 1 appears in `.env.example`:

```bash
cd /Users/BH7043/Skyulf && for v in $(grep -rhoE '^\s{4}[a-z_]+:' backend/config/mixins/*.py | tr -d ' :'); do grep -qi "$v" .env.example || echo "MISSING: $v"; done
```

Expected: no `MISSING:` lines.

### Task R1.3: skyulf-core packaging metadata and docs discoverability

**Files:**
- Modify: `skyulf-core/pyproject.toml` (currently ~9 lines, all dynamic)
- Modify: `skyulf-core/setup.py` (Changelog URL currently points at GitHub Releases, not the changelog file)
- Modify: `skyulf-core/README.md` (add a link to `docs/user_guide/extending_custom_nodes.md` and a "Use skyulf-core without the canvas" section documenting that `@NodeRegistry.register`/`@node_meta` registration is a decorator side-effect and standalone use works today)

**Interfaces:**
- Consumes: existing `setup.py` metadata.
- Produces: PyPI classifiers, `project.urls` (Homepage/Documentation/Changelog/Issues), long-description content type, and a documented standalone-usage entry point.

**Effort:** 3 hours. **Version impact:** core patch (part of 0.5.9).

- [ ] **Step 1: Move static metadata into `skyulf-core/pyproject.toml`** — `classifiers` (Development Status, Intended Audience, License :: OSI Approved :: Apache Software License, Programming Language :: Python :: 3.11/3.12, Topic :: Scientific/Engineering :: Artificial Intelligence), `keywords`, `project.urls`.
- [ ] **Step 2: Point the Changelog URL at the real file** in `skyulf-core/setup.py`:

```python
    project_urls={
        "Changelog": "https://github.com/flyingriverhorse/Skyulf/blob/master/CHANGELOG.md",
        "Documentation": "https://flyingriverhorse.github.io/Skyulf/",
        "Issues": "https://github.com/flyingriverhorse/Skyulf/issues",
    },
```

- [ ] **Step 3: Verify the built distribution carries the metadata**

```bash
cd /Users/BH7043/Skyulf/skyulf-core && python -m build --sdist --wheel && python -m twine check dist/*
```

Expected: `Checking dist/... PASSED`.

### Task R1.4: Positioning rewrite (README + skyulf-core README)

**Files:**
- Modify: `README.md` (headline, "What is Skyulf", new "Why Skyulf is different" section)
- Modify: `skyulf-core/README.md` (new "Leakage-safe by construction" and "Your artifacts are JSON, not pickles" sections)

**Interfaces:**
- Consumes: the already-true architectural facts — calculator/applier fit/apply split (`skyulf-core/skyulf/preprocessing/base.py`), JSON artifacts, `skyulf-core/skyulf/leakage.py`'s static leakage diagnostics, and the existing notebook export (`backend/ml_pipeline/_internal/_routers/notebook_export.py`).
- Produces: the messaging that R4 then makes literally enforceable in code.

**Effort:** 4 hours. **Version impact:** docs only; ships inside 0.7.10 / 0.5.9. See the full copy in the "Positioning" section at the end of this plan.

- [ ] **Step 1: Replace the README one-liner and lead paragraph** with the copy in the Positioning section below.
- [ ] **Step 2: Add the "Leakage-safe by construction" section to `skyulf-core/README.md`** with the runnable snippet given in the Positioning section.
- [ ] **Step 3: Verify every claim in the new copy is true today** — run the snippet:

```bash
cd /Users/BH7043/Skyulf/skyulf-core && python -c "
from skyulf.preprocessing.scaling.standard import StandardScalerCalculator
import pandas as pd, json
art = StandardScalerCalculator().fit(pd.DataFrame({'a':[1.0,2.0,3.0]}), {'columns':['a']})
print(json.dumps(art)[:200])
"
```

Expected: the artifact prints as JSON — this is the literal evidence for the "JSON, not pickles" claim. If it does not, soften the claim rather than shipping an untrue one.

### Task R1.5: Release cut R1

**Files:**
- Create: `changelog/0.7.x.md` entry `## v0.7.10 — Correctness & Honest Positioning` (series file already exists)
- Modify: `pyproject.toml`, `frontend/ml-canvas/package.json`, `skyulf-core/setup.py`

**Interfaces:** Consumes: R1.1–R1.4. Produces: published backend 0.7.10, frontend 0.7.10, PyPI `skyulf-core` 0.5.9.

**Effort:** 1 hour. **Version impact:** backend `0.7.9`→`0.7.10`, frontend `0.7.9`→`0.7.10`, core `0.5.8`→`0.5.9`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=0.7.10`, `FRONTEND_VER=0.7.10`, `CORE_VER=0.5.9`, `SERIES_FILE=changelog/0.7.x.md`, `TITLE=Correctness & Honest Positioning`. Step F applies (core changed).

**Gate:** all 9 bug-hunt repro steps in `initiatives/enterprise-readiness/2026-08-11-bug-hunt.md` fail to reproduce, and `python -m twine check dist/*` passes for skyulf-core.

---

## Release R2 — Five-Minute First Run

**Versions:** backend `0.8.0` (minor) · frontend `0.8.0` (minor) · skyulf-core `0.6.0` (minor)
**Effort:** ~2 weeks
**Draws from:** master-fix-list Phase 8 (all 7 quick wins), `smooth-experience-fixes.md` Top 3 (#1 sample-data baseline, #2 post-upload pipeline recommendation, #3 plain-English recovery), Phase 9 differentiation Bet #2, Phase 16b (test-only sample-dataset loader not exposed to users).
**Why second:** every item here is cheap because the underlying asset already exists in the repo and is simply not reachable — sample CSVs at `skyulf-core/examples/data/*/`, the profiler's recommendations at `skyulf-core/skyulf/profiling/_analyzer/recommendations.py`, the WebSocket connection-state callback `jobEventsSocket.onStatus`, the unused `Skeleton` component, the shared `ModalShell`. This is the highest visible-value-per-hour release on the entire roadmap.
**Dependencies:** R1 (do not demo a first-run flow that trains a misaligned model).

### Task R2.1: `skyulf.datasets` public sample-data API

**Files:**
- Create: `skyulf-core/skyulf/datasets/__init__.py` — `load_sample(name: str) -> tuple[SkyulfDataFrame, str]`, `list_samples() -> list[SampleDatasetInfo]`
- Create: `skyulf-core/skyulf/datasets/_registry.py` — `SampleDatasetInfo` TypedDict (`name`, `path`, `target_column`, `task`, `n_rows`, `description`, `license`)
- Modify: `skyulf-core/setup.py` / `MANIFEST.in` — ship `examples/data/*/*.csv` as package data
- Reference (do not delete): `skyulf-core/tests/utils/dataset_loader.py` — the existing test-only loader whose logic this promotes to public API
- Test: `skyulf-core/tests/test_datasets.py`

**Interfaces:**
- Consumes: existing CSVs at `skyulf-core/examples/data/online_retail/online_retail_sample.csv`, `.../credit_card_fraud/creditcard_sample.csv`, `.../santander/train_sample.csv`.
- Produces: `skyulf.datasets.load_sample(name) -> (df, target_column)` and `skyulf.datasets.list_samples() -> list[SampleDatasetInfo]`, consumed by Task R2.2's backend endpoint.

**Effort:** 1 day. **Version impact:** core minor (new public module).

- [ ] **Step 1: Write the failing test**

```python
def test_load_sample_returns_frame_and_target():
    df, target = load_sample("credit_card_fraud")
    assert target == "Class"
    assert len(df) > 0
```

- [ ] **Step 2: Run it** — `cd skyulf-core && python -m pytest tests/test_datasets.py -v`. Expected: FAIL, `ModuleNotFoundError: skyulf.datasets`.
- [ ] **Step 3: Implement `_registry.py` with one `SampleDatasetInfo` entry per shipped CSV**, and `load_sample` reading via `importlib.resources.files("skyulf") / ...` so it works from an installed wheel, not just a source checkout.
- [ ] **Step 4: Verify from a built wheel, not the source tree**

```bash
cd /Users/BH7043/Skyulf/skyulf-core && python -m build --wheel && python -m venv .wheeltest && .wheeltest/bin/pip install -q dist/*.whl && .wheeltest/bin/python -c "from skyulf.datasets import list_samples; print([s['name'] for s in list_samples()])"
```

Expected: prints all sample names. (Package-data omissions only show up this way.)

### Task R2.2: Sample datasets + baseline recommendation endpoints

**Files:**
- Create: `backend/data_ingestion/_routers/samples.py` — `GET /api/samples`, `POST /api/samples/{name}/import`
- Create: `backend/ml_pipeline/_internal/_routers/recommend_pipeline.py` — `POST /api/pipelines/recommend`
- Create: `skyulf-core/skyulf/profiling/baseline.py` — `build_baseline_pipeline(profile: DatasetProfile, target: str, task: str) -> PipelineConfig`
- Modify: `backend/main.py` (mount both routers)
- Test: `tests/test_samples_router.py`, `skyulf-core/tests/test_baseline_pipeline.py`

**Interfaces:**
- Consumes: `skyulf.datasets.list_samples()` (R2.1); the existing `EDAAnalyzer` and `skyulf-core/skyulf/profiling/_analyzer/recommendations.py` heuristics (missingness, imbalance, high-cardinality, skew).
- Produces: `build_baseline_pipeline(...) -> PipelineConfig` (the same `PipelineConfig` shape `skyulf/types.py` already defines, so the canvas can load it unchanged); `POST /api/pipelines/recommend` returning `{"pipeline": PipelineConfig, "rationale": [{"node_id": str, "reason": str}]}`.

**Effort:** 4 days. **Version impact:** backend minor (2 new routers), core minor (new public function).

- [ ] **Step 1: Write the failing core test** asserting that a profile with a 40%-missing numeric column and a 95:5 imbalanced target yields a pipeline containing an imputation step for that column and a resampling or `class_weight` choice, each with a non-empty `rationale` entry.
- [ ] **Step 2: Run it.** Expected: FAIL, `ImportError: cannot import name 'build_baseline_pipeline'`.
- [ ] **Step 3: Implement `build_baseline_pipeline`** as a pure function over the already-computed recommendation objects — it must not re-analyse the data, only assemble existing signals into ordered `PipelineConfig` steps.
- [ ] **Step 4: Implement the two routers**, with `POST /api/samples/{name}/import` writing the CSV through the existing ingestion service so a sample dataset becomes an ordinary `DataSource` with no special-casing downstream.
- [ ] **Step 5: Verify the round trip** — `POST /api/samples/credit_card_fraud/import` then `POST /api/pipelines/recommend` with the returned dataset id, and assert the returned `PipelineConfig` passes existing pipeline validation:

```bash
cd /Users/BH7043/Skyulf && python -m pytest tests/test_samples_router.py -v
```

### Task R2.3: First-run UI — sample data, bound template, plain-English failures

**Files:**
- Modify: `frontend/ml-canvas/src/pages/DataSources.tsx` and the `AddSourceModal` component (add a "Load sample dataset" option backed by `GET /api/samples`)
- Modify: `frontend/ml-canvas/src/core/.../pipelineTemplates.ts` (bind one starter template to a sample dataset id + target column)
- Modify: the templates gallery modal (surface the bound template as "Run this in one click")
- Modify: `frontend/ml-canvas/src/**/useRunControls.ts` (replace `"Pipeline execution failed — Check console for details"` with what-happened / affected-node / suggested-fix / retry, with raw logs behind a "Technical details" toggle)
- Modify: `frontend/ml-canvas/src/pages/Dashboard.tsx`, `Jobs.tsx`, and `JobListSidebar.tsx` empty states (distinguish "no data yet" from "no results for this filter", each with a next-action CTA)
- Test: `frontend/ml-canvas/src/pages/DataSources.test.tsx`, `Jobs.test.tsx`, plus a Playwright spec `sample-to-run.spec.ts` covering load-sample → template → run

**Interfaces:**
- Consumes: `GET /api/samples`, `POST /api/samples/{name}/import`, `POST /api/pipelines/recommend` (R2.2).
- Produces: a `SampleDatasetOption` type in the frontend API client mirroring `SampleDatasetInfo`'s fields exactly (`name`, `target_column`, `task`, `n_rows`, `description`).

**Effort:** 4 days. **Version impact:** frontend minor.

- [ ] **Step 1: Write the Playwright spec first** — `sample-to-run.spec.ts`: open Data Sources → "Load sample dataset" → pick `credit_card_fraud` → open Templates → the bound template is enabled without manual binding → Run → a job appears in Jobs. This spec must drive real UI interactions, not the dev seeding hook (per `testing-ci-audit.md`).
- [ ] **Step 2: Run it.** Expected: FAIL at the "Load sample dataset" locator.
- [ ] **Step 3: Implement the modal option, the template binding, and the empty-state/error copy changes.**
- [ ] **Step 4: Run lint/type/build and the spec**

```bash
cd /Users/BH7043/Skyulf/frontend/ml-canvas && npm run lint && npx tsc --project tsconfig.json --noEmit && npx playwright test sample-to-run.spec.ts
```

### Task R2.4: Perceived-quality quick wins

**Files:**
- Modify: the WebSocket client consumer to render a "Live / Reconnecting" indicator from the already-existing, currently-unused `jobEventsSocket.onStatus` callback
- Modify: `frontend/ml-canvas/src/pages/DataSources.tsx` (success toasts for delete-dataset and create-data-source)
- Modify: `frontend/ml-canvas/src/pages/InferencePage.tsx` (debounce `inputData` ~250ms into one parsed value shared by `analyseInput`, `schemaCheck`, `parsedInputRows` — currently three separate `JSON.parse` calls per keystroke)
- Modify: `BestParamsModal` → port onto the shared `ModalShell` (removes a keyboard-nav dead spot)
- Modify: node-config undo handling in `useGraphStore` (coalesce per-keystroke entries into one undo step per field-edit burst)
- Delete or adopt: the unused `Skeleton` component (decide here; R5 owns the broader loading-state standardisation)

**Interfaces:**
- Consumes: existing `jobEventsSocket.onStatus(status: "live" | "reconnecting" | "closed")`, existing `ModalShell` props.
- Produces: no new public interface.

**Effort:** 3 days. **Version impact:** frontend minor (folded into 0.8.0).

- [ ] **Step 1: Add a Vitest test** asserting that typing 10 characters into the inference textarea triggers exactly one `JSON.parse`-driven recomputation after the debounce window, not 30.
- [ ] **Step 2: Run it.** Expected: FAIL (currently 3 parses per keystroke).
- [ ] **Step 3: Implement the debounce and the remaining five items.**
- [ ] **Step 4: Verify no stale `eslint-disable` comments remain** (`npm run lint` runs with `--max-warnings 0`, so an unused disable comment is itself an error).

### Task R2.5: Release cut R2

**Files:** Create `changelog/0.8.x.md` with `## v0.8.0 — Five-Minute First Run`; modify all three version files.

**Interfaces:** Consumes R2.1–R2.4. Produces backend 0.8.0, frontend 0.8.0, PyPI `skyulf-core` 0.6.0.

**Effort:** 1 hour. **Version impact:** backend `0.7.10`→`0.8.0`, frontend `0.7.10`→`0.8.0`, core `0.5.9`→`0.6.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=0.8.0`, `FRONTEND_VER=0.8.0`, `CORE_VER=0.6.0`, `SERIES_FILE=changelog/0.8.x.md`, `TITLE=Five-Minute First Run`. Step F applies. Note `changelog/0.8.x.md` must be **created** — the drafter fails if the series file is missing.

**Gate:** a first-time visitor can go from a clean checkout to a completed training job in under five minutes without supplying their own data, verified by the `sample-to-run.spec.ts` Playwright run.

---

## Release R3 — skyulf-core Stands Alone

**Versions:** backend `0.8.1` (patch) · frontend **unchanged (0.8.0)** · skyulf-core `0.7.0` (minor)
**Effort:** ~2 weeks
**Draws from:** master-fix-list Phase 16b (all 6 DX items), Phase 16d (docstring coverage), Phase 17b ranked item #3 (standalone code-first usage), the "What NOT to do" caution about the sklearn adapter.
**Why here:** `skyulf-core` is Apache-2.0 and on PyPI — it is the only part of Skyulf a stranger can adopt with zero commitment. Its two adoption blockers are that a Calculator cannot be dropped into `sklearn.pipeline.Pipeline` (raises a confusing `TypeError`) and that bad config silently produces a `{}` artifact instead of an actionable error.
**Dependencies:** R1 (the shared validation helper generalises the Phase 12 #7/#8 fixes — do not write a second, divergent fix).
**This release deliberately does not touch the frontend.** `frontend/ml-canvas/package.json` stays at `0.8.0`.

### Task R3.1: One shared config-validation helper (kills the silent-no-op bug class)

**Files:**
- Modify: `skyulf-core/skyulf/config_validation.py` — add `require_config(config, *, node: str, required: Sequence[str], allowed: Mapping[str, Collection[Any]] | None = None) -> None`
- Modify: every calculator whose Phase 12 / Phase 16b evidence shows a silent `{}` return — starting with `preprocessing/feature_selection/`, `preprocessing/bucketing.py`, `preprocessing/feature_generation/generation.py`
- Test: `skyulf-core/tests/test_config_validation.py`, plus a registry-wide test in `skyulf-core/tests/test_node_contracts.py`

**Interfaces:**
- Consumes: existing `skyulf.config_validation` Pydantic models and `difflib.get_close_matches` (already imported there — reuse it for "did you mean" suggestions).
- Produces: `require_config(...)` raising `SkyulfConfigError(node=..., param=..., got=..., expected=..., suggestion=...)`, used by every subsequent node added in R6/R7/R15–R17.

**Effort:** 4 days. **Version impact:** core minor — this changes observable behaviour (raise instead of no-op) across many nodes, which is more than a patch even though each individual instance is a bug.

- [ ] **Step 1: Write the failing test**

```python
def test_unknown_method_raises_with_suggestion():
    with pytest.raises(SkyulfConfigError) as exc:
        require_config({"method": "uniformm"}, node="GeneralBinning",
                       required=["method"], allowed={"method": {"uniform", "quantile", "kmeans"}})
    assert "uniform" in str(exc.value)
```

- [ ] **Step 2: Run it.** Expected: FAIL, `cannot import name 'require_config'`.
- [ ] **Step 3: Implement `require_config` and `SkyulfConfigError`**, then adopt it in the three node families above.
- [ ] **Step 4: Add a registry-wide contract test** asserting that for every node in `NodeRegistry`, calling `fit` with `{"__nonsense__": 1}` raises `SkyulfConfigError` rather than returning `{}`. Mark nodes not yet migrated with an explicit `xfail(strict=True)` list so the remaining work is visible and shrinking, never silently skipped.
- [ ] **Step 5: Cross-check the frontend** — this changes *error behaviour*, not the set of valid values, so per the Backend/Core ↔ Frontend Sync Rule no node component changes. Confirm the backend surfaces `SkyulfConfigError.suggestion` in its error payload so R2.3's plain-English error UI can use it.

### Task R3.2: sklearn `BaseEstimator`/`TransformerMixin` adapter

**Files:**
- Create: `skyulf-core/skyulf/interop/sklearn_adapter.py` — `as_sklearn(calculator, applier, config) -> SkyulfTransformer`
- Modify: `skyulf-core/skyulf/interop/__init__.py`
- Test: `skyulf-core/tests/test_sklearn_adapter.py`
- Reference (do not duplicate): `skyulf-core/skyulf/modeling/_sklearn_compat.py`, `skyulf-core/skyulf/modeling/sklearn_wrapper.py`, `skyulf-core/skyulf/engines/sklearn_bridge.py` — existing sklearn-facing code to reuse rather than reimplement

**Interfaces:**
- Consumes: the existing `Calculator.fit(data, config) -> artifact` / `Applier.apply(data, artifact) -> data` contract.
- Produces: `SkyulfTransformer(BaseEstimator, TransformerMixin)` with `fit(X, y=None)`, `transform(X)`, `get_params(deep=True)`, `set_params(**params)`, and `fitted_artifact_` exposed as a public attribute so users can inspect/serialise the JSON artifact from inside an sklearn pipeline.

**Effort:** 3 days. **Version impact:** core minor (new public module).

- [ ] **Step 1: Write the failing test**

```python
def test_calculator_runs_inside_sklearn_pipeline():
    pipe = Pipeline([("scale", as_sklearn(StandardScalerCalculator(), StandardScalerApplier(), {"columns": ["a"]})),
                     ("clf", LogisticRegression())])
    pipe.fit(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}), [0, 0, 1, 1])
    assert pipe.named_steps["scale"].fitted_artifact_ != {}
```

- [ ] **Step 2: Run it.** Expected: FAIL with the confusing `TypeError` documented in `core-dx-improvements.md`.
- [ ] **Step 3: Implement one thin adapter class** — per the master fix list's explicit "What NOT to do", do **not** subclass ~50+ calculators individually.
- [ ] **Step 4: Verify sklearn's own estimator contract**

```bash
cd /Users/BH7043/Skyulf/skyulf-core && python -c "
from sklearn.utils.estimator_checks import check_estimator
from skyulf.interop import as_sklearn
" && python -m pytest tests/test_sklearn_adapter.py -v
```

Expected: PASS, including `clone()` round-tripping via `get_params`/`set_params`.

### Task R3.3: TypedDict config schemas + docstring coverage

**Files:**
- Create: `skyulf-core/skyulf/preprocessing/scaling/_config.py`, `.../imputation/_config.py`, `.../encoding/_config.py` — `TypedDict` input schemas per node family, mirroring the existing output-artifact TypedDict pattern
- Modify: `skyulf-core/skyulf/preprocessing/imputation/simple.py`, `knn.py`, `iterative.py`, `outliers/iqr.py`, `zscore.py` — add the missing docstrings (these five are at 0% per `core-docs-onboarding.md`)
- Modify: `skyulf-core/README.md` — link `docs/user_guide/extending_custom_nodes.md`
- Test: `skyulf-core/tests/test_config_typeddicts.py`

**Interfaces:**
- Consumes: existing artifact-TypedDict conventions in `skyulf-core/skyulf/preprocessing/_artifacts.py`.
- Produces: `StandardScalerConfig`, `SimpleImputerConfig`, `OneHotEncoderConfig` (etc.) exported from each family's `_config.py`, consumed by R7's schema-driven form work and by R11's declarative validation.

**Effort:** 3 days. **Version impact:** core minor (new exported types).

- [ ] **Step 1: Add the TypedDicts for the three families above**, each field annotated and `total=False` for optionals.
- [ ] **Step 2: Add a test asserting every key in each TypedDict is actually read by its calculator** (grep the module for `config.get("<key>"` / `config["<key>"`), so the types cannot drift from the implementation.
- [ ] **Step 3: Fill the five zero-docstring files** — one to two lines per function/method, per the repo-wide documentation rule.
- [ ] **Step 4: Verify docstring coverage improved**

```bash
cd /Users/BH7043/Skyulf/skyulf-core && python -c "
import ast, pathlib
for f in ['skyulf/preprocessing/imputation/simple.py','skyulf/preprocessing/imputation/knn.py','skyulf/preprocessing/imputation/iterative.py','skyulf/preprocessing/outliers/iqr.py','skyulf/preprocessing/outliers/zscore.py']:
    t = ast.parse(pathlib.Path(f).read_text())
    fns = [n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    missing = [n.name for n in fns if not ast.get_docstring(n)]
    print(f, 'missing:', missing)
"
```

Expected: `missing: []` for all five.

### Task R3.4: Release cut R3

**Files:** Add `## v0.8.1 — skyulf-core Stands Alone` to `changelog/0.8.x.md`; modify `pyproject.toml` and `skyulf-core/setup.py` **only**.

**Interfaces:** Consumes R3.1–R3.3. Produces backend 0.8.1, PyPI `skyulf-core` 0.7.0, frontend untouched.

**Effort:** 1 hour. **Version impact:** backend `0.8.0`→`0.8.1` (patch: re-pin core, surface `SkyulfConfigError.suggestion`), frontend **stays 0.8.0**, core `0.6.0`→`0.7.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=0.8.1`, `FRONTEND_VER=` *(skip — do not edit `package.json`)*, `CORE_VER=0.7.0`, `SERIES_FILE=changelog/0.8.x.md`, `TITLE=skyulf-core Stands Alone`. In Step C, `frontend/ml-canvas/package.json` must still print `0.8.0`. Step F applies.

**Gate:** `pip install skyulf-core==0.7.0` in a clean venv, then run a `sklearn.pipeline.Pipeline` containing a Skyulf transformer end-to-end, and confirm a deliberately-misspelled config parameter raises a named error with a suggestion.

---

## Release R4 — Leakage-Safe by Construction

**Versions:** backend `0.9.0` (minor) · frontend `0.9.0` (minor) · skyulf-core `0.8.0` (minor)
**Effort:** ~3 weeks
**Draws from:** Phase 17b ranked items #1 (artifact diffability) and #2 (default-on train/test row-overlap detection), Phase 9 foundational item "versioned artifact schema/migration path", Phase 9 "universal calculator contract tests", Phase 9 Bet #1 first increment (surface the *already-computed* server-side leakage/correlation checks as real-time canvas warnings).
**Why here:** R1.4 wrote the positioning claim. This release makes it literally true and enforceable, which is the difference between marketing and a product. Phase 17b's research found **no competitor library markets leakage-safety as a headline feature** — that is the single strongest unclaimed position available, and it costs Medium effort because the artifact structure that makes it possible already exists.
**Dependencies:** R3 (`require_config` and the artifact TypedDicts are what the version stamp and diff hang off). Artifact versioning must land **before** R6 adds ~12 new node types and long before DL (R15+) — every new node type created without it is another artifact that can silently break on a core upgrade.

### Task R4.1: `artifact_schema_version` + migration registry

**Files:**
- Create: `skyulf-core/skyulf/artifacts/__init__.py` — `stamp(artifact, node_type) -> dict`, `migrate(artifact, node_type) -> dict`, `CURRENT_SCHEMA_VERSIONS: dict[str, int]`
- Create: `skyulf-core/skyulf/artifacts/_migrations.py` — `register_migration(node_type, from_version, fn)` and the (initially empty) migration table
- Modify: `skyulf-core/skyulf/preprocessing/base.py` — `StatefulTransformer` stamps on fit and migrates on load
- Test: `skyulf-core/tests/test_artifact_versioning.py`

**Interfaces:**
- Consumes: the existing JSON artifact dicts produced by every calculator.
- Produces: every artifact gains `"__schema__": {"node_type": str, "version": int}`; `migrate()` is the single entry point every artifact loader (including `backend/`'s S3/local loaders) calls before use.

**Effort:** 5 days. **Version impact:** core minor (additive key + new public module).

- [ ] **Step 1: Write the failing test** asserting that a v1 artifact with a registered `1 → 2` migration loads as v2, and that an artifact with a version *newer* than `CURRENT_SCHEMA_VERSIONS` raises a clear "artifact was produced by a newer skyulf-core" error rather than silently misbehaving.
- [ ] **Step 2: Run it.** Expected: FAIL, `ModuleNotFoundError: skyulf.artifacts`.
- [ ] **Step 3: Implement stamp/migrate and wire them into `StatefulTransformer`.**
- [ ] **Step 4: Add a backward-compatibility test** loading an unstamped (pre-0.8.0) artifact fixture and asserting it is treated as version 1 and migrates cleanly — existing users' saved artifacts must not break.

### Task R4.2: `skyulf.artifacts.diff()` — artifact diffability as a feature

**Files:**
- Create: `skyulf-core/skyulf/artifacts/_diff.py` — `diff(a: dict, b: dict) -> ArtifactDiff`, `format_diff(d: ArtifactDiff) -> str`
- Test: `skyulf-core/tests/test_artifact_diff.py`

**Interfaces:**
- Consumes: stamped artifacts from R4.1.
- Produces: `ArtifactDiff` TypedDict (`added: list[str]`, `removed: list[str]`, `changed: list[tuple[str, Any, Any]]`, `schema_version_changed: bool`), consumed by R8's run-comparison UI.

**Effort:** 2 days. **Version impact:** core minor.

- [ ] **Step 1: Write the failing test** comparing two `StandardScaler` artifacts fitted on different data and asserting `changed` names the `mean` path with both values.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement a recursive dict/list diff with numeric tolerance** (`math.isclose`, `rel_tol=1e-9`) so float noise does not produce spurious diffs.
- [ ] **Step 4: Verify `format_diff` output is readable in a terminal** for a 200-column artifact by truncating per-path value display to 80 characters.

### Task R4.3: Default-on train/test row-overlap detection

**Files:**
- Modify: `skyulf-core/skyulf/leakage.py` — add `fingerprint_rows(df) -> frozenset[int]`, `check_apply_overlap(fit_fingerprint, apply_df, policy) -> LeakageWarning | None`
- Modify: `skyulf-core/skyulf/preprocessing/base.py` — store a bounded row fingerprint in the artifact at fit time; check it at apply time
- Create: `skyulf-core/skyulf/leakage_policy.py` — `LeakagePolicy` enum (`WARN` default, `RAISE`, `OFF`), read from `SKYULF_LEAKAGE_POLICY`
- Test: `skyulf-core/tests/test_leakage_overlap.py`

**Interfaces:**
- Consumes: R4.1's artifact stamping (the fingerprint lives under `__schema__`-adjacent metadata, versioned like everything else).
- Produces: `LeakageWarning(overlap_rows: int, overlap_fraction: float, message: str)` surfaced by the backend in Task R4.4.

**Effort:** 4 days. **Version impact:** core minor (new default-on diagnostic; `WARN`, not `RAISE`, so it cannot break an existing pipeline).

- [ ] **Step 1: Write the failing test**

```python
def test_apply_on_fit_rows_warns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    art = StandardScalerCalculator().fit(df, {"columns": ["a"]})
    with pytest.warns(SkyulfLeakageWarning, match="4 rows"):
        StandardScalerApplier().apply(df, art)
```

- [ ] **Step 2: Run it.** Expected: FAIL (no warning today).
- [ ] **Step 3: Implement the fingerprint** as a hash of row *content* (not index), capped at a sampled subset (default 50,000 rows) so memory stays bounded on large datasets — record the sampling rate in the artifact so the reported overlap fraction is honest.
- [ ] **Step 4: Verify the escape hatch works** — `SKYULF_LEAKAGE_POLICY=OFF` produces no warning; `RAISE` raises. Document all three in `skyulf-core/README.md`.
- [ ] **Step 5: Benchmark** the fingerprint cost on a 1M-row frame; if it exceeds 500ms, reduce the default sample cap rather than making the check opt-in.

### Task R4.4: Real-time canvas guardrail warnings

**Files:**
- Create: `backend/ml_pipeline/_internal/_routers/guardrails.py` — `POST /api/pipelines/guardrails` returning the already-computed leakage/correlation/quality findings for a graph, without running it
- Modify: `frontend/ml-canvas/src/pages/CanvasPage.tsx` and the node components' shared warning affordance — render guardrail findings as in-canvas node badges with a one-click fix where the finding has a deterministic remedy (e.g. "move this learned transform after the split")
- Test: `tests/test_guardrails_router.py`, `frontend/ml-canvas/src/pages/CanvasPage.test.tsx`

**Interfaces:**
- Consumes: `skyulf.leakage.validate_leakage_safety()` (exists today, opt-in), the >0.95 target-correlation check in `skyulf-core/skyulf/profiling/analyzer.py`, and R4.3's `LeakageWarning`.
- Produces: `GuardrailFinding` schema `{node_id: str, severity: "error" | "warning" | "info", code: str, message: str, fix: {action: str, params: dict} | null}` — the frontend's `GuardrailBadge` consumes exactly these fields.

**Effort:** 5 days. **Version impact:** backend minor (new endpoint), frontend minor (new UI).

- [ ] **Step 1: Write the failing backend test** posting a graph with a `StandardScaler` placed *before* the train/test split and asserting one `GuardrailFinding` with `code == "learned_transform_before_split"` and a non-null `fix`.
- [ ] **Step 2: Run it.** Expected: FAIL (route not mounted).
- [ ] **Step 3: Implement the router** by calling the existing checks — do not write new leakage logic in the backend; core owns that.
- [ ] **Step 4: Implement `GuardrailBadge`** and wire the one-click fix through the existing graph-mutation actions in `useGraphStore`.
- [ ] **Step 5: Verify the Sync Rule** — `GuardrailFinding.code` is a backend-owned enum; add its literal union to the frontend types and add a test that fails if the backend emits a code the frontend does not render.

### Task R4.5: Universal calculator contract tests

**Files:**
- Modify: `skyulf-core/tests/test_node_contracts.py` — parametrise over **every** node in `NodeRegistry`, not a curated subset
- Test: same file

**Interfaces:** Consumes `NodeRegistry` and R4.1's stamping. Produces a registry-wide guarantee that every node's artifact is JSON-serialisable, stamped, migrate-able, and round-trips through fit → serialise → deserialise → apply.

**Effort:** 3 days. **Version impact:** core patch-level work, shipped inside the 0.8.0 minor.

- [ ] **Step 1: Parametrise the contract test over `NodeRegistry.all()`** with an explicit, shrinking `KNOWN_UNCONTRACTED` list (empty is the goal) and `xfail(strict=True)` so a node that starts passing forces the list to be updated.
- [ ] **Step 2: Run it** — `cd skyulf-core && python -m pytest tests/test_node_contracts.py -q`. Expected: every registered node either passes or is named in `KNOWN_UNCONTRACTED`; no skips.
- [ ] **Step 3: Assert resampling nodes are included** (they are currently skipped in the smoke tests per `differentiation-strategy.md` Part 3).

### Task R4.6: Release cut R4

**Files:** Create `changelog/0.9.x.md` with `## v0.9.0 — Leakage-Safe by Construction`; modify all three version files.

**Effort:** 1 hour. **Version impact:** backend `0.8.1`→`0.9.0`, frontend `0.8.0`→`0.9.0`, core `0.7.0`→`0.8.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=0.9.0`, `FRONTEND_VER=0.9.0`, `CORE_VER=0.8.0`, `SERIES_FILE=changelog/0.9.x.md`, `TITLE=Leakage-Safe by Construction`. Step F applies.

**Gate:** building a leakage-prone graph on the canvas produces a visible warning **before** the run, applying an artifact to its own fit data warns, and every registered node passes the contract test.

---

## Release R5 — Shared Frontend Infrastructure

**Versions:** backend **unchanged (0.9.0)** · frontend `0.10.0` (minor) · skyulf-core **unchanged (0.8.0)**
**Effort:** ~3 weeks
**Draws from:** master-fix-list Phase 4 (all 7 items) and the explicit "What NOT to do" caution: *don't skip Phase 4 and jump straight to Phase 5 page redesigns*.
**Why here:** R6 adds roughly a dozen new node UIs and R8/R9 redesign six pages. Both are materially cheaper after the shared `DataTable`, the single token source, the standardised state components, and — most importantly — the schema-driven node settings renderer exist. Building R6's node components by hand first would mean rewriting them.
**Dependencies:** R4 (the guardrail badge introduced in R4.4 becomes one of the standard node-level affordances this release systematises).
**This is a frontend-only release.** `pyproject.toml` and `skyulf-core/setup.py` are not touched.

### Task R5.1: One `DataTable`, one `StatusBadge`, one token source

**Files:**
- Create: `frontend/ml-canvas/src/components/data-table/DataTable.tsx` — sticky header, sort, density toggle, skeleton rows, empty/filter states, row-action overflow, detail drawer
- Delete: the page-local `StatusBadge` reimplementation confirmed in `frontend/ml-canvas/src/pages/Jobs.tsx`; import the existing shared one instead
- Modify: `frontend/ml-canvas/src/index.css` and `frontend/ml-canvas/src/styles/variables.css` — collapse the two parallel token systems into one semantic set; add an ESLint rule or stylelint check banning raw Tailwind colour classes in `src/pages/`
- Test: `frontend/ml-canvas/src/components/data-table/DataTable.test.tsx`

**Interfaces:**
- Consumes: existing `VirtualList`.
- Produces: `DataTable<TRow>` props `{rows, columns, sort, onSortChange, density, emptyState, filteredEmptyState, rowActions, onRowClick, virtualized}` — every page in R6–R9 consumes exactly this signature.

**Effort:** 6 days. **Version impact:** frontend minor.

- [ ] **Step 1: Write the failing test** asserting `DataTable` renders a filtered-empty state distinct from a first-use empty state, and virtualises when `rows.length > 200`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement `DataTable` on top of the existing `VirtualList`.**
- [ ] **Step 4: Migrate the Dataset table to `DataTable`** (it currently renders every row unvirtualized — the concrete finding in `scale-load-audit.md` and `technical-debt-deep-dive.md` §B2).
- [ ] **Step 5: Verify the duplicate badge is gone** — `grep -rn "StatusBadge" frontend/ml-canvas/src/ | grep -v components/` returns only imports, no local definitions.

### Task R5.2: Standardised state components + `useGraphStore` split

**Files:**
- Create: `frontend/ml-canvas/src/components/states/EmptyState.tsx`, `LoadingState.tsx`, `ErrorState.tsx` — variants `first-use`, `filtered-empty`, `permission-error`, `recoverable-failure`
- Modify: `frontend/ml-canvas/src/core/**/useGraphStore.ts` — split into `useExecutionSlice`, `useSchemaSlice`, `useCanvasDerivedSlice`
- Modify: the canvas header — add an explicit `dirty | synced | conflict` indicator for autosave-vs-server-save divergence
- Test: `frontend/ml-canvas/src/core/**/useGraphStore.test.ts`

**Interfaces:**
- Consumes: existing store actions (names preserved exactly — a rename here breaks every page).
- Produces: `useExecutionSlice()`, `useSchemaSlice()`, `useCanvasDerivedSlice()`, and `SyncStatus = "dirty" | "synced" | "conflict"`.

**Effort:** 6 days. **Version impact:** frontend minor.

- [ ] **Step 1: Write a test** asserting that editing a node config marks status `dirty`, a successful save marks `synced`, and a server version newer than the local base marks `conflict`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Split the store slice-by-slice**, keeping the existing public action names so no page needs editing in this task.
- [ ] **Step 4: Verify no behavioural regression** — `npx vitest run` must pass with zero changes to page-level tests.

### Task R5.3: Schema-driven node settings renderer

**Files:**
- Create: `frontend/ml-canvas/src/modules/nodes/shared/NodeSettingsForm.tsx` — renders a form from a declarative field schema
- Create: `frontend/ml-canvas/src/modules/nodes/shared/nodeFieldSchemas.ts` — the frontend-local schema registry (one entry per node type)
- Modify: three existing node components as the proof (the largest ones, currently 135–1,171 LOC each) to render via `NodeSettingsForm`
- Test: `frontend/ml-canvas/src/modules/nodes/shared/NodeSettingsForm.test.tsx`

**Interfaces:**
- Consumes: nothing from the backend **in this release** — the schemas are frontend-local. R6 adds `GET /api/nodes/metadata` and switches the registry to be backend-driven; keeping that out of this release is what makes R5 frontend-only.
- Produces: `NodeFieldSchema = {key: string; label: string; kind: "number" | "text" | "select" | "multiselect" | "columns" | "boolean"; options?: {value: string; label: string}[]; default?: unknown; help?: string}` and `NodeSettingsForm` props `{schema: NodeFieldSchema[]; value: Record<string, unknown>; onChange(next): void}`.

**Effort:** 7 days. **Version impact:** frontend minor.

- [ ] **Step 1: Write the failing test** rendering a three-field schema (select + columns + number) and asserting `onChange` fires with the correct partial update, and that an option list marked required blocks submission when empty.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement `NodeSettingsForm`** and migrate exactly three node components — do not migrate all of them in this release; prove the abstraction first.
- [ ] **Step 4: Verify no dropdown option was lost in migration** — for each migrated node, diff the option arrays before and after:

```bash
cd /Users/BH7043/Skyulf && git diff master -- frontend/ml-canvas/src/modules/nodes/ | grep -E "^[-+].*(value=|options)" | sort | uniq -c
```

Expected: every removed option string reappears as an added one in `nodeFieldSchemas.ts`. This is the Backend/Core ↔ Frontend Sync Rule applied to a refactor — a silently dropped option is exactly the class of shipped bug the rule exists to prevent.

### Task R5.4: Unified app shell

**Files:**
- Create: `frontend/ml-canvas/src/components/shell/AppShell.tsx` — Build / Operate / Observe / Settings navigation grouping, with a named `slotOrgSwitcher` render-prop that renders `null` today
- Modify: `frontend/ml-canvas/src/App.tsx`
- Test: `frontend/ml-canvas/src/components/shell/AppShell.test.tsx`

**Interfaces:**
- Consumes: existing route definitions.
- Produces: `AppShell` props `{slotOrgSwitcher?: ReactNode}` — Track B's `@skyulf/enterprise-ui` fills this slot without forking the shell. This is the single, deliberate extension point that keeps Track B from requiring changes to Track A code.

**Effort:** 3 days. **Version impact:** frontend minor.

- [ ] **Step 1: Write the failing test** asserting all existing routes are reachable from the new grouped navigation and that `slotOrgSwitcher` renders nothing when not supplied.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement `AppShell` and re-parent `App.tsx`.**
- [ ] **Step 4: Verify keyboard navigation** reaches every nav group with Tab and activates with Enter — this is a prerequisite for R9's accessibility work.

### Task R5.5: Release cut R5

**Files:** Add `## v0.9.0 — Shared Frontend Infrastructure`… **no.** The backend is unchanged, so the release drafter (which keys on `pyproject.toml`) will not fire. Cut this release as a frontend-only tag.

**Interfaces:** Consumes R5.1–R5.4. Produces frontend 0.10.0.

**Effort:** 1 hour. **Version impact:** frontend `0.9.0`→`0.10.0` only.

- [ ] **Step 1: Bump `frontend/ml-canvas/package.json` to `0.10.0`.** Do not touch `pyproject.toml` or `skyulf-core/setup.py`.
- [ ] **Step 2: Add a `### 🎨 Frontend` block to the *next* backend release's changelog entry** (R6's `## v0.10.0`), noting the frontend shipped 0.10.0 ahead of it — the drafter is backend-version-keyed, so a frontend-only release has no GitHub Release of its own by design.
- [ ] **Step 3: Verify version independence**

```bash
cd /Users/BH7043/Skyulf && grep -m1 '^version' pyproject.toml && grep -m1 '"version"' frontend/ml-canvas/package.json && grep -m1 'version=' skyulf-core/setup.py
```

Expected: `0.9.0`, `0.10.0`, `0.8.0` — exactly the R5 row of the Version Ledger.
- [ ] **Step 4: Run the frontend gate**

```bash
cd /Users/BH7043/Skyulf/frontend/ml-canvas && npm run lint && npx tsc --project tsconfig.json --noEmit && npm run build && npx vitest run
```

- [ ] **Step 5: Commit and tag**

```bash
git add frontend/ml-canvas/package.json
git commit -m "chore: release frontend v0.10.0 (shared frontend infrastructure)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git tag frontend-v0.10.0 && git push && git push origin frontend-v0.10.0
```

**Gate:** the Dataset page renders 10,000 rows without freezing, no page defines its own `StatusBadge`, and three node components render from `nodeFieldSchemas.ts` with zero dropdown options lost.

---

## Release R6 — Node Coverage Parity & Paper-Backed Nodes

**Versions:** backend `0.10.0` (minor) · frontend `0.11.0` (minor) · skyulf-core `0.9.0` (minor)
**Effort:** ~4 weeks
**Draws from:** master-fix-list Phase 16c (all 8 coverage gaps), Phase 17a (CatBoost, MLflow-skinny hook), Phase 18a (optimal binning, mutual-information/distance-correlation selection, PyOD outliers), Phase 18b (TabPFN instant baseline), Phase 18c (cleanlab label-quality report — explicitly reusable for classical ML, not just DL), Phase 7 items 1–2 (`ManualBounds` missing from the frontend UI, `one_hot.py` allow-list gap).
**Why here:** these are the cheapest possible "new capability" wins (every one wraps a maintained library rather than reimplementing published math, per the master fix list's explicit "What NOT to do"), and R5 just made adding a node UI cheap. Coverage breadth is also what makes the R1.4/R4 positioning credible — "leakage-safe" is not persuasive if the node palette is thin.
**Dependencies:** R3 (`require_config` — every new node uses it, so no new node joins the silent-no-op bug class), R4 (`artifact_schema_version` — every new node's artifact is stamped from birth), R5 (`NodeSettingsForm` — new node UIs are schema entries, not 300-LOC components).

### Task R6.1: Boosting and sklearn-parity nodes

**Files:**
- Create: `skyulf-core/skyulf/modeling/catboost_models.py` — `CatBoostClassifierCalculator/Applier`, `CatBoostRegressorCalculator/Applier`
- Create: `skyulf-core/skyulf/preprocessing/transformations/quantile.py` — `QuantileTransformerCalculator/Applier`
- Modify: `skyulf-core/skyulf/preprocessing/feature_generation/generation.py` — add cyclical (sin/cos) calendar encoding to `DateFeatures`
- Modify: `skyulf-core/skyulf/modeling/cross_validation.py` — add `GroupKFold` and `StratifiedGroupKFold`
- Modify: `skyulf-core/setup.py` extras — `catboost` under a `boosting` extra, mirroring the existing xgboost/lightgbm optional pattern
- Test: `skyulf-core/tests/test_catboost_nodes.py`, `test_quantile_transformer.py`, `test_group_cv.py`

**Interfaces:**
- Consumes: `SklearnCalculator`/`SklearnApplier` (`skyulf-core/skyulf/modeling/sklearn_wrapper.py:21+`), the lazy-optional-import guard pattern already used for LightGBM in `modeling/classification.py`.
- Produces: registry ids `catboost_classifier`, `catboost_regressor`, `quantile_transformer`, cv strategies `group_kfold`, `stratified_group_kfold`, and `DateFeatures` methods `cyclical_month`, `cyclical_dayofweek`, `cyclical_hour`.

**Effort:** 6 days. **Version impact:** core minor.

- [ ] **Step 1: Write the failing test** for each node (fit → JSON-serialisable stamped artifact → apply → parity between the pandas and Polars engines, per this repo's dual-engine convention).
- [ ] **Step 2: Run them.** Expected: FAIL, node not registered.
- [ ] **Step 3: Implement each by copying the proven pattern** — for CatBoost, copy `LGBMClassifierCalculator`'s structure including applier-level warning suppression (CatBoost is verbose by default).
- [ ] **Step 4: Verify optional-dependency behaviour** — with `catboost` uninstalled, importing `skyulf` must still succeed and instantiating the node must raise a message naming `pip install skyulf-core[boosting]`:

```bash
cd /Users/BH7043/Skyulf/skyulf-core && python -c "import skyulf; print('import ok')"
```

### Task R6.2: Paper-backed preprocessing and diagnostics nodes

**Files:**
- Create: `skyulf-core/skyulf/preprocessing/bucketing_optimal.py` — `OptimalBinningCalculator/Applier` wrapping `optbinning` (Navas-Palencia 2020, arXiv:2001.08025)
- Create: `skyulf-core/skyulf/preprocessing/feature_selection/mutual_info.py` — wraps `sklearn.feature_selection.mutual_info_classif/regression`; optional `dcor` for distance correlation (Schellhas et al. 2020, arXiv:2006.12919)
- Create: `skyulf-core/skyulf/preprocessing/outliers/pyod_detectors.py` — HBOS/COPOD/ECOD/IsolationForest/LOF via `pyod` (Zhao et al. 2019, arXiv:1901.01588; PyOD 2, arXiv:2412.12154)
- Create: `skyulf-core/skyulf/profiling/label_quality.py` — `LabelQualityReport` via `cleanlab` (Northcutt et al. 2021, arXiv:1911.00068), consuming predicted probabilities so it works for classical ML today and DL later
- Create: `skyulf-core/skyulf/preprocessing/feature_selection/vif.py` — VIF/multicollinearity node; add Kendall to the pipeline-composable correlation node
- Create: `skyulf-core/skyulf/preprocessing/inspection_tests.py` — Shapiro/KS/ANOVA/chi-square-of-independence as registered nodes (they exist only inside profiling reports today)
- Modify: `skyulf-core/setup.py` extras — `optbinning`, `pyod`, `cleanlab`, `dcor` under a `research` extra
- Test: one test module per node under `skyulf-core/tests/`

**Interfaces:**
- Consumes: `require_config` (R3.1), `skyulf.artifacts.stamp` (R4.1).
- Produces: registry ids `optimal_binning`, `mutual_info_selection`, `pyod_outliers`, `label_quality_report`, `vif_selection`, `hypothesis_tests`.

**Effort:** 8 days. **Version impact:** core minor.

- [ ] **Step 1: For each node, write the failing artifact-shape test first** — the artifact must be JSON-serialisable (optbinning's bin edges and PyOD's threshold are; a fitted PyOD *model object* is not, so persist scores/thresholds, not the estimator, wherever possible).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement each as a wrapper** — per the master fix list, do not reimplement any published algorithm that has a maintained package.
- [ ] **Step 4: Verify every new node appears in the R4.5 universal contract test** with no `KNOWN_UNCONTRACTED` entry.

### Task R6.3: TabPFN instant-baseline node

**Files:**
- Create: `skyulf-core/skyulf/modeling/tabpfn.py` — `TabPFNClassifierCalculator/Applier` (Hollmann et al., ICLR 2023, arXiv:2207.01848)
- Modify: `skyulf-core/setup.py` extras — `tabpfn` under the `research` extra
- Test: `skyulf-core/tests/test_tabpfn_node.py`

**Interfaces:** Consumes `SklearnCalculator` (TabPFN exposes an sklearn-compatible API). Produces registry id `tabpfn_classifier`, which **must** raise a clear, actionable error above its supported dataset size rather than silently degrading.

**Effort:** 2 days. **Version impact:** core minor.

- [ ] **Step 1: Write the failing test** covering both a small dataset (fits, produces sane accuracy) and an oversized one (raises `SkyulfConfigError` naming the row/feature limit).
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement with the documented size guard read from the installed TabPFN version**, not hardcoded.
- [ ] **Step 4: Confirm the CI time budget** — cap the test dataset so this node adds under 60 seconds to the suite.

### Task R6.4: MLflow-skinny fit hook

**Files:**
- Create: `skyulf-core/skyulf/integrations/mlflow.py` — `enable_mlflow_logging(tracking_uri: str | None = None) -> None`
- Modify: `skyulf-core/skyulf/modeling/base.py` — one additive post-fit callback hook on `BaseModelCalculator.fit()`
- Modify: `skyulf-core/setup.py` extras — `mlflow-skinny` under a `tracking` extra
- Test: `skyulf-core/tests/test_mlflow_integration.py`

**Interfaces:** Consumes the existing `BaseModelCalculator.fit()` return contract, which must **not** change (per Phase 17a: purely additive). Produces `enable_mlflow_logging()` and an internal `register_fit_callback(fn: Callable[[str, dict, dict], None]) -> None` seam that R17's DL telemetry also uses.

**Effort:** 2 days. **Version impact:** core minor.

- [ ] **Step 1: Write the failing test** using a fake callback (no MLflow installed) asserting the hook fires once per `fit` with `(node_type, config, metrics)`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement the callback registry and the MLflow adapter behind a lazy import.**
- [ ] **Step 4: Verify zero overhead when unused** — with no callbacks registered, `fit` must make no additional calls (assert via a `unittest.mock` patch count).

### Task R6.5: Backend node metadata endpoint + frontend node UIs

**Files:**
- Create: `backend/ml_pipeline/_internal/_routers/node_metadata.py` — `GET /api/nodes/metadata` serialising `@node_meta` for every registered node
- Modify: `frontend/ml-canvas/src/modules/nodes/shared/nodeFieldSchemas.ts` — hydrate from `GET /api/nodes/metadata`, keeping the local schemas as the fallback/offline source
- Create: one node component (or schema entry) per node added in R6.1–R6.3, plus the missing `ManualBounds` outlier node UI (Phase 7 item 1)
- Modify: the one-hot node component — close the `prefix_separator`/`drop_original` allow-list gap (Phase 7 item 2)
- Test: `tests/test_node_metadata_router.py`, `frontend/ml-canvas/src/modules/nodes/**/*.test.tsx`

**Interfaces:**
- Consumes: `NodeRegistry` + `@node_meta` (backend side), `NodeFieldSchema` (R5.3).
- Produces: `GET /api/nodes/metadata` → `{nodes: [{id: str, label: str, category: str, fields: NodeFieldSchema[]}]}` — the field shape is byte-identical to R5.3's `NodeFieldSchema` so the frontend needs no adapter.

**Effort:** 6 days. **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the failing sync test** — a test that fetches `/api/nodes/metadata`, and for every node asserts the frontend's rendered option set equals the backend's declared option set. This is the Backend/Core ↔ Frontend Sync Rule turned into CI, and it is the highest-value single test in this release.
- [ ] **Step 2: Run it.** Expected: FAIL (route missing).
- [ ] **Step 3: Implement the route and the frontend hydration.**
- [ ] **Step 4: Verify the two historical Sync-Rule bugs are now impossible** — add explicit assertions for `ALLOWED_DATETIME_FEATURES` (`season`/`time_of_day`/`minute`/`second`) and for every `InvalidValueReplacementNode` mode string being recognised by the backend rule matcher. If any UI-only alias is still unrecognised, wire up backend support rather than deleting the option.
- [ ] **Step 5: Run the full frontend gate.**

### Task R6.6: Release cut R6

**Files:** Create `changelog/0.10.x.md` with `## v0.10.0 — Node Coverage Parity & Paper-Backed Nodes` (include the `### 🎨 Frontend` note that frontend 0.10.0 shipped in R5); modify all three version files.

**Effort:** 1 hour. **Version impact:** backend `0.9.0`→`0.10.0`, frontend `0.10.0`→`0.11.0`, core `0.8.0`→`0.9.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=0.10.0`, `FRONTEND_VER=0.11.0`, `CORE_VER=0.9.0`, `SERIES_FILE=changelog/0.10.x.md`, `TITLE=Node Coverage Parity & Paper-Backed Nodes`. Step F applies.

**Gate:** the metadata sync test passes for every node, and every new node passes the R4.5 universal contract test.

---

## Release R7 — Engine-Agnostic Core, Data Contracts & Forecasting

**Versions:** backend `0.11.0` (minor) · frontend `0.12.0` (minor) · skyulf-core `0.10.0` (minor)
**Effort:** ~4 weeks
**Draws from:** Phase 17a (Narwhals as an additive third `EngineRegistry` backend, Pandera `SchemaContract` node, StatsForecast forecasting family, DuckDB as ingestion convenience only), Phase 17b ranked item #4 (fit/apply-native schema-drift detection auto-derived from artifacts), Phase 16c (no classical forecasting models), Phase 9 Bet #5, Phase 18d's TFDV validation note.
**Sequencing rule (from Phase 17a, non-negotiable):** **Narwhals lands before Pandera.** Pandera 0.32's validation engine is itself Narwhals-powered, so adopting Narwhals first makes the Pandera integration strictly easier.
**What this release deliberately does NOT do (per "What NOT to do"):** no full lazy DuckDB execution path, no internal Narwhals engine *replacement*. Both are L-effort and gated on the partitionable calculator contract (R11).
**Dependencies:** R6 (the new node families are the ones that most benefit from a single Narwhals code path instead of hand-written pandas/Polars branches).

### Task R7.1: Narwhals as an additive third engine backend

**Files:**
- Create: `skyulf-core/skyulf/engines/narwhals_engine.py` — `SkyulfNarwhalsWrapper(BaseEngine)`
- Modify: `skyulf-core/skyulf/engines/registry.py` — add `EngineName.NARWHALS`
- Modify: `skyulf-core/setup.py` extras — `narwhals` under a `lazy` extra
- Test: `skyulf-core/tests/test_narwhals_engine.py`

**Interfaces:** Consumes the existing `SkyulfDataFrame` protocol (`skyulf-core/skyulf/engines/protocol.py`) and `BaseEngine` subclassing pattern used by `SkyulfPandasWrapper`. Produces DuckDB/PyArrow/Dask/Modin as accepted **lazy inputs** — the existing pandas and Polars dispatch paths are untouched.

**Effort:** 5 days. **Version impact:** core minor.

- [ ] **Step 1: Write the failing test** passing a PyArrow table and a DuckDB relation through a simple scaling node and asserting the output matches the pandas path.
- [ ] **Step 2: Run it.** Expected: FAIL, unsupported engine.
- [ ] **Step 3: Implement `SkyulfNarwhalsWrapper` as an additive backend**, not a replacement — the existing `_to_positional_values` index-alignment shim in `pandas_engine.py` stays exactly as-is.
- [ ] **Step 4: Verify no regression on the default path** — run the whole core suite with the `lazy` extra uninstalled.

### Task R7.2: `SchemaContract` node + auto-derived schema drift

**Files:**
- Create: `skyulf-core/skyulf/preprocessing/schema_contract.py` — `SchemaContractCalculator` (captures the fit-time schema via `pandera.io.infer_schema`, persists dtype/nullability/min/max/category-set into the JSON artifact) and `SchemaContractApplier` (runs `schema.validate(df, lazy=True)` and returns a structured drift report rather than raising)
- Create: `skyulf-core/skyulf/artifacts/implied_schema.py` — `implied_output_schema(artifact, node_type) -> SchemaSpec`, derived from artifacts that already exist (no hand-written schema required)
- Modify: `skyulf-core/setup.py` extras — `pandera` under a `schema` extra
- Test: `skyulf-core/tests/test_schema_contract.py`, `test_implied_schema.py`

**Interfaces:** Consumes R7.1's Narwhals backend and R4.1's stamped artifacts. Produces `SchemaSpec` (`columns: list[{name, dtype, nullable, min, max, categories}]`) and `SchemaDriftReport` (`missing_columns`, `new_columns`, `dtype_changes`, `range_violations`), consumed by the backend drift UI.

**Effort:** 6 days. **Version impact:** core minor.

- [ ] **Step 1: Write the failing test** asserting a column dropped between fit and apply appears in `missing_columns`, and an `int64 → float64` change appears in `dtype_changes`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement both modules.** Per Phase 18d's TFDV evidence, schema-level anomalies are the priority — do not defer this in favour of statistical distance metrics (those come in R18).
- [ ] **Step 4: Verify the report never raises by default** — an apply-time violation must produce a report, not a crash, unless `on_violation="raise"` is configured.

### Task R7.3: StatsForecast classical forecasting family

**Files:**
- Create: `skyulf-core/skyulf/modeling/forecasting.py` — `AutoARIMAForecasterCalculator/Applier`, `AutoETSForecasterCalculator/Applier`
- Modify: `skyulf-core/setup.py` extras — `statsforecast` under a `forecasting` extra
- Test: `skyulf-core/tests/test_forecasting_nodes.py`

**Interfaces:** Consumes `SklearnCalculator`/`SklearnApplier` (StatsForecast exposes `.fit()`/`.predict()`, which maps directly). Produces registry ids `auto_arima_forecaster`, `auto_ets_forecaster`, integrating with the existing `preprocessing/time_series/` conventions for sort column and horizon.

**Effort:** 5 days. **Version impact:** core minor.

- [ ] **Step 1: Write the failing test** on a synthetic seasonal series asserting an `h`-step forecast of the correct length and a sane MASE against a naive baseline.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement both nodes.** StatsForecast was chosen over Prophet specifically to avoid Stan/PyStan install fragility (Phase 17a) — do not substitute Prophet.
- [ ] **Step 4: Verify the artifact is JSON-serialisable** and passes the R4.5 contract test.

### Task R7.4: DuckDB ingestion convenience + backend/frontend wiring

**Files:**
- Create: `skyulf-core/skyulf/data/duckdb_source.py` — `load_via_duckdb(query: str, files: list[str]) -> SkyulfDataFrame` (filter/aggregate/join down to a manageable size, hand off an Arrow table to the existing engines)
- Create: `backend/ml_pipeline/_internal/_routers/schema_contract.py` — expose `SchemaDriftReport` for a fitted node
- Create: frontend schema-contract node entry in `nodeFieldSchemas.ts` + a drift-report panel in the drift pages
- Create: frontend forecasting node entries
- Test: `skyulf-core/tests/test_duckdb_source.py`, `tests/test_schema_contract_router.py`

**Interfaces:** Consumes R7.2's `SchemaDriftReport` and R6.5's `/api/nodes/metadata`. Produces `GET /api/nodes/{node_id}/schema-drift` → `SchemaDriftReport`.

**Effort:** 5 days. **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the failing test** for the DuckDB loader over two Parquet files with a filter that reduces 10M rows to 10k.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement the loader, router, and frontend entries.** Do **not** build a lazy DuckDB execution path — this is ingestion/materialisation only.
- [ ] **Step 4: Run the R6.5 metadata sync test** to confirm the new nodes' options match end to end.

### Task R7.5: Release cut R7

**Files:** Create `changelog/0.11.x.md` with `## v0.11.0 — Engine-Agnostic Core, Data Contracts & Forecasting`; modify all three version files.

**Effort:** 1 hour. **Version impact:** backend `0.10.0`→`0.11.0`, frontend `0.11.0`→`0.12.0`, core `0.9.0`→`0.10.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=0.11.0`, `FRONTEND_VER=0.12.0`, `CORE_VER=0.10.0`, `SERIES_FILE=changelog/0.11.x.md`, `TITLE=Engine-Agnostic Core, Data Contracts & Forecasting`. Step F applies.

**Gate:** a pipeline runs unchanged with a PyArrow input, a schema contract detects a dropped column at apply time, and an AutoARIMA node beats a naive baseline on the synthetic series.

---

## Release R8 — Transparency & Diagnostics

**Versions:** backend `0.12.0` (minor) · frontend `0.13.0` (minor) · skyulf-core **unchanged (0.10.0)**
**Effort:** ~5 weeks
**Draws from:** Phase 15b tier (a) (post-fit diagnostics + opt-in PCA class-separation plot), Phase 15a Phase A (read-only per-node generated-code view + notebook export for every node type), Phase 16a (notebook export has never been checked for standalone execution correctness — fix **before** shipping Phase A, not after), Phase 9 Round-4 addition (per-node/per-step data preview — the most externally-validated UX gap found), Phase 9 Round-5 addition (unify job logs + per-node execution ledger + data-quality warnings into one canonical run timeline), Phase 5 redesigns for Pipeline Canvas / Experiments / Jobs.
**Why here:** `user-complaints-research.md`'s two strongest external signals are "vendor lock-in / no code export" and "the schema guessing game" (no per-node preview). Both are answered here, and both are mostly **wiring existing backend data into UI** rather than new capability — which is exactly the right cost profile.
**Dependencies:** R5 (`DataTable`, state components, `AppShell`), R4 (`artifacts.diff()` powers run comparison).
**Explicitly out of scope:** code-escape-hatch Phase B (needs auth — R10) and Phase C (blocked; Track B only).

### Task R8.1: Notebook export correctness, then all-node coverage

**Files:**
- Modify: `backend/ml_pipeline/_internal/_routers/_notebook_builders.py` — add generators for loaders, splitters, models, resampling (today only `preprocess` nodes go through `node_to_cell`)
- Modify: `backend/ml_pipeline/_internal/_routers/notebook_export.py` — emit a correct `pip install skyulf-core[...]` line reflecting the extras the exported graph actually needs
- Create: `tests/test_notebook_export_executes.py` — execute the exported notebook with `nbclient` against a sample dataset and assert it completes
- Modify: `requirements-dev.txt` — add `nbclient`

**Interfaces:** Consumes `NodeRegistry` and R6.5's node metadata. Produces an exported notebook that **runs**, verified in CI.

**Effort:** 6 days. **Version impact:** backend minor.

- [ ] **Step 1: Write the failing execution test** exporting a graph that includes a loader, a splitter, two preprocessing nodes, and a model, then running it via `nbclient`.
- [ ] **Step 2: Run it.** Expected: FAIL — either a missing cell type or a wrong install line.
- [ ] **Step 3: Add the missing generators and fix the install line.**
- [ ] **Step 4: Verify the extras line is derived, not hardcoded** — a graph containing a CatBoost node must emit `skyulf-core[boosting]`.

### Task R8.2: Read-only per-node generated-code view

**Files:**
- Create: `backend/ml_pipeline/_internal/_routers/node_code.py` — `GET /api/nodes/{node_id}/code?graph_id=...`
- Create: `frontend/ml-canvas/src/modules/nodes/shared/NodeCodePanel.tsx` — a read-only, copyable code panel in the node inspector
- Test: `tests/test_node_code_router.py`, `NodeCodePanel.test.tsx`

**Interfaces:** Consumes the same builders as R8.1 (one code path, not two). Produces `GET /api/nodes/{node_id}/code` → `{language: "python", source: str, fidelity: "generated-execution-equivalent"}`.

**Effort:** 4 days. **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the failing test** asserting the returned source, run standalone with the node's config, produces the same artifact as the in-platform run.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement the route and panel.**
- [ ] **Step 4: Label fidelity honestly in the UI** — per the code-escape-hatch study, the panel must say "generated execution-equivalent code", **not** "the literal code that ran". Assert the label string in the component test so it cannot be quietly reworded.
- [ ] **Step 5: Confirm scope containment** — the panel is read-only. Grep the diff for `exec(`, `eval(`, or any writable code field; there must be none. Phase A must not become an accidental Phase C.

### Task R8.3: Per-node data preview

**Files:**
- Modify: `backend/ml_pipeline/_internal/_routers/preview.py` — return a bounded materialised sample at any selected node without requiring a full run
- Modify: the canvas node inspector — a "Data at this node" tab rendering via R5.1's `DataTable`
- Test: `tests/test_preview_router.py`, canvas inspector test

**Interfaces:** Consumes the existing preview execution path. Produces `GET /api/pipelines/{id}/nodes/{node_id}/preview?rows=100` → `{columns: [{name, dtype}], rows: list[dict], truncated: bool, row_count_estimate: int}`.

**Effort:** 7 days. **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the failing test** asserting a preview at a mid-graph node returns post-transform columns (not the raw input schema) and sets `truncated: true` beyond the row cap.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement with a hard row and byte cap** so a preview can never load a full large dataset into the API process.
- [ ] **Step 4: Verify blocking I/O stays out of the async handler** (this is also a Phase 2 finding) — the dataframe read runs in a thread executor.

### Task R8.4: Unified run diagnostic timeline + post-fit diagnostics

**Files:**
- Create: `backend/monitoring/_routers/run_timeline.py` — `GET /api/jobs/{job_id}/timeline` merging job logs, the per-node execution ledger, and data-quality/guardrail warnings into one ordered stream
- Create: `frontend/ml-canvas/src/pages/jobs/RunTimeline.tsx`
- Modify: the job-completion view — surface the existing `ClassificationChartsForSplit` / `RegressionChartsForSplit` and feature-importance output prominently instead of behind navigation
- Create: an opt-in final PCA class-separation plot reusing the existing Plotly scatter component, labelled as an **input-space projection** (not a learned embedding — that distinction becomes real only for DL in R17)
- Modify: the Experiments/Run-Comparison page to use R4.2's `artifacts.diff()` for "what changed between these two runs"
- Test: `tests/test_run_timeline.py`, `RunTimeline.test.tsx`

**Interfaces:** Consumes existing job logs, notification history, preview-node failure cards, and R4.4's `GuardrailFinding`. Produces `TimelineEvent = {ts: str, source: "log" | "node" | "guardrail" | "notification", node_id: str | null, severity: str, message: str, detail: dict | null}`.

**Effort:** 8 days. **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the failing test** asserting a failed run's timeline contains, in order, the node-start event, the guardrail warning, and the failure log for the same `node_id`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement the merge server-side** — this is wiring existing data, not new capability; do not add new instrumentation in this task.
- [ ] **Step 4: Verify the PCA plot is opt-in** and never computed for a completed job unless requested.

### Task R8.5: Canvas / Experiments / Jobs redesigns

**Files:**
- Modify: `frontend/ml-canvas/src/pages/CanvasPage.tsx` (health strip, command bar, inspector tabs — the tabs now host R8.2's code panel and R8.3's data preview)
- Modify: the Experiments/Run-Comparison page (ranked table via `DataTable`, decision rail)
- Modify: `frontend/ml-canvas/src/pages/Jobs.tsx` + the job drawer — one source of truth instead of a drawer and a routed page that diverge
- Test: the existing page test files

**Interfaces:** Consumes R5.1–R5.4 and R8.1–R8.4. Produces no new API surface.

**Effort:** 8 days. **Version impact:** frontend minor.

- [ ] **Step 1: Write failing tests** asserting the Jobs drawer and the routed Jobs page render from the same hook (assert a single shared query key).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement all three redesigns against `redesign-existing-pages.md` §1–§3.**
- [ ] **Step 4: Run the full frontend gate plus Playwright.**

### Task R8.6: Release cut R8

**Files:** Create `changelog/0.12.x.md` with `## v0.12.0 — Transparency & Diagnostics`; modify `pyproject.toml` and `package.json` only.

**Effort:** 1 hour. **Version impact:** backend `0.11.0`→`0.12.0`, frontend `0.12.0`→`0.13.0`, core **stays 0.10.0**.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=0.12.0`, `FRONTEND_VER=0.13.0`, `CORE_VER=` *(skip — do not edit `setup.py`)*, `SERIES_FILE=changelog/0.12.x.md`, `TITLE=Transparency & Diagnostics`. **Skip Step F** — no core tag, because core did not change.

**Gate:** the exported notebook executes end-to-end in CI, every node exposes a code view and a data preview, and a failed run's timeline explains the failure without opening the browser console.

---

## Release R9 — Hardening, Contracts & Accessibility

**Versions:** backend `0.13.0` (minor) · frontend `0.14.0` (minor) · skyulf-core **unchanged (0.10.0)**
**Effort:** ~6 weeks
**Draws from:** Phase 2 (all 11 resilience/correctness items, including `pipeline_schema_version` + migration registry), Phase 3 (all 4 accessibility items), Phase 10 (SSRF SEC-01/SEC-02, per-user resource quotas, concurrent Celery worker deployment, memory budgets, large-table virtualisation), Phase 11 (all 5 testing/CI items), Phase 13 (all 5 API-contract items), Phase 5 remaining redesigns (Dataset, Drift, Model Registry).
**Why here, and why before R10:** `pipeline_schema_version` must exist **before** DL adds new node types (Phase 2's explicit note). `/api/v1` versioning must exist before R10 makes the API public-facing and authenticated. The testing/CI foundations must exist before Ray and DL plug into exactly the areas found weakest.
**Dependencies:** R8 (redesigns build on the same shared components; do the remaining three pages here).

### Task R9.1: Pipeline schema versioning + migration registry

**Files:**
- Create: `backend/ml_pipeline/_schema_versions.py` — `PIPELINE_SCHEMA_VERSION`, `register_pipeline_migration(from_v, fn)`, `migrate_pipeline(graph) -> graph`
- Modify: the pipeline save/load service and `PipelineVersion` persistence
- Test: `tests/test_pipeline_schema_migration.py`

**Interfaces:** Consumes saved graph JSON. Produces `graph["schema_version"]: int` on every saved pipeline and a migration applied on load. This is the mechanism that lets R15–R17 add DL node types without breaking saved graphs.

**Effort:** 6 days. **Version impact:** backend minor.

- [ ] **Step 1: Write the failing test** loading an unversioned (pre-0.13.0) saved graph fixture and asserting it migrates to the current version.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement, defaulting unversioned graphs to version 1.**
- [ ] **Step 4: Verify a graph saved by a *newer* backend is rejected with a clear message** rather than silently mis-loaded.

### Task R9.2: Concurrency, cancellation and resource correctness

**Files:**
- Modify: `backend/ml_pipeline/tasks.py` and the job service — close the cancellation race (a queued job must not be resurrected and trained after cancel), add a heartbeat/lease-based reaper independent of API restarts, add Celery time limits, add S3/Redis retry with backoff that **never** coerces a storage failure into "no artifacts"
- Modify: the `PipelineVersion` model — a uniqueness constraint on `version_int` closing the version-collision race; optimistic locking/row revision on `TrainingJob`
- Modify: `backend/data_ingestion/service.py` — decompression-bomb/resource limits on XLSX/JSON parsing
- Modify: DB models — composite indexes for the job/log query patterns actually used
- Modify: async request handlers — move blocking Polars/Pandas reads into a thread executor
- Split: `backend/monitoring/router.py` (1,970 lines / 21 endpoints) into focused routers
- Test: `tests/test_job_cancellation_race.py`, `tests/test_upload_limits.py`, `tests/test_pipeline_version_uniqueness.py`

**Interfaces:** Consumes existing job/celery plumbing. Produces `JobLease(job_id, worker_id, expires_at)` and a `reaper` entry point run on a schedule.

**Effort:** 10 days. **Version impact:** backend minor.

- [ ] **Step 1: Write the failing race test** — enqueue, cancel, then let a worker pick up the message; assert the job ends `cancelled`, never `completed`.
- [ ] **Step 2: Run it.** Expected: FAIL (the job trains anyway).
- [ ] **Step 3: Implement the lease/state-guard, then the remaining items.**
- [ ] **Step 4: Do NOT change the orphaned-dataset-file behaviour** — per the master fix list's "What NOT to do", it is a deliberate, log-visible tradeoff. Add only a reconciliation sweep on top of it.

### Task R9.3: Security & scale fixes

**Files:**
- Create: `backend/utils/url_guard.py` — one shared allow-list/SSRF guard
- Modify: the EDA S3 `endpoint_url` path (SEC-01) and pipeline resolution's nested `client_kwargs.endpoint_url` (SEC-02) — both call the shared guard
- Create: `backend/middleware/quotas.py` — per-user resource quotas (queued/running jobs, stored bytes, CPU time); the IP-only rate limiter is not a substitute
- Modify: `docker-compose.yml` + docs — explicit concurrent Celery worker deployment with separate queues, never the default `solo` pool
- Modify: dataset preview/result tables — paginate/virtualise via R5.1's `DataTable`
- Modify: `backend/data_ingestion/service.py` — input/RSS memory budgets with a clear error above the budget
- Test: `tests/test_url_guard.py`, `tests/test_quotas.py`

**Interfaces:** Consumes settings. Produces `assert_safe_url(url: str, *, allow_hosts: set[str]) -> str` (raises `SkyulfSecurityError`) and `QuotaExceeded` (HTTP 429 with a `Retry-After`).

**Effort:** 7 days. **Version impact:** backend minor.

- [ ] **Step 1: Write the failing SSRF test** posting `endpoint_url=http://169.254.169.254/latest/meta-data/` through both paths and asserting a rejection.
- [ ] **Step 2: Run it.** Expected: FAIL (request is attempted today).
- [ ] **Step 3: Implement one guard, called from both sites** — SEC-01 and SEC-02 are the same fix.
- [ ] **Step 4: Verify quotas are principal-keyed**, so they still work correctly once R10 introduces real identities.

### Task R9.4: API contract hardening

**Files:**
- Modify: `backend/main.py` and every router — mount under `/api/v1`, keeping unprefixed routes as deprecated aliases that emit a `Deprecation` header
- Create: `.github/workflows/` step — generate frontend TypeScript types from the backend OpenAPI spec and fail CI on drift
- Modify: `JobInfo` schema — fix `created_at` nullability, add `preview`, rename `output` → `output_artifact_id` consistently on both sides
- Modify: EDA job-status handling — a real typed union with exhaustive mapping tests, replacing the `as JobStatus` force-cast
- Create: Zod runtime validation of WebSocket message envelopes on the frontend (Zod is already a dependency), with versioned fixture tests
- Create: `docs/api-versioning.md` — the breaking-change/deprecation policy
- Test: `tests/test_openapi_drift.py`, frontend `wsEnvelope.test.ts`

**Interfaces:** Produces `/api/v1/*` as the supported surface and a generated `frontend/ml-canvas/src/lib/api/generated.ts` that no human edits.

**Effort:** 7 days. **Version impact:** backend minor (aliases keep it non-breaking **in this release**; R10 removes the aliases, which is why R10 is the major bump), frontend minor.

- [ ] **Step 1: Write the failing drift test** comparing the committed generated types against a fresh generation from the live OpenAPI spec.
- [ ] **Step 2: Run it.** Expected: FAIL, showing the three known `JobInfo` drifts.
- [ ] **Step 3: Fix the drifts, wire the generator into CI, add the Zod envelopes.**
- [ ] **Step 4: Verify the deprecated aliases still work** so this release breaks no existing client.

### Task R9.5: Accessibility

**Files:**
- Modify: the node palette — convert inert `<div>`s to focusable, keyboard-activatable buttons (drag becomes an enhancement, not the only path)
- Create: a keyboard node-connection flow (select source port → select target port) with labelled ports
- Modify: icon-only controls — add `aria-label`s; upload progress — proper `role="progressbar"` semantics
- Modify: the axe CI configuration — promote `serious` violations to blocking (only `critical` fails today)
- Test: a Playwright spec building a complete pipeline using **only** the keyboard

**Interfaces:** Produces no new API surface. Produces the evidence base for a future VPAT (which Track B's compliance work would otherwise have to create from scratch).

**Effort:** 7 days. **Version impact:** frontend minor.

- [ ] **Step 1: Write the failing keyboard-only Playwright spec** — Tab to the palette, Enter to add two nodes, keyboard-connect them, run.
- [ ] **Step 2: Run it.** Expected: FAIL — the core build-a-pipeline flow is provably impossible via keyboard today.
- [ ] **Step 3: Implement the focusable palette and the port-selection connection flow.**
- [ ] **Step 4: Flip axe `serious` to blocking and fix what it reports** on the pages touched in R8 and R9.

### Task R9.6: Testing/CI foundations + remaining page redesigns

**Files:**
- Create: `tests/integration/test_full_pipeline.py` — a required API → DB → broker/worker → artifact → inference test on Docker Compose (Postgres/Redis/worker/MinIO), replacing the skipped, machine-specific inference test
- Create: a real canvas drag/connect Playwright spec (the current one seeds graph state via a dev hook, bypassing the interaction entirely)
- Modify: CI — coverage gates/ratchets for backend and frontend
- Create: `tests/test_job_service.py`, `tests/test_pipeline_versions_service.py` (retry/cancel/ownership/failure-state)
- Modify: `frontend/ml-canvas/src/pages/DataSources.tsx`, `DataDriftPage.tsx`, `ModelRegistry.tsx` — the remaining Phase 5 redesigns (catalog + asset detail; Overview/Analysis split with triage table; unified lifecycle shell with preflight checks)
- Test: as listed

**Interfaces:** Consumes `docker-compose.test.yml` (already in the repo). Produces a CI job that runs the integration suite on every PR.

**Effort:** 10 days. **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the integration test first**, then delete the skipped machine-specific one.
- [ ] **Step 2: Run it.** Expected: FAIL until Compose services are wired into CI.
- [ ] **Step 3: Implement, then add the coverage ratchet** at the current measured level (never above — a ratchet that fails on day one gets disabled).
- [ ] **Step 4: Complete the three page redesigns.**

### Task R9.7: Release cut R9

**Files:** Create `changelog/0.13.x.md` with `## v0.13.0 — Hardening, Contracts & Accessibility`; modify `pyproject.toml` and `package.json` only.

**Effort:** 1 hour. **Version impact:** backend `0.12.0`→`0.13.0`, frontend `0.13.0`→`0.14.0`, core **stays 0.10.0**.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=0.13.0`, `FRONTEND_VER=0.14.0`, `CORE_VER=` *(skip)*, `SERIES_FILE=changelog/0.13.x.md`, `TITLE=Hardening, Contracts & Accessibility`. **Skip Step F.**

**Gate:** the keyboard-only pipeline spec passes, the Compose integration test passes in CI, the OpenAPI drift check is green, and the cancellation-race test passes 100 consecutive runs.

---

## Release R10 — v1.0: Self-Hostable & Authenticated

**Versions:** backend `1.0.0` (**MAJOR**) · frontend `1.0.0` (**MAJOR**) · skyulf-core **unchanged (0.10.0)**
**Effort:** ~8 weeks
**Draws from:** Phase 0 (real authentication replacing hardcoded `user_id=1`, wiring the already-declared-but-uninstalled `passlib`/`bcrypt`/`python-jose`; PostgreSQL mandatory + Alembic migrations replacing the SQLite default and exception-swallowing `ALTER` statements; encrypted managed object storage; the `DataSource.has_permission()` placeholder; signed/verified or non-pickle artifact loads), Phase 1 (secrets-manager integration, production Dockerfile without `--reload`, real Compose/Helm reference with health probes, structured JSON logging + Prometheus metrics + OpenTelemetry tracing, readiness probes that actually check DB/Redis, `/v1` finalisation, Redis-backed rate limiting keyed on principal, mounted security-headers middleware), Phase 16a (backup/DR runbook with RPO/RTO targets).
**Why this is MAJOR and why it is in Track A, not Track B:** requiring authentication, removing the unprefixed route aliases, and dropping SQLite-in-production are all breaking changes to a public contract — that is exactly what the semver rule reserves `X.0.0` for. And a login screen plus per-user ownership is a **self-hosting** requirement, not an enterprise upsell: any open-source deployment reachable on a network needs it. Track B builds *organisations* on top of users; it does not own authentication.
**Dependencies:** R9 (`/api/v1` aliases exist; removing them here is the breaking step). Also unblocks: code-escape-hatch Phase B (R19) and everything in Track B.

### Task R10.1: Real authentication and per-user ownership

**Files:**
- Create: `backend/auth/` — `models.py` (`User` with hashed password), `service.py` (register/login/refresh), `dependencies.py` (`current_user`), `_routers/auth.py`
- Modify: every router currently assuming `user_id=1` — take `current_user` via dependency injection
- Modify: `backend/data/models.py` — implement `DataSource.has_permission()` for real (per the master fix list's corrected attribution, this is a `DataSource` placeholder, not `User`)
- Modify: `requirements-fastapi.txt` — actually install `passlib[bcrypt]` and `python-jose[cryptography]`
- Create: `frontend/ml-canvas/src/pages/LoginPage.tsx` + auth token handling in the API client
- Test: `tests/test_auth.py`, `tests/test_authz_matrix.py` (an endpoint × role matrix, per Phase 11)

**Interfaces:** Produces `current_user: User` as a FastAPI dependency, `POST /api/v1/auth/login` → `{access_token, refresh_token, token_type}`, and an `owner_user_id` column on every user-owned table. Track B's `Organization`/`Workspace` model is designed to add `workspace_id` **alongside** `owner_user_id`, not replace it — this is the seam that keeps Track B additive.

**Effort:** 15 days. **Version impact:** backend MAJOR, frontend MAJOR.

- [ ] **Step 1: Write the failing authz-matrix test** asserting every mutating endpoint returns 401 without a token and 403 for a non-owner.
- [ ] **Step 2: Run it.** Expected: FAIL — everything currently succeeds as `user_id=1`.
- [ ] **Step 3: Implement auth, then migrate ownership** with a data migration assigning existing rows to a bootstrap admin user.
- [ ] **Step 4: Verify no hardcoded principal remains** — `grep -rn "user_id=1\|user_id = 1" backend/` returns nothing outside tests and the bootstrap migration.

### Task R10.2: PostgreSQL + Alembic + encrypted storage + safe artifact loads

**Files:**
- Create: `alembic/` (env, versions) with an initial migration capturing the current schema
- Modify: `backend/database/` — remove the SQLite production default and the exception-swallowing `ALTER` statements; fail fast on a non-Postgres production URL
- Modify: the S3/local artifact storage layer — SSE-KMS, per-user prefixes, IAM-role access instead of static keys, and retries with backoff
- Create: `backend/utils/artifact_signing.py` — sign artifacts on write, verify on read; refuse to `joblib.load` an unsigned artifact
- Modify: `Dockerfile` — remove `--reload` from the production stage
- Create: `deploy/compose/`, `deploy/helm/` — reference deployments with health probes and HPA
- Create: `docs/operations/backup-and-dr.md` — procedure plus explicit RPO/RTO targets
- Test: `tests/test_artifact_signing.py`, `tests/test_alembic_upgrade.py`

**Interfaces:** Produces `sign_artifact(bytes) -> bytes`, `verify_artifact(bytes) -> bytes` (raises `ArtifactSignatureError`), and an Alembic revision graph as the only schema-change mechanism.

**Effort:** 15 days. **Version impact:** backend MAJOR (SQLite-in-production removed).

- [ ] **Step 1: Write the failing test** asserting `verify_artifact` rejects a tampered payload and that loading an unsigned artifact raises rather than unpickling.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement signing, then Alembic, then the storage changes.**
- [ ] **Step 4: Verify `alembic upgrade head` reproduces the current schema exactly** on an empty Postgres, diffed against the live schema.

### Task R10.3: Production operating model

**Files:**
- Create: `backend/observability/` — structured JSON logging, Prometheus metrics endpoint, OpenTelemetry tracing setup
- Modify: `backend/health/` — readiness actually checks DB and Redis
- Modify: `backend/middleware/` — mount the security-headers middleware (declared but unmounted today); Redis-backed rate limiting keyed on the authenticated principal
- Create: `backend/config/secrets.py` — pluggable secrets backend (env / Vault / AWS Secrets Manager)
- Modify: `backend/main.py` — remove the deprecated unprefixed route aliases added in R9.4
- Test: `tests/test_readiness.py`, `tests/test_security_headers.py`

**Interfaces:** Produces `/api/v1` as the **only** API surface, `/metrics`, and `/health/ready` returning 503 when a dependency is down.

**Effort:** 10 days. **Version impact:** backend MAJOR (alias removal).

- [ ] **Step 1: Write the failing test** asserting `/health/ready` returns 503 when Redis is unreachable and that a response carries the security headers.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement, then remove the aliases.**
- [ ] **Step 4: Verify the migration path is documented** — `docs/api-versioning.md` (R9.4) gets a "0.13 → 1.0" section listing every removed alias.

### Task R10.4: Release cut R10

**Files:** Create `changelog/1.0.x.md` with `## v1.0.0 — Self-Hostable & Authenticated`; modify `pyproject.toml` and `package.json`.

**Effort:** 2 hours (plus a written upgrade guide). **Version impact:** backend `0.13.0`→`1.0.0`, frontend `0.14.0`→`1.0.0`, core **stays 0.10.0**.

- [ ] **Step 1: Write `docs/upgrade/0.13-to-1.0.md`** listing every breaking change: authentication now required, `/api/v1` prefix mandatory, SQLite unsupported in production, artifacts must be re-signed (with a provided `python -m backend.utils.artifact_signing --resign-all` command).
- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=1.0.0`, `FRONTEND_VER=1.0.0`, `CORE_VER=` *(skip)*, `SERIES_FILE=changelog/1.0.x.md`, `TITLE=Self-Hostable & Authenticated`. **Skip Step F.**

**Gate:** a fresh Compose deployment comes up on Postgres, requires login, passes the authz matrix, serves `/metrics`, fails readiness when Redis is stopped, and the documented backup/restore procedure has been executed once end to end against real data.

---

## Release R11 — skyulf-core v1.0: Partitionable Calculator Contract

**Versions:** backend `1.1.0` (minor) · frontend **unchanged (1.0.0)** · skyulf-core `1.0.0` (**MAJOR**)
**Effort:** ~8 weeks
**Draws from:** Phase 9's two foundational items (**partitionable/stateless calculator contract** — XL; **versioned artifact schema/migration path** — completed in R4 and finalised here), Phase 9 "declarative per-node config validation (replace 246 ad-hoc `config.get` call sites)", Phase 17a's DuckDB/Narwhals L-effort items which are gated on exactly this contract.
**Why here and not earlier:** this is the single most consequential architectural item in the entire research corpus (`differentiation-strategy.md` Part 3's own closing summary), and it **blocks the Ray migration from working smoothly**. It is also XL and destabilising — doing it before R1–R10 would have meant shipping nothing visible for months from a 47-star standing start. Doing it after R6/R7 means the node set is broad enough that the contract is designed against real variety rather than a curated subset.
**Why MAJOR:** the calculator/applier base-class signature changes. That is a breaking change for anyone using `skyulf-core` directly — exactly what `X.0.0` is for.
**Dependencies:** R4 (artifact versioning — migrations are the escape hatch for the contract change), R6/R7 (node breadth), R3.3 (TypedDict config schemas are the input side of the declarative validation).

### Task R11.1: Partitionable/stateless calculator contract

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/base.py` — add `partial_fit(chunk, state) -> state` and `finalize(state) -> artifact` alongside the existing `fit`; `fit` becomes the default single-partition implementation over the new pair
- Create: `skyulf-core/skyulf/preprocessing/_partition.py` — `PartitionState` protocol and combiners for the common statistics (count/sum/sumsq/min/max/category-set/quantile-sketch)
- Modify: every calculator whose statistic is trivially combinable (scalers, simple imputation, one-hot category collection, min/max bucketing)
- Create: `skyulf-core/tests/test_partition_contract.py` — for every registered node, assert `fit(whole) == finalize(reduce(partial_fit, chunks))` within tolerance, or that the node explicitly declares `partitionable = False`
- Modify: `skyulf-core/docs/` — a migration guide for the base-class change

**Interfaces:** Produces `Calculator.partial_fit(chunk, state: PartitionState | None) -> PartitionState`, `Calculator.finalize(state: PartitionState) -> dict`, and the class attribute `partitionable: bool`. `Applier.apply` is unchanged. Ray's branch executor (R13) consumes `partitionable` to decide whether a node may be split across workers.

**Effort:** 25 days. **Version impact:** core MAJOR.

- [ ] **Step 1: Write the failing equivalence test** for `StandardScaler` — fit on the whole frame vs. three chunks combined; assert means/scales match within `1e-9`.
- [ ] **Step 2: Run it.** Expected: FAIL, `partial_fit` does not exist.
- [ ] **Step 3: Implement the protocol and the combiners, then migrate calculators family by family**, each with its own equivalence test.
- [ ] **Step 4: Declare `partitionable = False` explicitly** for nodes that genuinely cannot be partitioned (e.g. iterative imputation) — the goal is an honest, machine-readable contract, not universal partitionability.
- [ ] **Step 5: Verify no existing single-partition behaviour changed** — the whole pre-existing core suite must pass unmodified.

### Task R11.2: Declarative per-node config validation

**Files:**
- Modify: `skyulf-core/skyulf/config_validation.py` — a declarative `NodeConfigSpec` (fields, types, allowed values, cross-field rules) resolved from each node's TypedDict + `@node_meta`
- Modify: the ~246 ad-hoc `config.get` call sites across 54 files — replaced family by family with a validated, typed config object
- Test: `skyulf-core/tests/test_declarative_config.py`

**Interfaces:** Consumes R3.1's `require_config` (which becomes the low-level primitive) and R3.3's TypedDicts. Produces `validate_node_config(node_type: str, config: dict) -> dict` returning a normalised config with defaults filled, called once at the top of every `fit`.

**Effort:** 12 days. **Version impact:** core MAJOR (folded into 1.0.0 — error types and normalisation change).

- [ ] **Step 1: Write the failing test** asserting a cross-field rule (e.g. `n_bins` required when `method == "uniform"`) raises with both field names in the message.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement, migrating one node family per commit.**
- [ ] **Step 4: Track progress mechanically**

```bash
cd /Users/BH7043/Skyulf/skyulf-core && grep -rn "config.get(" skyulf/ | wc -l
```

Expected: strictly decreasing per commit; the release gate is a documented remaining count, not necessarily zero.

### Task R11.3: Backend adoption of the new core contract

**Files:**
- Modify: `backend/ml_pipeline/_execution/engine/_node_runners.py` — call `validate_node_config` and surface its errors through the R2.3 plain-English error path
- Modify: `requirements-fastapi.txt` — pin `skyulf-core>=1.0.0,<2.0.0`
- Test: `tests/test_node_runner_config_errors.py`

**Interfaces:** Consumes `validate_node_config` and `partitionable`. Produces no new API surface.

**Effort:** 5 days. **Version impact:** backend minor (adopts a new capability; no public contract breaks).

- [ ] **Step 1: Write the failing test** asserting a bad node config returns HTTP 422 with `{field, expected, suggestion}` rather than a 500.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement and re-pin core.**
- [ ] **Step 4: Run the Compose integration test from R9.6** to confirm nothing regressed end to end.

### Task R11.4: Release cut R11

**Files:** Add `## v1.1.0 — skyulf-core 1.0 Adoption` to `changelog/1.0.x.md`… **no** — the backend version is `1.1.0`, so the series file is `changelog/1.1.x.md`. Create it.

**Effort:** 2 hours. **Version impact:** backend `1.0.0`→`1.1.0`, frontend **stays 1.0.0**, core `0.10.0`→`1.0.0`.

- [ ] **Step 1: Write `skyulf-core/MIGRATION-0.10-to-1.0.md`** documenting the `partial_fit`/`finalize` addition, the `partitionable` attribute, and the config-error type change, with a before/after snippet for a custom third-party calculator.
- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=1.1.0`, `FRONTEND_VER=` *(skip)*, `CORE_VER=1.0.0`, `SERIES_FILE=changelog/1.1.x.md`, `TITLE=skyulf-core 1.0 — Partitionable Calculator Contract`. Step F applies (`git tag core-v1.0.0`).

**Gate:** every registered node either passes the partition-equivalence test or declares `partitionable = False`; the full pre-existing core suite passes unmodified; the Compose integration test is green.

---

## Release R12 — Ray Migration I (execution backend + attempt lifecycle)

**Versions:** backend `1.2.0` (minor) · frontend `1.1.0` (minor) · skyulf-core **unchanged (1.0.0)**
**Effort:** ~6 weeks
**Draws from:** `initiatives/ray-migration/2026-08-10-01-execution-backend-foundation-plan.md` (1,464 lines) and `2026-08-10-02-job-attempt-lifecycle-plan.md` (1,730 lines). **Do not re-derive these — they are already written task-by-task.** This release decides only *where they sit in the sequence* and *what they bump*.
**Why here:** Ray plan 01 is deliberately Ray-free (it introduces a backend-neutral submission contract behind which the existing local and Celery adapters keep working), so it is safe and non-breaking. It also needs R11's `partitionable` contract to be meaningful — without it, plan 04's distributed compute has nothing safe to split.
**Dependencies:** R11 (partitionable contract), R10 (Postgres mandatory — Ray production mode requires it per the ray-migration Global Constraints), R9.2 (job lease/heartbeat, which the attempt model builds on).

### Task R12.1: Execute ray-migration plan 01 — execution backend foundation

**Files:** As enumerated in `initiatives/ray-migration/2026-08-10-01-execution-backend-foundation-plan.md` (backend-neutral submission contract, config split, event-transport split).

**Interfaces:** Produces the `ExecutionBackend` submission interface with `local` and `celery` adapters behind it. Consumed by R12.2 and R13.

**Effort:** 15 days. **Version impact:** backend minor (new internal capability, existing public API paths and response shapes preserved per that plan's Global Constraints).

- [ ] **Step 1: Execute that plan's tasks in order**, honouring its **Foundation gate**: existing local and Celery behaviour passes behind the new interface before any Ray dependency is introduced.
- [ ] **Step 2: Verify the gate** — run the R9.6 Compose integration test with `EXECUTION_BACKEND=celery` and `EXECUTION_BACKEND=local`; both must pass with identical job outcomes.

### Task R12.2: Execute ray-migration plan 02 — job attempt lifecycle

**Files:** As enumerated in `initiatives/ray-migration/2026-08-10-02-job-attempt-lifecycle-plan.md` (durable execution attempts, `cancel-requested` state, retry lineage), plus the frontend job/attempt UI.

**Interfaces:** Produces the attempt model in which manual retry returns the **same logical job ID** and appends a physical attempt — an intentional identity change called out in that plan's Global Constraints. The frontend gains an attempt list per job.

**Effort:** 15 days. **Version impact:** backend minor, frontend minor (new attempt UI).

- [ ] **Step 1: Execute that plan's tasks in order**, honouring its **Lifecycle gate**: attempts, `cancel-requested`, and retries work with the *existing Celery adapter*, before Ray exists.
- [ ] **Step 2: Verify against R9.2's cancellation-race test** — the new state machine must not reintroduce the race.
- [ ] **Step 3: Update the frontend job views** to render attempts, reusing R5.1's `DataTable` and R8.4's `RunTimeline` (an attempt is a timeline scope, not a new page).

### Task R12.3: Release cut R12

**Files:** Create `changelog/1.2.x.md` with `## v1.2.0 — Execution Backend Abstraction & Job Attempts`; modify `pyproject.toml` and `package.json`.

**Effort:** 1 hour. **Version impact:** backend `1.1.0`→`1.2.0`, frontend `1.0.0`→`1.1.0`, core **stays 1.0.0**.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=1.2.0`, `FRONTEND_VER=1.1.0`, `CORE_VER=` *(skip)*, `SERIES_FILE=changelog/1.2.x.md`, `TITLE=Execution Backend Abstraction & Job Attempts`. **Skip Step F.**

**Gate:** the ray-migration Foundation gate and Lifecycle gate both pass, with Celery still the default backend.

---

## Release R13 — Ray Migration II (Ray Jobs runtime + distributed compute)

**Versions:** backend `1.3.0` (minor) · frontend **unchanged (1.1.0)** · skyulf-core **unchanged (1.0.0)**
**Effort:** ~7 weeks
**Draws from:** `initiatives/ray-migration/2026-08-10-03-ray-jobs-pipeline-runtime-plan.md` (1,090 lines), `2026-08-10-04-distributed-compute-plan.md` (794 lines), plus Phase 18b's **ASHA asynchronous successive-halving scheduler** (Li et al., MLSys 2020, arXiv:1810.05934) — the existing halving strategies are synchronous and will bottleneck under real distributed parallelism; Ray Tune ships the maintained scheduler.
**Dependencies:** R12 (both gates passed). Ray remains **opt-in**; Celery is still the default, so this release is not breaking.

### Task R13.1: Execute ray-migration plans 03 and 04

**Files:** As enumerated in those two plans (Ray Jobs adapter, pipeline entrypoint, Ray branch executor, joblib tuning integration, resource and artifact safety).

**Interfaces:** Produces `EXECUTION_BACKEND=ray` as a supported value and `resource_spec_for_job(job) -> ResourceSpec` — the same function DL's Phase 5 (R17) extends for per-job-type GPU sizing.

**Effort:** 25 days. **Version impact:** backend minor.

- [ ] **Step 1: Execute plan 03**, honouring its **Ray runtime gate**: a whole pipeline can be submitted, queried, stopped, and completed on a single-node Ray cluster.
- [ ] **Step 2: Execute plan 04**, honouring its **Compute gate**: Ray produces result parity *and a measured benefit* for branch or tuning workloads.
- [ ] **Step 3: Apply the migration's own Execution Rule** — if Ray does not provide a measurable benefit for the selected workloads, **stop here**: keep the backend abstraction from R12, do not proceed to R14, and do not remove Celery. Record the measurement in `initiatives/ray-migration/` rather than deleting the finding.
- [ ] **Step 4: Verify `partitionable = False` nodes are never split** across Ray workers — assert this in a test, since silently splitting a non-partitionable calculator produces a wrong artifact rather than an error.

### Task R13.2: ASHA scheduler for distributed tuning

**Files:**
- Modify: `skyulf-core/skyulf/modeling/_tuning/engine.py` consumers on the backend side — select Ray Tune's ASHA scheduler when `EXECUTION_BACKEND=ray`, keep the synchronous strategy otherwise
- Test: `tests/test_asha_scheduler.py`

**Interfaces:** Consumes `resource_spec_for_job`. Produces a `scheduler` field on the tuning job record recording which scheduler ran, so results across backends stay comparable.

**Effort:** 5 days. **Version impact:** backend minor (core is untouched — the scheduler is an orchestration choice, not a core algorithm).

- [ ] **Step 1: Write the failing test** asserting that with the Ray backend, a tuning job with 20 trials terminates early trials and records `scheduler == "asha"`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement via Ray Tune's maintained scheduler** — do not reimplement successive halving.
- [ ] **Step 4: Verify parity** — best-score distribution across 5 seeded runs is not worse than the synchronous strategy.

### Task R13.3: Release cut R13

**Files:** Create `changelog/1.3.x.md` with `## v1.3.0 — Ray Jobs Runtime & Distributed Compute`; modify `pyproject.toml` only.

**Effort:** 1 hour. **Version impact:** backend `1.2.0`→`1.3.0`, frontend **stays 1.1.0**, core **stays 1.0.0**.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=1.3.0`, `FRONTEND_VER=` *(skip)*, `CORE_VER=` *(skip)*, `SERIES_FILE=changelog/1.3.x.md`, `TITLE=Ray Jobs Runtime & Distributed Compute`. **Skip Step F.**

**Gate:** the Ray runtime gate and the Compute gate both pass, with a recorded, quantified speed-up for at least one real workload.

---

## Release R14 — Ray Migration III (operations + Celery removal)

**Versions:** backend `2.0.0` (**MAJOR**) · frontend `1.2.0` (minor) · skyulf-core **unchanged (1.0.0)**
**Effort:** ~6 weeks
**Draws from:** `initiatives/ray-migration/2026-08-10-05-operations-deployment-plan.md` (1,344 lines) and `2026-08-10-06-cutover-celery-removal-plan.md` (971 lines).
**Why MAJOR:** removing Celery changes the required deployment topology and removes a supported configuration value (`EXECUTION_BACKEND=celery`). Operators must change their infrastructure. That is a breaking change to a public contract.
**Dependencies:** R13's Compute gate passed with a measured benefit. If it did not, **this release does not happen** and the roadmap continues at R15 with Celery retained.

### Task R14.1: Execute ray-migration plans 05 and 06

**Files:** As enumerated in those plans (reconciliation, health, scheduler, Compose, observability; then parity rollout, default switch, Celery drain and removal), plus the frontend operational views for Ray job state.

**Interfaces:** Removes `EXECUTION_BACKEND=celery`. Produces the Ray-based deployment topology as the only supported production configuration.

**Effort:** 25 days. **Version impact:** backend MAJOR, frontend minor.

- [ ] **Step 1: Execute plan 05**, honouring its **Operations gate**: restart, orphan, storage, and scheduler behaviour is observable and recoverable.
- [ ] **Step 2: Execute plan 06**, honouring its **Cutover gate**: Ray becomes default only after production-like acceptance tests; Celery removal is the final, reversible-boundary change.
- [ ] **Step 3: Verify `celery_worker.py` and the Celery dependency are fully removed** — `grep -rn "celery" backend/ requirements-fastapi.txt` returns nothing outside historical changelog entries and the upgrade guide.
- [ ] **Step 4: Update every lint/type command that named `celery_worker.py`** — the repo-wide gate commands in this plan's Global Constraints, `.github/scripts/run_check.sh` invocations, and the ray-migration/deep-learning plans' constraint blocks all reference it explicitly and will fail on a missing path.

### Task R14.2: Release cut R14

**Files:** Create `changelog/2.0.x.md` with `## v2.0.0 — Ray Cutover & Celery Removal`; create `docs/upgrade/1.3-to-2.0.md`; modify `pyproject.toml` and `package.json`.

**Effort:** 2 hours. **Version impact:** backend `1.3.0`→`2.0.0`, frontend `1.1.0`→`1.2.0`, core **stays 1.0.0**.

- [ ] **Step 1: Write `docs/upgrade/1.3-to-2.0.md`** — the required Ray cluster, the removed `EXECUTION_BACKEND=celery` value, the removed Redis-as-broker role (Redis remains for pub/sub), and a rollback procedure to 1.3.0.
- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=2.0.0`, `FRONTEND_VER=1.2.0`, `CORE_VER=` *(skip)*, `SERIES_FILE=changelog/2.0.x.md`, `TITLE=Ray Cutover & Celery Removal`. **Skip Step F.**

**Gate:** production-like acceptance tests pass on Ray, and the documented rollback to 1.3.0 has been rehearsed once.

---

## Release R15 — Deep Learning I (shared infra + tabular MLP)

**Versions:** backend `2.1.0` (minor) · frontend `1.3.0` (minor) · skyulf-core `1.1.0` (minor)
**Effort:** ~6 weeks
**Draws from:** `initiatives/deep-learning/2026-08-11-implementation-roadmap.md` Phases 0 and 1 (already written task-by-task, with a post-rubber-duck correction: `_run_training` needs exactly **one** new direct-fit dispatch branch, parallel to the existing clustering branch — the originally-proposed `DLTrainingManager`/`JobStrategyFactory` design is not implementable), plus `2026-08-11-architecture-design.md` §4.3 and `2026-08-11-frontend-design.md`.
**Why here — all four prerequisites are now satisfied, and none were before:**
1. `pipeline_schema_version` + migration registry (R9.1) — DL adds exactly the kind of new node types that break old saved pipelines without it.
2. `artifact_schema_version` (R4.1) — DL adds a new `.pt` artifact format.
3. Partitionable calculator contract (R11.1) — the master fix list says explicitly to do this **before piling on more node types, including DL**.
4. Ray (R13/R14) — DL Phase 5's GPU scheduling depends on it, and the Celery `solo` pool constraint that made multi-epoch DL jobs block the whole queue is gone.
**Dependencies:** R9.1, R4.1, R11.1, R14 (or R13 if the Ray cutover was correctly abandoned — DL phases 0–4 ship and are useful standalone either way).

### Task R15.1: Execute deep-learning Phases 0 and 1

**Files:** As enumerated in that roadmap — `skyulf-core/skyulf/deep_learning/{__init__,base,_training_loop}.py`, the tabular MLP nodes, the single `_run_training` direct-fit branch in `backend/ml_pipeline/_execution/engine/_node_runners.py`, the `.pt` artifact format, `RUN_MODE_TRAINING_TYPES` additions in `frontend/ml-canvas/src/core/utils/pipelineConverter.ts`, and the DL settings panel (built on R5.3's `NodeSettingsForm`, which did not exist when that roadmap was written — use it rather than writing another bespoke panel).

**Interfaces:** Produces `BaseDLCalculator` (with `is_deep_learning = True` as the marker `_run_training` detects, avoiding an `isinstance` import cycle), `BaseDLApplier`, `resolve_device(preferred) -> torch.device`, `TrainingLoopConfig`, and `run_training_loop(model, train_ds, val_ds, config, progress_callback, log_callback) -> TrainingResult` — where `progress_callback(current_epoch, total_epochs, score=...)` is the per-epoch seam that R17's live curves consume.

**Effort:** 25 days. **Version impact:** core minor (new `deep_learning` module + `dl` extra), backend minor (one dispatch branch, no API shape change), frontend minor (new node types + settings panel).

- [ ] **Step 1: Execute the DL roadmap's Phase 0 tasks**, honouring its **Phase 0 gate**: a tabular MLP trains end-to-end through the real job pipeline (submit → epochs → `.pt` artifact → loadable → predict), with cancel and log behaviour identical to an existing sklearn node.
- [ ] **Step 2: Verify the `torch.load` safety constraint** — `grep -rn "torch.load" skyulf-core/ backend/` must show `weights_only=True` at every call site, each carrying the same arbitrary-code-execution warning docstring the existing `joblib.load` wrappers have.
- [ ] **Step 3: Execute Phase 1**, honouring its **Phase 1 gate**: MLP classifier/regressor pass accuracy/R² sanity checks on iris/diabetes comparable to an sklearn baseline, and tuned mode (Optuna over lr/batch-size/preset) completes.
- [ ] **Step 4: Verify the Sync Rule and the CI budget** — new frontend dropdowns cross-checked against backend allow-lists via R6.5's metadata sync test; `requirements-ci.txt` explicitly installs the `dl` extra wherever a gate actually trains, with the added install/runtime cost measured, not assumed free.
- [ ] **Step 5: Confirm no default-install weight gain** — in a clean venv, `pip install skyulf-core==1.1.0` must not pull `torch`.

### Task R15.2: Release cut R15

**Files:** Create `changelog/2.1.x.md` with `## v2.1.0 — Deep Learning: Tabular`; modify all three version files.

**Effort:** 1 hour. **Version impact:** backend `2.0.0`→`2.1.0`, frontend `1.2.0`→`1.3.0`, core `1.0.0`→`1.1.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=2.1.0`, `FRONTEND_VER=1.3.0`, `CORE_VER=1.1.0`, `SERIES_FILE=changelog/2.1.x.md`, `TITLE=Deep Learning: Tabular`. Step F applies (`git tag core-v1.1.0`).
- [ ] **Positioning note for the changelog copy** (Phase 18b, Grinsztajn et al., arXiv:2207.08815): trees still beat DL on typical tabular sizes. Scope the DL messaging at genuinely DL-favourable regimes (large data, embeddings, multi-modal) — do **not** market it as a blanket accuracy upgrade over the existing XGBoost/LightGBM stack.

**Gate:** DL Phase 0 and Phase 1 gates pass; a saved pre-DL pipeline still loads (R9.1's migration proves it).

---

## Release R16 — Deep Learning II (text + time-series)

**Versions:** backend `2.1.1` (patch) · frontend `1.4.0` (minor) · skyulf-core `1.2.0` (minor)
**Effort:** ~6 weeks
**Draws from:** deep-learning roadmap Phases 2 (transformer text classification via a fine-tuned pretrained encoder) and 3 (windowing transform + LSTM/TCN forecaster), plus Phase 18c's two cheapest diagnostics: **Integrated Gradients "Explain Prediction" node via Captum** (Sundararajan et al. 2017, arXiv:1703.01365) and the **LR Range Finder pre-flight node** (Smith 2017, arXiv:1506.01186).
**Why the backend bump is only a patch:** Phases 2 and 3 add new *core* node types and new *frontend* components, but the backend's single direct-fit dispatch branch from R15 already handles them. The only backend change is the `skyulf-core` pin and the `dl` extra's dependency list — a patch by this plan's semver rules. This is a deliberate demonstration that the three lines move independently.
**Dependencies:** R15 (Phase 0 gate).

### Task R16.1: Execute deep-learning Phases 2 and 3

**Files:** As enumerated in that roadmap — the transformer text-classification node, the time-series windowing transform and LSTM/TCN forecaster, matching `skyulf-core/skyulf/preprocessing/time_series/`'s existing windowing conventions.

**Interfaces:** Consumes `BaseDLCalculator`/`run_training_loop` (R15.1). Produces registry ids `dl_text_classifier`, `dl_sequence_forecaster`, plus a `WindowSpec` config shared with the existing time-series preprocessing nodes.

**Effort:** 20 days. **Version impact:** core minor, frontend minor.

- [ ] **Step 1: Execute Phase 2**, honouring its gate: fine-tunes on a small text dataset within the CI time budget (capped epochs/dataset size), produces sane accuracy, and falls back cleanly to CPU.
- [ ] **Step 2: Execute Phase 3**, honouring its gate: valid multi-step predictions on a synthetic seasonal series with windowing semantics matching the existing conventions.
- [ ] **Step 3: Verify forecasting consistency** — the DL forecaster's horizon/sort-column config keys must match R7.3's StatsForecast nodes exactly, so users are not learning two vocabularies for one concept.

### Task R16.2: Captum explainability + LR range finder

**Files:**
- Create: `skyulf-core/skyulf/deep_learning/explain.py` — `IntegratedGradientsCalculator/Applier` wrapping Captum
- Create: `skyulf-core/skyulf/deep_learning/lr_finder.py` — `LRRangeFinderCalculator` (a pre-flight node that suggests a learning rate instead of diagnosing a wasted run after the fact)
- Modify: `skyulf-core/setup.py` — `captum` under the `dl` extra
- Create: frontend schema entries for both, plus a chart for the LR-range curve reusing the existing Plotly components
- Test: `skyulf-core/tests/test_dl_explain.py`, `test_lr_finder.py`

**Interfaces:** Consumes a fitted `BaseDLApplier` artifact. Produces attribution arrays in the same shape the existing SHAP-based classical-ML explainability nodes emit, so the frontend renders both through one component.

**Effort:** 6 days. **Version impact:** core minor, frontend minor.

- [ ] **Step 1: Write the failing test** asserting attributions sum (within tolerance) to the difference between the model output at the input and at the baseline — Integrated Gradients' completeness axiom, which is the correct correctness check for this node.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement both as wrappers** over Captum and the documented LR-range recipe.
- [ ] **Step 4: Verify shape parity with the SHAP node output** so the frontend needs no branch.

### Task R16.3: Release cut R16

**Files:** Add `## v2.1.1 — Deep Learning: Text & Time-Series` to `changelog/2.1.x.md`; modify all three version files.

**Effort:** 1 hour. **Version impact:** backend `2.1.0`→`2.1.1`, frontend `1.3.0`→`1.4.0`, core `1.1.0`→`1.2.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=2.1.1`, `FRONTEND_VER=1.4.0`, `CORE_VER=1.2.0`, `SERIES_FILE=changelog/2.1.x.md`, `TITLE=Deep Learning: Text & Time-Series`. Step F applies (`git tag core-v1.2.0`).

**Gate:** DL Phase 2 and Phase 3 gates pass; the Integrated Gradients completeness test passes.

---

## Release R17 — Deep Learning III (image, GPU via Ray, live training curves)

**Versions:** backend `2.2.0` (minor) · frontend `1.5.0` (minor) · skyulf-core `1.3.0` (minor)
**Effort:** ~7 weeks
**Draws from:** deep-learning roadmap Phase 4 (image ingestion as a new modality + CNN/transfer-learning classifier) and Phase 5 (GPU scheduling via Ray — extend `resource_spec_for_job` for per-job-type GPU sizing), plus `initiatives/training-visualization/2026-08-11-feasibility-and-plan.md` tier (b): live loss/metric/LR curves via epoch-end telemetry over the existing WebSocket job-events channel, gradient-norm/LR series, bounded validation confusion snapshots, and the DL embedding-separation view.
**Why the live curves land here and not in R8:** the training-visualization study's verdict is explicit — do **not** promise "live" curves for ordinary sklearn `.fit()` calls, which are not iterative. R8 shipped the honest classical-ML answer (fast post-fit diagnostics). Genuinely live telemetry becomes real only now that the DL direct-fit path with a per-epoch `progress_callback` exists.
**Dependencies:** R15 (the `progress_callback` seam), R13/R14 (Ray — DL Phase 5's gate is explicitly conditional on the Ray migration's own Compute gate having independently passed).

### Task R17.1: Execute deep-learning Phase 4 (image)

**Files:** As enumerated in that roadmap — image ingestion (zip/folder + label CSV, streamed via `DataLoader` without loading the full set into memory) and the CNN/transfer-learning classifier with a frozen backbone.

**Interfaces:** Produces registry id `dl_image_classifier` and an `ImageDatasetSpec` ingestion config.

**Effort:** 15 days. **Version impact:** core minor, backend minor (new ingestion modality), frontend minor.

- [ ] **Step 1: Execute Phase 4**, honouring its gate: trains on a few hundred thumbnails via frozen-backbone transfer learning on CPU within a reasonable time budget, streaming rather than loading everything into memory.
- [ ] **Step 2: Verify the memory budget from R9.3 applies** to image ingestion too — a 50GB image archive must produce a clear quota error, not an OOM.

### Task R17.2: Execute deep-learning Phase 5 (GPU scheduling via Ray)

**Files:** Modify `resource_spec_for_job` for per-job-type GPU sizing; route DL jobs through it.

**Interfaces:** Consumes R13.1's `resource_spec_for_job(job) -> ResourceSpec`. Produces `ResourceSpec.num_gpus` honoured by the Ray scheduler.

**Effort:** 8 days. **Version impact:** backend minor.

- [ ] **Step 1: Confirm the precondition** — DL Phase 5's gate may only be entered once the Ray migration's Compute gate (plan 04) has independently passed. If Ray was abandoned at R13.3 Step 3, **skip this task entirely** and ship R17 without it; DL phases 0–4 remain useful standalone.
- [ ] **Step 2: Implement per-job-type GPU sizing.**
- [ ] **Step 3: Verify the gate** — DL jobs declaring `num_gpus=1` are observed running on a GPU-capable Ray worker with a measured speed-up over CPU for at least the image and transformer nodes.

### Task R17.3: Live training telemetry and curves

**Files:**
- Create: `skyulf-core/skyulf/deep_learning/telemetry.py` — `TrainingMetricSnapshot` TypedDict with an explicit `schema_version`, plus cadence and payload-size limits
- Modify: `run_training_loop`'s `progress_callback` consumer in the backend — publish snapshots onto the existing WebSocket job-events channel
- Create: `frontend/ml-canvas/src/pages/jobs/TrainingCurves.tsx` — live loss/metric/LR curves, gradient-norm series (global norm only by default; per-layer activation stats are an advanced debug mode, not default UI), bounded validation-confusion snapshots at a fixed epoch cadence, and a PCA embedding-separation view over penultimate-layer embeddings on a bounded sample
- Test: `skyulf-core/tests/test_telemetry_schema.py`, `TrainingCurves.test.tsx`

**Interfaces:** Produces `TrainingMetricSnapshot = {schema_version: int, epoch: int, total_epochs: int, train_loss: float, val_loss: float | null, metrics: dict[str, float], lr: float, grad_norm: float | null, ts: str}` — validated on the frontend by R9.4's Zod envelope layer.

**Effort:** 10 days. **Version impact:** core minor, backend minor, frontend minor.

- [ ] **Step 1: Write the failing test** asserting a 100-epoch run emits at most the configured cadence's number of snapshots and that each payload stays under the size limit (a 200-class confusion matrix must be bounded or omitted, not streamed every epoch).
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement the schema, the publisher, and the chart.**
- [ ] **Step 4: Label the embedding view correctly** — for DL this is a genuine learned representation, unlike R8.4's classical-ML input-space PCA projection. Assert both label strings in tests so they cannot be conflated later.

### Task R17.4: Release cut R17

**Files:** Create `changelog/2.2.x.md` with `## v2.2.0 — Deep Learning: Image, GPU & Live Training Curves`; modify all three version files.

**Effort:** 1 hour. **Version impact:** backend `2.1.1`→`2.2.0`, frontend `1.4.0`→`1.5.0`, core `1.2.0`→`1.3.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=2.2.0`, `FRONTEND_VER=1.5.0`, `CORE_VER=1.3.0`, `SERIES_FILE=changelog/2.2.x.md`, `TITLE=Deep Learning: Image, GPU & Live Training Curves`. Step F applies (`git tag core-v1.3.0`).

**Gate:** DL Phase 4 gate passes; Phase 5 gate passes **or** is documented as skipped with the reason; live curves render within one epoch of training start.

---

## Release R18 — Deployment & Registry Maturity

**Versions:** backend `2.3.0` (minor) · frontend `1.6.0` (minor) · skyulf-core `1.4.0` (minor)
**Effort:** ~7 weeks
**Draws from:** Phase 16a (model registry allows only one globally-"live" deployment platform-wide — deploying any model un-deploys every other one; no `pipeline_id`/environment scoping), Phase 9 Bet #4 (deployment/serving DX: prediction telemetry, performance-decay monitoring, canary/champion-challenger), Phase 18d (multivariate/joint drift with "typifying exemplar" surfacing via `alibi-detect`, Rabanser et al. NeurIPS 2018, arXiv:1810.11953; multi-seed variance "Reproducibility Score", Pineau et al. JMLR 2020, arXiv:2003.12206), Phase 16c/Phase 9 (concept/performance drift and calibration diagnostics missing from core).
**Why here:** the single-global-live-model limitation is a real product defect for OSS users, not only an enterprise gap — but fixing it well needs the identity model (R10) and the observability stack (R10.3) to attach telemetry to. Phase 18d's statistical drift work is deliberately **after** R7's schema contracts, per TFDV's production evidence that schema anomalies catch more real incidents than distribution-distance tests.
**Dependencies:** R10 (identity + observability), R7 (schema contracts land first).

### Task R18.1: Model registry scoping and progressive delivery

**Files:**
- Modify: `backend/.../deployment/service.py` — replace the single global "active" model with `(pipeline_id, environment, traffic_weight)` scoping; deploying a model must no longer deactivate unrelated models
- Create: `backend/.../deployment/progressive.py` — canary and champion/challenger traffic splitting with a documented rollback
- Modify: `frontend/ml-canvas/src/pages/ModelRegistry.tsx` — per-pipeline/environment lifecycle view with preflight checks
- Create: an Alembic migration adding the scoping columns and backfilling the existing single active model into `(its pipeline, "production", 100)`
- Test: `tests/test_deployment_scoping.py`, `tests/test_progressive_delivery.py`

**Interfaces:** Produces `Deployment(pipeline_id, environment: str, model_version_id, traffic_weight: int)` with the invariant that weights within one `(pipeline_id, environment)` sum to 100.

**Effort:** 15 days. **Version impact:** backend minor (additive scoping with a backfill — existing single-model deployments keep working), frontend minor.

- [ ] **Step 1: Write the failing test** asserting that deploying model B for pipeline 2 leaves pipeline 1's model A active.
- [ ] **Step 2: Run it.** Expected: FAIL — model A is deactivated today.
- [ ] **Step 3: Implement scoping, then progressive delivery.**
- [ ] **Step 4: Verify the backfill migration** on a database snapshot containing one active model.

### Task R18.2: Prediction telemetry and performance decay

**Files:**
- Create: `backend/monitoring/prediction_telemetry.py` — record prediction volume, latency, input-schema violations (reusing R7.2's `SchemaDriftReport`) and, when ground truth arrives, accuracy decay
- Create: `skyulf-core/skyulf/profiling/performance_drift.py` — accuracy/calibration decay over time given `(prediction, outcome)` pairs
- Create: `skyulf-core/skyulf/profiling/calibration.py` — Brier score, ECE, reliability curve (the calibrated-classifier node exists today with no way to verify calibration)
- Modify: the drift pages — a decay panel alongside the existing data-drift views
- Test: `skyulf-core/tests/test_calibration.py`, `test_performance_drift.py`, `tests/test_prediction_telemetry.py`

**Interfaces:** Produces `calibration_report(y_true, y_prob) -> {brier: float, ece: float, curve: [{bin, mean_pred, frac_pos}]}` and `performance_drift(window_a, window_b) -> {metric, delta, significant: bool}`.

**Effort:** 12 days. **Version impact:** core minor, backend minor, frontend minor.

- [ ] **Step 1: Write the failing calibration test** asserting a deliberately overconfident classifier scores a worse ECE than its calibrated counterpart.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement calibration, then decay, then the telemetry recorder.**
- [ ] **Step 4: Verify telemetry is bounded** — a high-volume `/predict` endpoint must aggregate, not write one row per prediction.

### Task R18.3: Multivariate drift + Reproducibility Score

**Files:**
- Create: `skyulf-core/skyulf/profiling/multivariate_drift.py` — MMD / classifier-based two-sample test via `alibi-detect`, plus "typifying exemplar" surfacing (show the rows most responsible for the detected shift), reusing existing fitted scaler/PCA-like artifacts
- Create: `skyulf-core/skyulf/profiling/reproducibility.py` — `reproducibility_score(runs: list[RunResult]) -> {mean, std, cv, n_seeds}` operationalising multi-seed variance reporting
- Modify: `skyulf-core/setup.py` — `alibi-detect` under the `research` extra
- Modify: the Experiments page — show the reproducibility score next to the headline metric
- Test: `skyulf-core/tests/test_multivariate_drift.py`, `test_reproducibility.py`

**Interfaces:** Produces `multivariate_drift(ref, cur) -> {p_value: float, drifted: bool, exemplars: list[dict]}` and `reproducibility_score(...)`.

**Effort:** 8 days. **Version impact:** core minor, frontend minor.

- [ ] **Step 1: Write the failing test** asserting a joint shift that leaves every marginal distribution unchanged is still detected (this is precisely what univariate KS/PSI misses and why this node exists).
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement by wrapping `alibi-detect`** — do not reimplement MMD.
- [ ] **Step 4: Verify the reproducibility score refuses to report on fewer than 3 seeds** rather than emitting a meaningless standard deviation.

### Task R18.4: Release cut R18

**Files:** Create `changelog/2.3.x.md` with `## v2.3.0 — Deployment & Registry Maturity`; modify all three version files.

**Effort:** 1 hour. **Version impact:** backend `2.2.0`→`2.3.0`, frontend `1.5.0`→`1.6.0`, core `1.3.0`→`1.4.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=2.3.0`, `FRONTEND_VER=1.6.0`, `CORE_VER=1.4.0`, `SERIES_FILE=changelog/2.3.x.md`, `TITLE=Deployment & Registry Maturity`. Step F applies (`git tag core-v1.4.0`).

**Gate:** two models for two different pipelines are simultaneously live; a canary rollout and its rollback have both been exercised; joint drift is detected on the marginal-invariant fixture.

---

## Release R19 — Code-First Loop & Conditional i18n

**Versions:** backend `2.4.0` (minor) · frontend `1.7.0` (minor) · skyulf-core `1.5.0` (minor)
**Effort:** ~8 weeks
**Draws from:** Phase 9 Bet #3 (two-way notebook export/import — "graduate to code, don't leave"), Phase 17b ranked item #3 (a standalone, code-first `skyulf.pipeline.Sequence` usable with zero canvas dependency, exposing each step's artifact and intermediate dataframe for inline debugging), Phase 15a **Phase B only** (a constrained advanced-parameter editor over an allow-listed expression grammar compiled back to canonical params — **not** arbitrary Python), Phase 15b tier (c) (sklearn learning/validation curves as separate, explicitly-requested diagnostic jobs), Phase 14 (i18n architecture, browser support matrix, centralised numeric/metric formatting), Phase 18b (multi-objective tuning via Optuna's existing API; cross-run warm-starting via `enqueue_trial` over Skyulf's own historical `TrainingJob.best_params`/`best_score`/`model_type`).
**Why last, and what is conditional:** the i18n/RTL work is explicitly *conditional on an international GTM push* per `i18n-mobile-crossbrowser-audit.md` — ship the browser-support matrix and centralised number formatting (cheap, unconditional), and start the message-catalog architecture only if that push is real. Everything else here is genuine capability that benefits from every prior release.
**Explicitly excluded, permanently, from Track A:** code-escape-hatch **Phase C** (arbitrary custom-Python node execution). Per the feasibility study and the master fix list's "What NOT to do", it must not be built on shared workers under any circumstances, and requires tenancy plus a dedicated, adversarially-tested, network-isolated executor. It lives in Track B (E8) and nowhere else. Do not let Phase B scope-creep into it.

### Task R19.1: Standalone code-first `skyulf.pipeline.Sequence`

**Files:**
- Create: `skyulf-core/skyulf/pipeline_sequence.py` — `Sequence(steps).fit(X, y)`, `.transform(X)`, `.step_artifact(name)`, `.step_output(name)` (the intermediate dataframe, for inline debugging)
- Modify: `skyulf-core/README.md` — a code-first quickstart that never mentions the canvas
- Test: `skyulf-core/tests/test_pipeline_sequence.py`

**Interfaces:** Consumes R11.1's calculator contract and R3.2's sklearn adapter (a `Sequence` must be usable *inside* an sklearn `Pipeline` and must accept sklearn transformers as steps). Produces `Sequence`, `SequenceStep(name, calculator, applier, config)`.

**Effort:** 10 days. **Version impact:** core minor.

- [ ] **Step 1: Write the failing test** asserting `seq.step_output("scale")` returns the dataframe *after* the scale step and `seq.step_artifact("scale")` returns its JSON artifact — the exact debuggability practitioners hand-roll their own step classes to get (Phase 17b §1.2/§7.1).
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement, reusing `FeatureEngineer`'s existing step-execution logic** rather than writing a second execution path.
- [ ] **Step 4: Verify leakage safety is preserved** — `Sequence.transform` on the fit data triggers R4.3's overlap warning.

### Task R19.2: Two-way notebook loop

**Files:**
- Create: `backend/ml_pipeline/_internal/_routers/notebook_import.py` — `POST /api/v1/pipelines/import-notebook` parsing an exported notebook back into a canvas graph as a new pipeline version
- Modify: `backend/ml_pipeline/_internal/_routers/_notebook_builders.py` — emit a machine-readable provenance cell (graph JSON + `pipeline_schema_version`) so import is a parse, not a guess
- Create: frontend import affordance on the Canvas page
- Test: `tests/test_notebook_roundtrip.py`

**Interfaces:** Produces a round-trip guarantee: export → import → export yields a byte-identical provenance cell.

**Effort:** 12 days. **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the failing round-trip test.**
- [ ] **Step 2: Run it.** Expected: FAIL (export is one-way today).
- [ ] **Step 3: Implement the provenance cell first, then the importer.**
- [ ] **Step 4: Verify hand-edited notebooks fail loudly** — if a user edits cells outside the provenance block, import must report exactly which steps could not be reconstructed rather than silently importing a stale graph.

### Task R19.3: Constrained advanced-parameter editor (code-escape-hatch Phase B)

**Files:**
- Create: `backend/ml_pipeline/expressions/grammar.py` — an allow-listed expression grammar (literals, column references, arithmetic, a fixed function allow-list) compiled back to canonical node params
- Create: `backend/ml_pipeline/expressions/compile.py` — `compile_expression(src: str, allowed_columns: set[str]) -> dict` raising `ExpressionRejected` for anything outside the grammar
- Create: frontend advanced-editor panel, labelled honestly as an **advanced transform editor**, not "Python"
- Test: `tests/test_expression_grammar.py` — property-based tests proving rejection of imports, statements, attribute access, dunder access, comprehension-based escapes, and lambda definitions

**Interfaces:** Produces `compile_expression(...) -> dict` whose output is an ordinary node param dict — the executor never sees a string of code.

**Effort:** 12 days. **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the property-based rejection tests first**, per the feasibility study's requirement that Phase B be threat-modelled with property tests proving rejected imports/statements/attribute-escapes **before** shipping.
- [ ] **Step 2: Run them.** Expected: FAIL (no grammar exists).
- [ ] **Step 3: Implement an allow-list parser over `ast`** — parse and walk, rejecting any node type not explicitly permitted. Never `eval`/`exec` the source.
- [ ] **Step 4: Verify there is no execution sink** — `grep -rn "eval(\|exec(\|compile(" backend/ml_pipeline/expressions/` returns only the `ast.parse` call. If any real execution sink appears, this task has become Phase C and must stop.

### Task R19.4: Tuning intelligence, diagnostic curve jobs, and unconditional i18n groundwork

**Files:**
- Modify: the tuning service — multi-objective optimisation (accuracy vs. latency/model size) via Optuna's existing multi-objective API (zero new dependencies; Optuna is already pinned), and cross-run warm-starting via `enqueue_trial` seeded from historical `TrainingJob.best_params`/`best_score`/`model_type`
- Create: `backend/.../diagnostics/curve_jobs.py` — sklearn learning-curve and validation-curve jobs as **separate, explicitly-requested** jobs (never inline — each retrains the estimator N times), showing an estimated fit count and runtime before the user commits
- Create: `frontend/ml-canvas/src/lib/format.ts` — centralised numeric/metric rendering (consistent significant digits, p-value/scientific-notation rules, `Intl.NumberFormat` for counts)
- Modify: `playwright.config.ts` — add Firefox and WebKit projects plus a tablet viewport; document the browser support matrix in `docs/browser-support.md`
- Create (**only if an international GTM push is real**): the i18n message-catalog architecture, locale persistence, `Intl`-based date/number formatting, and a string-extraction workflow. RTL is a separate, later workstream.
- Test: `tests/test_multiobjective_tuning.py`, `tests/test_warm_start.py`, `frontend/.../format.test.ts`

**Interfaces:** Produces `TuningObjective = {metric: str, direction: "maximize" | "minimize"}[]` and a Pareto-front result shape rendered by the Experiments page.

**Effort:** 12 days (excluding the conditional i18n work, which is Large on its own). **Version impact:** backend minor, frontend minor.

- [ ] **Step 1: Write the failing multi-objective test** asserting a Pareto front with at least two non-dominated trials is returned, not a single "best".
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement multi-objective and warm-starting**, recording in each job which historical runs seeded it so results stay explainable.
- [ ] **Step 4: Make the i18n decision explicitly** — write the answer into `docs/browser-support.md` and into the release changelog: either "international GTM is real, i18n architecture starts now" or "deferred, revisit at <named trigger>". An undecided item is the failure mode this audit called out; a written "no" is a valid outcome.

### Task R19.5: Release cut R19

**Files:** Create `changelog/2.4.x.md` with `## v2.4.0 — Code-First Loop & Tuning Intelligence`; modify all three version files.

**Effort:** 1 hour. **Version impact:** backend `2.3.0`→`2.4.0`, frontend `1.6.0`→`1.7.0`, core `1.4.0`→`1.5.0`.

- [ ] Run the **Global Procedure: Release Cut** with `BACKEND_VER=2.4.0`, `FRONTEND_VER=1.7.0`, `CORE_VER=1.5.0`, `SERIES_FILE=changelog/2.4.x.md`, `TITLE=Code-First Loop & Conditional i18n`. Step F applies (`git tag core-v1.5.0`).

**Gate:** export → import → export round-trips byte-identically; every expression-grammar rejection property test passes; the Playwright suite runs green on Chromium, Firefox, and WebKit.

---

## Track A — Items Deliberately Not Scheduled

Recording these explicitly so they are visible decisions, not oversights.

| Item | Source | Decision |
|---|---|---|
| Code escape hatch **Phase C** (arbitrary custom-Python node) | code-escape-hatch study; master-fix-list "What NOT to do" | **Never in Track A.** Requires tenancy plus a dedicated, network-isolated, adversarially-tested executor. Track B E8 only. |
| RTL support | Phase 14 | Deferred until i18n lands *and* Middle-East expansion is real. Not a fixed date. |
| Full lazy DuckDB execution path; internal Narwhals engine *replacement* | Phase 17a "What NOT to do" | L-effort; gated on R11's partitionable contract. Re-evaluate after R11 based on measured demand, not automatically. |
| TracIn per-example influence debugging; loss-landscape visualisation | Phase 18c | Explicitly labelled v2-roadmap candidates; defer past the initial DL module release. |
| Sampled t-SNE/UMAP views | Phase 15b tier (c) | Only if PCA (R8.4, R17.3) proves insufficient **in user research**. Never streamed; cache the artifact. |
| "Too many parameters / complexity fatigue" narrative | Phase 17b §6 | **Unverified** — GitHub/Reddit/G2 search was blocked during that research. Do not act on it until a follow-up pass with real search access corroborates it. |
| Changing the orphaned-dataset-file behaviour | backend-blockers §5; "What NOT to do" | Deliberate, log-visible tradeoff. R9.2 adds a reconciliation sweep on top; the logic itself is not "fixed". |
| OBOE-style cheap-probe model recommendation | Phase 18b | No maintained library to wrap; do only after R19.4's warm-starting proves the underlying data is useful. |

---

# TRACK B — Enterprise Roadmap (OPTIONAL)

> **This entire track is optional.** It is not a prerequisite for anything in Track A, it is not required for Skyulf to be a good product, and it should not be started on schedule — only on evidence. If it is never started, Track A still delivers a complete, self-hostable, authenticated, well-tested ML platform. Nothing in Track A is degraded to create demand for Track B.

## Why this track exists at all

The enterprise-readiness research is genuinely valuable and should not be thrown away — it found real gaps (no org model, no RBAC, no audit depth, no retention/DSAR workflow, no encryption-at-rest story, no backup/DR, no cost visibility, zero licensing/entitlement code anywhere). But the honest reading of the project's current position is:

- ~47 GitHub stars, one core maintainer, zero revenue, zero budget.
- No case studies, no reference customers, no SOC 2, no pen-test, no VPAT.
- `round6-gap-audit.md` confirmed at the implementation level that **no plan/tier/seat/usage-event model exists anywhere in the codebase** — so "enterprise" today is a research finding, not a product.

Building multi-tenancy, RBAC, SSO, billing enforcement and compliance artefacts *before* a single organisation has asked for them is the most expensive possible way to spend a solo maintainer's only scarce resource. Track A's work (correctness, first-run, library DX, leakage-safety, node breadth, self-hostability) is what produces the inbound demand that would justify Track B. So Track B waits.

## Track B — Gate (start condition, not a date)

**Do not write a single line of Track B code until BOTH of the following are true:**

**Hard technical prerequisite (non-negotiable):**
- Track A **R10** has shipped. Multi-tenancy is built *on top of* authentication and per-user ownership. Building organisations before users exist means building the same thing twice.

**Demand trigger — at least TWO of these three:**
1. **Three or more named organisations** are running self-hosted Skyulf and are reachable for a 30-minute call. Evidence: a written note per organisation (who, what they run it on, what they use it for). Not anonymous download counts.
2. **At least one inbound commercial conversation** with a named company, a named budget owner, and a written requirement that Track A cannot satisfy (typically SSO, RBAC, or an audit-log retention requirement). The existing `COMMERCIAL-LICENSE.md` process is the intake channel.
3. **Sustained adoption signal:** ≥500 GitHub stars **and** ≥1,000 monthly `skyulf-core` PyPI downloads, both sustained for three consecutive months. (The PyPI download badge is already in `README.md`, so this is measurable today with no new tooling.)

**If only trigger 2 fires, and it is a single company:** do **not** start Track B as a product. Scope a paid engagement instead (see Pricing → "Services"), deliver exactly what that customer needs, and only generalise it into Track B once a second customer asks for the same thing. One customer's requirement is a contract; two customers' identical requirement is a product.

**Re-evaluate the gate every quarter.** If none of the triggers has fired after a year of Track A releases, the correct conclusion is that Track B should not be built — not that the gate should be lowered.

## Track B — Packaging & Versioning Model

Track B uses a **fourth, independent version line** so it can never collide with Track A's numbers:

| Artifact | Path | Starting version | License |
|---|---|---|---|
| `skyulf-enterprise` (Python add-on package) | `enterprise/` (new top-level dir, own `pyproject.toml`) | `0.1.0` | Skyulf Enterprise License (source-available, not OSI) |
| `@skyulf/enterprise-ui` (frontend add-on package) | `frontend/enterprise-ui/` (own `package.json`) | `0.1.0` | Skyulf Enterprise License |

**How it attaches without forking Track A:**
- The Python add-on registers itself through a FastAPI router-discovery entry point and SQLAlchemy model registration; the OSS backend must remain fully functional with the package absent.
- The frontend add-on fills exactly one extension point: `AppShell`'s `slotOrgSwitcher` prop (created in Track A **R5.4** specifically for this purpose), plus route registration.
- Each Track B release declares a **minimum Track A version** as a dependency floor (e.g. `skyulf>=1.0.0,<3.0.0`). A floor is **not** a bump — Track B never changes `pyproject.toml`, `package.json`, or `setup.py` in Track A's directories.
- `skyulf-core` stays Apache-2.0 and is **never** touched by Track B. No enterprise gating ever enters the OSS library.

**Alternative if the maintainer prefers in-tree gating instead of a separate package:** allocate each Track B release the next unused **minor** on the Track A backend/frontend lines and record it in the Version Ledger before starting work, so no number is ever double-allocated. This plan recommends the separate-package model precisely because it makes double-allocation structurally impossible.

## Track B Releases

Each release below lists: what it delivers, its source phase, effort, its `skyulf-enterprise` version, and its Track A floor.

### E0 — Licensing & Commercial Decision (no code)

**Version:** none. **Effort:** 3 days. **Track A floor:** R10.
**Draws from:** Phase 16a (zero licensing/entitlement code exists), the existing `COMMERCIAL-LICENSE.md` and `CLA.md`.

- [ ] **Step 1: Confirm the CLA covers relicensing** — `CLA.md` already exists; verify it grants the rights needed to ship a source-available enterprise add-on alongside AGPLv3 code. If it does not, fix the CLA *before* accepting any further external contributions, not after.
- [ ] **Step 2: Write `enterprise/LICENSE`** (source-available terms: read/modify/self-host with a valid subscription; no redistribution; no removal of the license check).
- [ ] **Step 3: Publish the pricing page** using the table in the Pricing section below, with the "aspirational" items honestly framed.
- [ ] **Step 4: Set up manual invoicing** (Stripe payment links or plain invoices). **Do not build billing enforcement code yet** — E5 exists for that and should not be started before the second paying customer.

### E1 — Organizations, Workspaces & Membership

**Version:** `skyulf-enterprise 0.1.0`, `@skyulf/enterprise-ui 0.1.0`. **Effort:** ~6 weeks. **Track A floor:** backend ≥1.0.0, frontend ≥1.0.0.
**Draws from:** Phase 0 (multi-tenant/organization data model, `workspace_id` on every table), Phase 6 (Organization & Workspace Settings page, unified app shell org switcher).

- Deliverables: `Organization`, `Workspace`, `Membership` models; a `workspace_id` column added by an add-on Alembic branch to every user-owned table, defaulting existing rows into a bootstrap "Default" workspace; a workspace-scoped query dependency that the OSS routers consume through the existing `current_user` seam; per-workspace artifact-storage prefixes; the org switcher rendered into `AppShell.slotOrgSwitcher`; Organization & Workspace Settings pages.
- Files: `enterprise/skyulf_enterprise/tenancy/{models,service,dependencies}.py`, `enterprise/alembic/`, `frontend/enterprise-ui/src/OrgSwitcher.tsx`, `frontend/enterprise-ui/src/pages/OrgSettings.tsx`.
- Gate: with the add-on installed, a user in workspace A provably cannot read, list, or infer the existence of workspace B's datasets, pipelines, jobs, or artifacts — proven by an isolation test matrix, not by inspection. With the add-on **uninstalled**, the OSS backend still starts and every Track A test passes.

### E2 — RBAC, Member Management & Service Accounts

**Version:** `0.2.0`. **Effort:** ~4 weeks. **Track A floor:** backend ≥1.0.0.
**Draws from:** Phase 6 (Member/Role Management, API Keys/Service Accounts), Phase 16a (broken function-level authorization risk).

- Deliverables: role and permission models (Owner/Admin/Member/Viewer as defaults, custom roles as data); permission enforcement at the router-dependency layer, not scattered in handlers; member invite/remove flows; API keys and service accounts with scoped permissions and rotation.
- Files: `enterprise/skyulf_enterprise/rbac/`, `enterprise/skyulf_enterprise/service_accounts/`, `frontend/enterprise-ui/src/pages/Members.tsx`, `ApiKeys.tsx`.
- Gate: an endpoint × role matrix test (extending Track A R10.1's authz matrix) with an explicit assertion that **every** mutating endpoint is covered — an uncovered endpoint fails the test rather than silently passing.

### E3 — SSO (OIDC / SAML)

**Version:** `0.3.0`. **Effort:** ~4 weeks. **Track A floor:** backend ≥1.0.0, `skyulf-enterprise` ≥0.2.0.
**Draws from:** Phase 6 (Login/SSO page), Phase 0.

- Deliverables: OIDC and SAML identity-provider integration mapping external identities onto Track A's `User` and E1's `Membership`; just-in-time provisioning; a SCIM user-sync stub. SSO is the single most commonly named enterprise procurement requirement and is the most likely *first* reason a company pays.
- Files: `enterprise/skyulf_enterprise/sso/`, `frontend/enterprise-ui/src/pages/SsoLogin.tsx`.
- Gate: a full login round-trip against a real IdP (Keycloak in CI), including group-to-role mapping and a revoked-user lockout test.

### E4 — Audit Depth, Retention/DSAR & Encryption Governance

**Version:** `0.4.0`. **Effort:** ~6 weeks. **Track A floor:** backend ≥1.0.0, `skyulf-enterprise` ≥0.2.0.
**Draws from:** Phase 1 (append-only audit-event table, data-retention/deletion workflows with DSAR support), Phase 10 / `data-governance-audit.md` (the two Critical procurement blockers: retention/DSAR workflow and encryption at rest; DG-01 broaden PII detection beyond email/phone and add masking/tokenization rather than an advisory alert), Phase 6 (extend the **existing** `pages/AuditLogPage.tsx` — do not rebuild it), Phase 16a (backup/DR with RPO/RTO — the Track A R10.2 runbook is extended here with per-tenant restore).

- Deliverables: an append-only, tamper-evident audit-event store with actor/resource/before-after capture; configurable retention policies per workspace; a DSAR export-and-delete workflow with a verifiable completion record; PII detection beyond email/phone plus a masking/tokenization pipeline step; per-tenant encryption-key scoping on top of Track A's SSE-KMS storage; per-tenant restore procedures.
- Gate: a DSAR request for a named subject produces a complete export and a verified deletion across the database, object storage, and audit trail — with the audit trail itself retaining the tamper-evident record of the deletion.

### E5 — Licensing, Entitlement & Usage Metering

**Version:** `0.5.0`. **Effort:** ~5 weeks. **Track A floor:** backend ≥1.0.0, `skyulf-enterprise` ≥0.2.0.
**Draws from:** Phase 1 (usage-metering/entitlement service tied to the commercial license tier), Phase 16a (zero licensing/billing/entitlement code exists anywhere — no plan/tier/seat/usage-event models).

- Deliverables: `Plan`, `Subscription`, `Entitlement`, `UsageEvent` models; a signed offline license key (self-hosted customers cannot be assumed to have outbound internet); seat counting; quota enforcement per organisation built on Track A R9.3's quota middleware; a usage/billing dashboard.
- **Do not start this before the second paying customer.** With one customer, manual invoicing is strictly cheaper and carries zero maintenance burden. Enforcement code that guards revenue you do not yet have is pure cost.
- Gate: license expiry degrades gracefully to Community behaviour with a clear, non-destructive warning — it never deletes data, blocks exports, or locks a customer out of their own artifacts. A licensing system that can hold a customer's data hostage is a liability, not an asset.

### E6 — Cost / FinOps Visibility

**Version:** `0.6.0`. **Effort:** ~4 weeks. **Track A floor:** backend ≥1.3.0 (Ray per-job resource accounting), `skyulf-enterprise` ≥0.5.0.
**Draws from:** Phase 16a (no cost/FinOps visibility anywhere; no cost data computed or stored, including in the Ray-migration design) and the master fix list's explicit "What NOT to do": **do not bolt usage metering onto an already-built scheduler** — design it alongside Ray's per-job resource accounting.

- Deliverables: per-job CPU/GPU/memory/duration cost attribution derived from Ray's resource specs, rolled up per workspace, per pipeline, and per user; a cost panel; configurable budget alerts.
- **Sequencing warning:** if Track B starts *before* Track A R13, raise the resource-accounting requirement into the Ray plans at that time rather than retrofitting it here — retrofitting is materially more expensive, which is exactly the mistake the research doc names.
- Gate: the sum of per-job attributed costs for a workspace reconciles with the cluster's total resource-seconds within 5%.

### E7 — Governance & Promotion Workflows

**Version:** `0.7.0`. **Effort:** ~4 weeks. **Track A floor:** backend ≥2.3.0 (R18's registry scoping), `skyulf-enterprise` ≥0.2.0.
**Draws from:** Phase 16a (model registry scoping — the OSS half ships in Track A R18), Phase 9 Bet #4, `user-complaints-research.md` ("no Git-like collaboration — multiple people editing the same workflow without knowing of each other").

- Deliverables: approval workflows for production promotion (request → review → approve → deploy) with the audit trail from E4; environment-level promotion gates; concurrent-edit detection and conflict resolution on shared pipelines, building on Track A R5.2's `dirty | synced | conflict` indicator.
- Gate: an unapproved model cannot reach the production environment through any code path, proven by a negative test per promotion route.

### E8 — Isolated Code Executor (code-escape-hatch Phase C)

**Version:** `0.8.0`. **Effort:** ~10 weeks. **Track A floor:** backend ≥2.0.0, `skyulf-enterprise` ≥0.2.0.
**Draws from:** `initiatives/code-escape-hatch/2026-08-11-feasibility-and-security.md` Phase C, and the master fix list's most emphatic "What NOT to do".

- **Hard preconditions, all of which must be true before a single line is written:** E1 tenancy has shipped; a dedicated executor image exists with its **own** credentials, **no** shared DB/Redis/AWS access, and network default-deny; the executor has been adversarially tested by someone who did not build it.
- Deliverables: a `custom_python` node kind (never an arbitrary `step_type` or a `params["code"]` field routed through the existing generic runner); persisted source revision, runtime version, dependency allow-list, input/output schema contract, immutable provenance (generated base source plus user patch), and content hash; an executor protocol that materialises only the authorised input partition and accepts only validated tabular output (Arrow/Parquet/JSON), never live Python objects; timeouts, cancellation, and resource accounting; faithful export that emits the exact frozen source with its environment manifest and **visibly identifies** it as a custom-code node rather than silently falling back to a template.
- Gate: an independent adversarial review signs off in writing. Absent that sign-off, this release does not ship — regardless of customer pressure.

### E9 — Compliance Package

**Version:** `1.0.0`. **Effort:** ~12 weeks plus external vendor time and real money. **Track A floor:** `skyulf-enterprise` ≥0.4.0.
**Draws from:** Phase 3 (accessibility — Track A R9.5 produced the evidence base for a VPAT), Phase 10 / `data-governance-audit.md`, Phase 16a.

- Deliverables: a SOC 2 Type I readiness assessment and evidence collection; a third-party penetration test; a VPAT authored from Track A R9.5's axe results and keyboard-navigation coverage; a security whitepaper; a data-processing addendum; a public trust page.
- **Cost reality:** a SOC 2 Type I plus a credible pen-test is typically a five-figure external spend, before the maintainer's own time. **Do not start E9 until annual recurring revenue exceeds that spend by a comfortable margin.** Compliance is a cost centre that unlocks deals you can already name — never a speculative investment.
- Gate: an external auditor's readiness report, not a self-assessment.

## Track B Version Ledger

| # | Deliverable | `skyulf-enterprise` | `@skyulf/enterprise-ui` | Track A floor |
|---|---|---|---|---|
| E0 | Licensing & commercial decision | — | — | backend ≥1.0.0 |
| E1 | Organizations, Workspaces, Membership | **0.1.0** | **0.1.0** | backend ≥1.0.0, frontend ≥1.0.0 |
| E2 | RBAC, members, service accounts | **0.2.0** | **0.2.0** | backend ≥1.0.0 |
| E3 | SSO (OIDC/SAML) | **0.3.0** | **0.3.0** | backend ≥1.0.0 |
| E4 | Audit depth, retention/DSAR, encryption governance | **0.4.0** | **0.4.0** | backend ≥1.0.0 |
| E5 | Licensing, entitlement, usage metering | **0.5.0** | **0.5.0** | backend ≥1.0.0 |
| E6 | Cost / FinOps visibility | **0.6.0** | **0.6.0** | backend ≥1.3.0 |
| E7 | Governance & promotion workflows | **0.7.0** | **0.7.0** | backend ≥2.3.0 |
| E8 | Isolated code executor (Phase C) | **0.8.0** | **0.8.0** | backend ≥2.0.0 |
| E9 | Compliance package | **1.0.0** | **1.0.0** | — |

No number in this table appears in the Track A Version Ledger. The two tracks cannot collide.

---

# Pricing

## Method and honesty caveat

These numbers are calibrated against publicly-listed pricing for comparable open-core / source-available developer and data tooling (dbt Cloud, Metabase, GitLab, Weights & Biases, n8n, Dagster+, Prefect) as a **band**, not as products to imitate. **Live pricing pages could not be re-fetched during this planning session** (the fetch tool returned HTTP 403 for vendor pricing pages and the search API was rate-limited/unauthenticated), so the comparison band below is stated from general market knowledge and **must be re-verified against the vendors' live pricing pages before any number is published**. The band is used only to establish an order of magnitude; the recommended Skyulf numbers are derived primarily from Skyulf's own situation, not from the comparables.

**Calibration band (verify before publishing):** per-seat developer tooling in this category generally lists in the **$20–$100 per user per month** range (GitLab Premium and W&B Pro sit toward the lower-middle; dbt Cloud Team sits at the top). Flat per-instance / per-deployment tiers for self-hosted analytics tooling generally list in the **$100–$500 per month** range (Dagster+ and Prefect starter tiers toward the bottom, Metabase Pro toward the top). Enterprise tiers across essentially all of them are **"contact us"** with typical entry contracts in the low five figures per year.

## Recommended tiers

| Tier | Price | What you get | Status |
|---|---|---|---|
| **Community** | **$0, forever** | Everything in Track A: the full canvas, all nodes, all engines, DL, Ray, notebook export/import, leakage guardrails, authentication, Postgres, self-hosting. AGPLv3 backend/frontend; Apache-2.0 `skyulf-core`. No feature is removed to create upsell pressure. | **Ship now.** This tier already exists — it just needs to be named on a pricing page. |
| **Team** (self-hosted) | **$299 / month per production instance**, or **$2,990 / year** (2 months free). Includes up to **25 named users**; additional 25-user blocks **+$199/month** each. | Community, plus Track B E1–E4: organisations & workspaces, RBAC, SSO (OIDC/SAML), service accounts, deep audit log, retention/DSAR workflows. Email support with a 2-business-day response target. | **Realistic starting point** — but do not publish it until Track B E1–E3 actually exist. Publishing a paid tier whose features are unbuilt is the fastest way to burn the credibility this roadmap is trying to build. |
| **Enterprise** | **Custom — "contact us."** Indicative floor: **$15,000 / year.** | Team, plus E6–E9: cost/FinOps, promotion governance, the isolated code executor, compliance artefacts, a named support contact, an SLA, and roadmap input. | **Aspirational.** See the justification below — this number is a placeholder until real logos exist. |
| **Commercial License Exception** | **$5,000 / year, flat** | Not a feature tier: an AGPLv3 exception permitting proprietary/closed-source use of the backend and frontend. `skyulf-core` never needs this (Apache-2.0). | **Realistic starting point, available today.** Requires zero new product code — the `COMMERCIAL-LICENSE.md` intake process already exists. This is the most likely first dollar Skyulf earns. |
| **Services** | **$2,000–$8,000 fixed-scope** per engagement | Integration help, a custom connector or node, a migration, a training workshop. Time-boxed and scoped in writing. | **Realistic starting point, available today.** Zero product code required. |
| **Sponsorship** | GitHub Sponsors: **$25 / $100 / $500 per month** | Logo placement in the README, prioritised issue triage at the $500 tier. | **Ship now.** Costs one configuration page and one README section. |
| **Cloud / hosted** | **Not offered** | — | **Deliberately declined for now.** A hosted tier means uptime obligations, on-call, per-tenant isolation, and a bill — with zero budget and one maintainer, offering it would degrade the OSS product. Revisit only if Track B E1 has shipped and paying customers explicitly ask. |

### Justification, per number

**Community at $0.** This is not charity, it is the entire distribution strategy. With ~47 stars, the binding constraint is awareness, not monetisation. `user-complaints-research.md` independently found that **pricing opacity and expensive-tier lock-in is the #2 most-cited complaint** across comparable AutoML/no-code tools (4+ sources), and vendor lock-in is #1. A generous, genuinely complete free self-hosted tier plus published prices directly attacks both of the two loudest complaints in the category — that is a positioning asset, not lost revenue. Nothing in Track A may ever be moved behind a paywall; doing so once would permanently poison the well.

**Team at $299/month flat per instance (not per seat).** Three reasons for flat-per-instance over per-seat, in order of importance. (1) **Enforcement cost:** `round6-gap-audit.md` confirmed there is literally no seat, plan, or usage-event model in the codebase. Per-seat pricing requires building and maintaining seat counting, reconciliation, and overage handling — E5, a five-week build — before you can bill anyone. A flat per-instance license key is a signed string with an expiry date, which is roughly a day of work. (2) **Sales friction:** $299/month clears most engineering managers' discretionary spend, so the first deals can close without procurement, legal review, or a security questionnaire — the only kind of deal a 47-star project can realistically win in 2026. (3) **Honesty:** per-seat pricing implies per-seat support, and one maintainer cannot credibly promise that. Against the calibration band, $299/month sits mid-range for self-hosted tooling — deliberately not at the bottom, because a price that is too low signals a hobby project, and not at the top, because there are no reference customers yet to justify it. The 25-user cap with +$199 blocks gives a growth path without building metering.

**Enterprise at "custom, indicative floor $15,000/year" — explicitly aspirational.** The floor is derived from the maintainer's own cost, not from perceived value: a real enterprise deal drags in a security questionnaire, an MSA negotiation, procurement onboarding, an SLA commitment, and ongoing named support — realistically 40–60 hours in year one before any engineering. Below roughly $15k/year that is net-negative work for a solo maintainer. **Be explicit with yourself: without SOC 2, a pen-test, reference customers, or a VPAT, this number is very unlikely to be winnable in 2026.** The realistic first "enterprise" deal is a $5,000 commercial-license exception plus a services engagement — which is why both are listed as separate, available-today line items. Publish Enterprise as "contact us" with no number at all; keep $15,000 as an internal floor below which you decline, and revisit it once E9's compliance artefacts exist and two reference customers will speak on record.

**Commercial License Exception at $5,000/year flat.** This is the most credible near-term revenue line because it requires no product work whatsoever — the AGPLv3 backend/frontend split and the `COMMERCIAL-LICENSE.md` intake already exist, and `skyulf-core`'s Apache-2.0 license means the exception is only needed by companies embedding the *platform*, who are by definition already deriving real value from it. $5,000 is low enough to be an easy yes for a funded startup and high enough not to be a rounding error.

**Services at $2,000–$8,000 fixed scope.** Fixed-scope, written-deliverable engagements, not hourly retainers. They generate revenue, force real-user contact (which is exactly the evidence the Track B gate needs), and require zero speculative code. The risk to manage is time: cap services at a fraction of available hours or they will consume the roadmap.

**Sponsorship at $25/$100/$500 per month.** Effectively free to set up and provides a legitimate, non-transactional way for individual users to support the project. Do not expect meaningful revenue; do expect useful signal about who is actually using it.

### What to do about pricing *this month*

1. Publish the **Community** tier and the **Sponsorship** tiers today. Both already exist.
2. Publish **Commercial License Exception** and **Services** with their real prices today. Both are deliverable today.
3. Publish **Team** and **Enterprise** as a "Planned" section with a "register interest" link and **no committed date**. Registered interest is the cleanest possible instrument for measuring Track B's demand trigger.
4. Never quote a number you cannot deliver against within 30 days.

---

# Positioning

## The problem with the current framing

`README.md` currently leads with *"The Visual MLOps Builder"* and *"a self-hosted, privacy-first MLOps platform... the glue that holds your data science workflow together."* Every one of those phrases is claimed by dozens of larger, better-funded projects. It says what the software *is*, not why anyone should switch — and it buries the two things that are genuinely unusual.

Phase 17b's research established, with cited external evidence, that:
- **No dedicated Python library markets leakage-safety as its headline feature** — and `skyulf-core`'s calculator/applier split structurally *is* that.
- **scikit-learn's own official docs concede** that pickle/joblib model persistence is fragile, insecure, and non-portable across versions — and Skyulf's artifacts are already JSON.

Both facts are true today, cost nothing to state, and are unclaimed. That is the strongest asset available, and it is currently invisible.

## Recommended `README.md` rewrite (action this in R1, Task R1.4)

Replace the tagline and lead paragraph with:

```markdown
> **Skyulf:** Leakage-safe ML pipelines you can actually read.

Skyulf is a self-hosted visual ML platform built on one idea: **a pipeline
step that learns from data (a scaler, an encoder, an imputer) must fit on
your training set and only *apply* to everything else.** Skyulf enforces
that split structurally — every step is a `Calculator` that fits and an
`Applier` that applies, and the thing in between is a **plain JSON
artifact you can open, diff, and version**, not an opaque pickle.

Three consequences you can check for yourself in five minutes:

- **Leakage-safe by construction.** The fit/apply split isn't a convention
  you have to remember — it's the only way the pipeline can be built. The
  canvas warns you *before* you run when a learned transform sits on the
  wrong side of your split.
- **Your artifacts are JSON, not pickles.** scikit-learn's own docs call
  pickle persistence fragile, insecure, and non-portable across versions.
  Open a Skyulf artifact in a text editor. Diff two of them. Commit them.
- **No lock-in, by design.** Export any pipeline to a runnable notebook at
  any time. `skyulf-core` is Apache-2.0 and works standalone with no
  canvas, no server, and no account — `pip install skyulf-core`.
```

Then, in order, immediately below: the live-demo badge (currently buried under 16 badges — move it above the badge wall), a 30-second animated GIF of the sample-dataset → run flow shipped in R2, and the existing Quick Start.

## Recommended `skyulf-core/README.md` additions (also R1.4)

Add two sections near the top, with runnable code:

````markdown
## Leakage-safe by construction

Every transformation splits into a `Calculator` (fits on training data,
returns a JSON artifact) and an `Applier` (applies that artifact to any
data). You cannot accidentally fit on your test set, because fitting and
applying are different objects.

```python
from skyulf.preprocessing.scaling.standard import StandardScalerCalculator, StandardScalerApplier

artifact = StandardScalerCalculator().fit(X_train, {"columns": ["age", "income"]})
X_train_scaled = StandardScalerApplier().apply(X_train, artifact)
X_test_scaled  = StandardScalerApplier().apply(X_test,  artifact)   # same artifact, no refit
```

## Your artifacts are JSON

`artifact` above is an ordinary dict. Print it, diff it, commit it to git,
review it in a pull request. There is no pickle in the loop.
````

## Cheap, free, actionable now (do these alongside R1 and R2)

Each of these costs hours, not weeks, and each is a credibility action a pre-PMF project can take without spending money:

1. **Move the live-demo link above the badge wall** in `README.md`. Sixteen badges before the demo link is a conversion problem.
2. **Record one 30-second GIF** of the R2 sample-dataset → template → run flow and put it directly under the tagline. A visual tool with no visual above the fold is losing visitors in the first five seconds.
3. **Write one short post per release** on the single most interesting thing in it. R1's post is "Nine bugs I found in my own ML platform and how" — public bug-hunting builds more trust in a small project than a feature announcement.
4. **Pin a GitHub Discussion**: "What are you building with Skyulf?" This is the intake channel for the Track B demand trigger, and it costs one click.
5. **Submit `skyulf-core` to relevant `awesome-*` lists** (awesome-python, awesome-machine-learning, awesome-mlops) once R3 ships and the standalone story is genuinely good. Do it *after* R3, not before — a submission that lands on a rough first impression is spent.
6. **Publish the pricing page** with the Community/Sponsorship/License-exception/Services rows. Transparent pricing is itself a differentiator in a category whose #2 complaint is pricing opacity.
7. **Add a "Compare" section** naming what Skyulf is *not*: not an AutoML black box, not a hosted service, not a scheduler. Honest scoping earns more trust than claiming everything.

---

# Self-Review

## 1. Spec coverage

Every explicit requirement from the request, mapped to where it is satisfied:

| Requirement | Where satisfied |
|---|---|
| Two clearly separated tracks (Core/OSS, Enterprise-optional) | `# TRACK A` (R1–R19) and `# TRACK B` (E0–E9), with Track B opened by an explicit "this entire track is optional" callout |
| Do not lead with "become an enterprise product" | Track A is presented first and is 19 of 29 releases; Track B's "Why this track exists at all" section states the 47-star/zero-revenue/zero-compliance reality plainly and defers the whole track |
| Exact version numbers per unit of work, per component | Version Ledger (Track A), Track B Version Ledger, and a **Version impact** line on every single task |
| Semver logic (patch/minor/major) stated and applied | Global Constraints; applied and justified at every release, with MAJOR reserved for R10 (auth required, aliases removed, SQLite dropped), R11 (core base-class change), R14 (Celery removed) |
| Start at backend 0.7.10 / frontend 0.7.10 / core 0.5.9 | R1 |
| Components not required to stay in lockstep | Demonstrated concretely: R3 (frontend untouched), R5 (frontend-only), R8/R9/R10 (core untouched), R13 (backend-only), R16 (backend patch while core/frontend take minors) |
| Only bump what changed | Global Constraints rule + `—` rows in the ledger + explicit "skip Step F"/"skip `package.json`" instructions in the affected release-cut tasks |
| Realistic, priced Enterprise tiers with justification | `# Pricing` — seven line items, each with a per-number justification paragraph |
| Explicitly flag realistic vs. aspirational | Every row carries a **Status** cell; the Enterprise justification states plainly that $15k/yr is unlikely to be winnable in 2026 |
| Ground pricing in the differentiation doc's thinking | Pricing cites `user-complaints-research.md`'s finding that pricing opacity is the #2 complaint and vendor lock-in the #1, and `round6-gap-audit.md`'s finding that zero entitlement code exists (which drives the flat-per-instance recommendation) |
| Positioning rewrite, actionable for free | `# Positioning` — literal replacement copy for both READMEs plus seven zero-cost actions |
| Cheap high-visibility wins first | R1 (bugs + docs), R2 (Phase 8 quick wins), R3 (core DX) precede all XL work |
| Weave in deep-learning, ray-migration, training-visualization, code-escape-hatch by reference, deciding placement | R12–R14 (Ray, referencing the six existing plans), R15–R17 (DL, referencing that roadmap's phases and gates), R8.4 + R17.3 (training visualization tiers a and b), R8.2 + R19.3 + E8 (code escape hatch A, B, C) — each with an explicit "why here" paragraph |
| Honour the master fix list's "What NOT to do" | Referenced at the point of use: sklearn adapter (R3.2 Step 3), shared validation helper (R3.1), Narwhals-before-Pandera (R7 sequencing rule), partitionable contract before more node types (R11 "why here"), pipeline schema versioning before DL (R9.1 → R15), no Phase C on shared workers (R19 exclusion + E8 preconditions), no lazy DuckDB/full Narwhals replacement as a quick win (R7 scope note), orphaned-dataset behaviour untouched (R9.2 Step 4), Phase 4 before Phase 5 redesigns (R5 before R8/R9), no live curves for sklearn `.fit()` (R17 "why here"), no reimplementing published algorithms with maintained wrappers (R6.2 Step 3), schema capture before statistical drift (R18 "why here"), "complexity fatigue" excluded as unverified (Deferred table) |
| Enterprise track gated on a concrete trigger, not a date | "Track B — Gate": one hard technical prerequisite plus two-of-three demand triggers, with a quarterly re-evaluation and an explicit "if none fire in a year, don't build it" |
| Self-Review with spec/placeholder/type-consistency checks | This section |
| No execution-choice prompt at the end | Omitted deliberately, per instruction |

**Gaps found and closed during review:** Phase 5's remaining page redesigns (Dashboard/EDA/Error Log/Slow Nodes) were only partially placed — Canvas/Experiments/Jobs are in R8.5 and Dataset/Drift/Model Registry in R9.6; the four "follow-up, not yet designed" pages named in the master fix list are **not** individually scheduled, because that doc itself says they need a design pass first. They are the correct first candidate for a fast follow-up inside R9. This is recorded here rather than invented as a fake task.

## 2. Placeholder scan

Searched for: `TBD`, `TODO`, `implement later`, `fill in`, `appropriate error handling`, `add validation`, `handle edge cases`, `similar to Task`, `etc.` used as a substitute for content.

- **No `TBD`/`TODO`/`implement later` appears in any task.**
- **No "add appropriate X"** phrasing: every error path names the exact exception type (`SkyulfConfigError`, `SkyulfSecurityError`, `ArtifactSignatureError`, `QuotaExceeded`, `ExpressionRejected`, `SkyulfLeakageWarning`).
- **No "similar to Task N."** The one intentionally shared procedure (Release Cut) is written out in full once with real commands, and every release supplies its own literal parameter values — no reader has to look elsewhere for a value.
- Verified by `grep -in "TBD\|TODO\|appropriate error handling\|add appropriate\|handle edge cases\|similar to task\|implement later\|fill in details"` over the finished document: the **only** matches are the four lines of this placeholder-scan section itself, which quote the patterns being searched for. No task body contains any of them.
- Releases that delegate to an existing plan (R12–R17) do **not** restate that plan's contents — but each names the exact document, its phase/plan numbers, its gates, and its version impact, which is the deliberate instruction from the request ("reference them, don't re-derive them").

## 3. Type and version consistency

**Version conflicts:** each component's value was traced release by release through the Version Ledger. Every value is either a legal successor of the previous release's value under the stated semver rule, or an explicit `—` carry-forward. **No release assigns two different values to the same component**, and no version number is reused. Cross-checked specifically:
- backend: 0.7.9 → 0.7.10 → 0.8.0 → 0.8.1 → 0.9.0 → (0.9.0) → 0.10.0 → 0.11.0 → 0.12.0 → 0.13.0 → 1.0.0 → 1.1.0 → 1.2.0 → 1.3.0 → 2.0.0 → 2.1.0 → 2.1.1 → 2.2.0 → 2.3.0 → 2.4.0 — strictly increasing, three MAJORs each justified by a named breaking change.
- frontend: 0.7.9 → 0.7.10 → 0.8.0 → (0.8.0) → 0.9.0 → 0.10.0 → 0.11.0 → 0.12.0 → 0.13.0 → 0.14.0 → 1.0.0 → (1.0.0) → 1.1.0 → (1.1.0) → 1.2.0 → 1.3.0 → 1.4.0 → 1.5.0 → 1.6.0 → 1.7.0 — strictly increasing, one MAJOR (login required).
- skyulf-core: 0.5.8 → 0.5.9 → 0.6.0 → 0.7.0 → 0.8.0 → (0.8.0) → 0.9.0 → 0.10.0 → (0.10.0 ×3) → 1.0.0 → (1.0.0 ×3) → 1.1.0 → 1.2.0 → 1.3.0 → 1.4.0 → 1.5.0 — strictly increasing, one MAJOR (calculator base-class change).
- Track B's line (`skyulf-enterprise` 0.1.0 → 1.0.0) shares no value-space with any Track A line, by construction.

**Interface name consistency** across tasks that reference each other:
- `require_config` / `SkyulfConfigError` — defined R3.1, consumed R6.1, R6.2, R6.3, R11.2. Same names throughout.
- `skyulf.artifacts.stamp` / `migrate` / `CURRENT_SCHEMA_VERSIONS` — defined R4.1, consumed R4.2, R4.3, R6.2, R7.2.
- `ArtifactDiff` / `diff()` — defined R4.2, consumed R8.4.
- `GuardrailFinding` — defined R4.4, consumed R8.4's `TimelineEvent.source == "guardrail"`.
- `NodeFieldSchema` — defined R5.3, consumed R6.5 (`GET /api/nodes/metadata` returns the identical shape) and R15.1 (DL settings panel).
- `DataTable<TRow>` — defined R5.1, consumed R8.3, R8.5, R9.6, R12.2.
- `AppShell.slotOrgSwitcher` — defined R5.4, consumed by Track B E1. This is the single Track A↔Track B coupling point and it is named identically in both places.
- `SchemaSpec` / `SchemaDriftReport` — defined R7.2, consumed R7.4 and R18.2.
- `current_user` — defined R10.1, and E1's workspace scoping is explicitly described as attaching to it (`workspace_id` **alongside** `owner_user_id`, not replacing it).
- `resource_spec_for_job(job) -> ResourceSpec` — defined R13.1, extended R17.2, consumed E6.
- `partial_fit` / `finalize` / `partitionable` — defined R11.1, consumed R13.1 Step 4.
- `progress_callback(current_epoch, total_epochs, score=...)` — defined R15.1 (copied verbatim from the DL roadmap), consumed R17.3.
- `TrainingMetricSnapshot` — defined R17.3, validated by R9.4's Zod envelope layer.

**One inconsistency found and corrected during review:** the master fix list cites the leakage module as `skyulf-core/skyulf/preprocessing/_shared/leakage.py`, but the file actually lives at `skyulf-core/skyulf/leakage.py` (verified in-repo). This plan uses the real path in R4.3 and R4.4. Similarly, `profiling/recommendations.py` is cited in several research docs but the real path is `skyulf-core/skyulf/profiling/_analyzer/recommendations.py`; R2.2 uses the real path.

**Effort totals (sanity check, not a commitment):** Track A R1–R10 is roughly 6–8 months of solo full-time-equivalent work; R11–R19 roughly another 12–15 months. That is a multi-year roadmap for one person, which is the honest reading — the ordering matters far more than the total, because the value is front-loaded: R1–R4 alone (~2 months) delivers correctness, a working first-run, a genuinely good standalone library, and the leakage-safe positioning made real.
