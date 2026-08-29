# Plan — MLflow integration (core seams → working tracking, registry & artifacts)

**Status:** 🟨 planned (needs a go/no-go decision)
**Created:** 2026-08-29 · **Tracker:** [`skyulf-core-findings-tracker.md`](skyulf-core-findings-tracker.md)
**Touches:** skyulf-core, backend, frontend — full-stack, with deliberate design changes
**Cross-refs:** [`skyulf-core-joblib-migration-plan.md`](skyulf-core-joblib-migration-plan.md)
(artifact format), [`skyulf-core-onnx-support-plan.md`](skyulf-core-onnx-support-plan.md)
(alternative portable artifact), roadmap task **R6.4** (MLflow-skinny fit hook),
`initiatives/training-visualization/2026-08-11-feasibility-and-plan.md` (durable
metric history)

---

## 1. The "80% ready" claim — what it means and what it doesn't

The audit (F-27) and the module docstrings are explicit that three seams were
built *ahead of the Databricks/MLflow phases*, and that "the abstraction
points are well chosen — this is why the assessment concluded MLflow
integration would be cheap". Verified state on 2026-08-29:

| Seam | File | State |
|---|---|---|
| Compute backend | `skyulf-core/skyulf/core/compute.py` | `ComputeBackend` + `LocalComputeBackend`, ContextVar-scoped (`compute_backend()`); nothing wires a distributed backend |
| Model serialization | `skyulf-core/skyulf/core/serialization.py` | `ModelSerializer` + `JoblibModelSerializer`, ContextVar-scoped; docstring names MLflow as the intended next implementation |
| Model registry | `skyulf-core/skyulf/core/model_registry.py` | `ModelRegistry` + thread-safe `InMemoryModelRegistry` (name, auto-increment version); docstring names "MLflow / Unity Catalog" as the intended subclass |

Supporting groundwork also done: F-14 ContextVar scoping (concurrent jobs
can't reconfigure each other's serializer/backend), F-15 fingerprint
(a stable identity to tag runs with), F-07 `to_native()` (clean artifact
extraction), job metrics/thresholds already persisted by the strategies.

**Be honest about what "80%" covers:** the *abstraction surface* is ready.
There are **zero lines of MLflow code** anywhere — no `skyulf/integrations/`
package, no fit callback in `modeling/base.py`, no backend run wiring, no
frontend surfacing. The remaining 20% is the actual integration, and part of
it is design work (job↔run mapping, configuration, UI), not plumbing.

---

## 2. What MLflow buys Skyulf

1. **Durable, job-addressable history** — today run history lives in our own
   DB (`training_jobs.metrics`, streamed logs over WebSocket). MLflow Tracking
   gives an industry-standard, self-hostable store with a comparison UI.
2. **Model registry with versions** — our Experiments/Model Registry pages
   are job-keyed; MLflow registered models add versioning/aliasing semantics
   enterprises expect.
3. **Artifact lineage** — model blob + model card JSON + pipeline diagram
   travel together, per run, queryable by tools outside Skyulf.
4. **Ecosystem gravity** — `mlflow-skinny` has ~42.5M downloads/month
   (per the 2026-08-11 quickwin research); "works with your MLflow" is a
   sales answer, not a feature request.

---

## 3. Skyulf ↔ MLflow concept mapping (the design change)

| Skyulf | MLflow | Notes |
|---|---|---|
| training/tuning job | **run** (`run_name` = short job id) | one run per job, started in the Celery task |
| pipeline node configs | **params** (flattened, prefixed by node id) | MLflow truncates long values — digest/truncate deliberately |
| `handle_success` metrics | **metrics** | already a flat JSON dict |
| F-15 `fingerprint()` | **tag** `skyulf.fingerprint` | ties run → seal identity |
| tuned thresholds | **params/tags** (`skyulf.thresholds.*`) | provenance `source` preserved |
| model artifact | **artifact**: `{job_id}.joblib` + `model_card.json` + `pipeline_diagram.mmd` | model card from `export_model_card()` (has the mermaid diagram since 0.8.7) |
| Model Registry entry | **registered model version** (phase 4) | `MlflowModelRegistry` implements the core seam |
| tracking server URL | backend config mixin + optional per-job flag | default **off** |

Deliberate non-goals for phase 1: nested runs per node (flat is simpler and
matches the Experiments page granularity), mlflow Model Serving (needs full
`mlflow`, separate decision), autologging (we log explicitly — our metrics
are already computed).

---

## 4. Layered design

### 4.1 Core — `skyulf/integrations/` (roadmap R6.4, additive)

- New package `skyulf/integrations/` with `mlflow.py`:
  - `enable_mlflow_logging(tracking_uri: str | None = None, experiment_name: str | None = None)`
  - **fit-callback seam** on `BaseModelCalculator.fit()` (R6.4 contract):
    `register_fit_callback(fn: Callable[[str, dict, dict], None])` — fires once
    per fit with `(node_type, config, metrics)`; `fit()`'s return contract
    does **not** change; with zero callbacks registered, fit makes zero extra
    calls (test-asserted, per R6.4 step 4).
  - Import discipline: lazy `importlib.util.find_spec("mlflow")` probe
    (F-28 pattern); missing dependency raises a clear
    `SkyulfOptionalDependencyError` — core without the extra never imports
    mlflow.
- `MlflowModelRegistry(ModelRegistry)` — `register/get/versions` mapped to
  MLflow registered models (`mlflow.register_model`, version list, latest).
  Lives in the integrations package, installed via a setter like the other
  seams.
- `setup.py` extra: `tracking = ["mlflow-skinny>=2,<3"]`.
  **Capability note (important):** `mlflow-skinny` is tracking-only —
  `mlflow.sklearn.log_model` / model packaging needs **full `mlflow`**. The
  extra decision is therefore phase-dependent: phases 1-3 work with skinny;
  model packaging (phase 5) needs full.

### 4.2 Backend — one job, one run

- `execute_pipeline` (`_services/pipeline_execution_service.py`): when
  tracking is enabled, wrap execution in `mlflow.start_run(...) / end_run()`;
  log params after config resolution; the strategy `handle_success` logs
  metrics + artifacts + fingerprint tag. Failure isolation is mandatory:
  tracking errors **log and never fail the job** (same best-effort philosophy
  as the BLE001-triaged sites).
- Run id persists on the job row (`training_jobs.mlflow_run_id`) so the
  frontend can deep-link.
- Config: new `backend/config/mixins/mlflow.py` — `MLFLOW_TRACKING_URI`,
  `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_ENABLED` (default false). Celery workers
  pick these up from env; the ContextVar-scoped seams guarantee concurrent
  jobs in the same worker don't clobber each other.
- Dev story: `tests/docker/` compose service running an mlflow tracking
  server for integration tests (mirrors the existing redis pattern).

### 4.3 Frontend — minimal, honest surfacing

- **Job submission:** optional tracking flag only if the backend reports
  tracking configured (schema-preview-style capability ping); otherwise the
  UI stays untouched — no dead toggle.
- **Experiments page:** a run with `mlflow_run_id` shows a "View in MLflow"
  link (tracking URI + `#/experiments/.../runs/...`). No new tab, no
  duplication of what MLflow's own UI already renders better.
- **Model Registry / Deployments (phase 4):** registry-sync indicator and
  "registered in MLflow as version N" line when applicable.

This is the "design changes from what we have already built" part: we do
**not** rebuild metric history UIs the Experiments page already has — MLflow
is linked, not mirrored.

---

## 5. Phases

| # | Phase | Deliverable | Est. |
|---|---|---|---|
| 1 | Core integrations | `skyulf/integrations/mlflow.py` + fit-callback seam + `tracking` extra + fake-callback tests (R6.4, TDD) | 2 days |
| 2 | Backend run wiring | start/end run, params/metrics/artifacts/fingerprint tag, config mixin, `mlflow_run_id` column + migration, best-effort isolation | 2 days |
| 3 | Frontend links | capability ping, submit flag, "View in MLflow" deep link | 1-2 days |
| 4 | Registry sync | `MlflowModelRegistry` + deployment hooks + registry UI line | 2 days |
| 5 | Model packaging *(optional)* | full `mlflow` extra, `mlflow.sklearn.log_model`, serving targets | 2-3 days |

**Phases 1-3 ≈ 1 week**, each independently shippable and gate-covered.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Tracking server outage breaks training | Best-effort isolation: all mlflow calls wrapped, logged, continued (design rule, test-pinned) |
| skinny/full capability confusion | Plan §4.1 names it: phases 1-3 skinny, packaging needs full; extra split accordingly |
| MLflow param/value length limits | Deliberate flattening + truncation in one helper, unit-pinned |
| Celery worker env drift (URI, creds) | Config mixin + startup log line stating effective tracking target; integration test with compose service |
| Run-state leakage across concurrent jobs | One explicit run per task (no relying on global "active run" outside the task); ContextVar seams already isolate serializer/backend |
| mlflow major-version churn | Cap the extra (`>=2,<3`), single owner module (`integrations/mlflow.py`) |
| UI duplication temptation | §4.3 design rule: link, don't mirror |

## 7. Decision points (need user call)

1. **Tracking-only first** (phases 1-3, `mlflow-skinny`) — recommended — or
   straight to model packaging (phase 5, full `mlflow`)?
2. **Enablement granularity:** global backend config only, or also per-job
   submit flag (frontend toggle)?
3. **Dev tracking server:** add an mlflow service to `tests/docker/`
   (recommended, mirrors redis)?
4. **Registry sync (phase 4)** in scope now or after tracking proves out?
5. **Tag F-15 fingerprint + mermaid diagram on every run** (recommended —
   costs nothing, makes runs verifiable against the seal)?
