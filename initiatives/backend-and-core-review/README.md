# Backend & skyulf-core Review — Issues, Fixes, and Missing Capabilities

**Date:** 2026-08-26 · **Status:** Investigation complete, prioritized fix list ready, **no fixes applied**

## What this is

A fresh code review of `backend/` and `skyulf-core/` by four parallel
full-scope investigators: cross-cutting layer (auth/config/DB/monitoring),
pipeline execution (jobs/Celery/realtime), data & model lifecycle
(ingestion/registry/deployment/artifacts), and the skyulf-core SDK
(engines/transforms/serialization). Every finding carries file:line
evidence and a concrete fix.

**Methodology caveat:** static analysis by code review, not a live
runtime trace or test-suite run. Reproduce before fixing anything marked
CRITICAL.

**Companion:** [2026-08-26-skyulf-core-second-pass.md](2026-08-26-skyulf-core-second-pass.md)
— a deeper core-only pass finding ~41 NEW issues (time-series y
misalignment, `group_agg` leakage, metric-direction inversion, 9 more
engine divergences, worker-concurrency hazards, README quickstart crash).

## Headline numbers

| Severity | Count | Examples |
|---|---|---|
| CRITICAL | 3 | No authentication anywhere; cancelled jobs resurrect on redelivery; batch cancel kills sibling jobs |
| HIGH | 12 | Plaintext S3 credentials; preview errors swallowed ("Check console" root cause); 5 engine-parity bugs; no live job progress |
| MED | ~30 | Races, unbounded fetches, silent no-op transforms, drift PSI=0.0 on error |
| LOW | ~8 | Heavy query rows, versioning gaps, misc |
| Missing capabilities | 10 | Per-epoch events, runtime estimates, drift histograms (mostly exists!), profile endpoint |

---

## S. Security & access (P0 — start here)

**S1. No authentication or tenancy exists at all (CRITICAL, confirmed by two reviewers).**
`backend/dependencies.py` has only `get_db`/`get_config`; ingestion guard
stubs return `None` (`data_ingestion/dependencies.py:18-25`); routers
hardcode `user_id = 1` (`data_ingestion/router.py:150,172`). Every route
is anonymous, including destructive ones: `DELETE /api/monitoring/errors`
(`monitoring/router.py:1032`), deployments, pipeline execution, the
`/ws/jobs` websocket. All auth config is **dead code**: `SECRET_KEY`, JWT
fields, `MAX_LOGIN_ATTEMPTS` (`config/mixins/security.py:13-49`) have zero
consumers. Default `HOST=0.0.0.0` (`environments.py:20`) exposes this to
the LAN.
**Fix (minimal first step):** one `get_current_user` dependency validating
a static API token via the existing `SECRET_KEY`, applied as router-level
`dependencies=[Depends(...)]`; `is_admin` for DELETE/disposition routes;
derive `actor` from auth instead of client-supplied strings
(`monitoring/router.py:683`). Full multi-user tenancy remains a later
phase (enterprise-readiness Phase 0), but token auth is days, not weeks.

**S2. CORS wildcard + credentials in default mode (HIGH).**
`FASTAPI_ENV` defaults to development → `CORS_ORIGINS=["*"]`
(`environments.py:21`) with `allow_credentials=True, allow_headers=["*"]`
(`main.py:360-366`); `SECURITY_HEADERS` is defined
(`environments.py:40-51`) but never applied.
**Fix:** reject `"*"` when credentials enabled; apply security headers via
a small middleware.

**S3. Rate limiter is not mounted (HIGH).**
`default_limits=["200/minute"]` (`middleware/rate_limiter.py:16-19`) only
works if slowapi middleware is mounted — it never is. Unauthenticated
`POST /monitoring/pipeline-logs` accepts an unbounded batch
(`monitoring/router.py:1553-1576`) → trivial DB-bloat spam.
**Fix:** mount `SlowAPIASGIMiddleware`; cap batch size.

**S4. Credentials stored in plaintext (HIGH).**
S3 keys persist raw in `data_sources.config` JSON
(`data_ingestion/service.py:354-370`); the `credentials` column comment
says "encrypted in production" — false (`models.py:108`). Responses
redact, but `DataSource.to_dict()` returns full config (`models.py:144`).
**Fix:** Fernet encryption at rest (key from settings); never echo back.

**S5. Pickle deserialization of artifacts (MED).**
`joblib.load` on artifacts in `predict()` (`deployment/service.py:237`,
`artifacts/local.py:44`, `s3.py:144`) executes arbitrary code if an
attacker can write to artifact storage. Also skyulf-core's
`SkyulfPipeline.save/load` is raw unversioned pickle (see E8).
**Fix:** restrict load paths (partially done), integrity hashes now;
safetensors/ONNX long-term.

---

## J. Job execution correctness (P0/P1)

**J1. Cancelled jobs resurrect on redelivery (CRITICAL).**
`execute_pipeline` unconditionally flips the row to `running`
(`ml_pipeline/_execution/pipeline_execution_service.py:185-189`). Cancel
uses `revoke(terminate=True, SIGTERM)` (`job_manager_base.py:66`); with
`task_acks_late=True` + `task_reject_on_worker_lost=True`
(`celery_app.py:14-15`) the killed message is re-queued and another
worker re-runs the **cancelled** job.
**Fix:** before starting, refresh status and bail if terminal; poison or
ack redelivered messages for terminal jobs.

**J2. Cancelling one branch kills siblings, orphaning them as "running" (CRITICAL).**
All branches share one Celery task id (`run_pipeline.py:279-284`) in one
ThreadPoolExecutor (`tasks.py:125-140`); SIGTERM kills the process,
siblings never write a final status (stuck until the 2h orphan reset,
`main.py:176-222`).
**Fix:** one task per branch (or cooperative cancellation); mark
interrupted siblings failed at worker startup.

**J3. Cancellation never stops actual work (HIGH).**
No cancellation poll between nodes (`engine/__init__.py:135-172`); only a
post-run guard. Non-Celery mode has no task id, so revoke is a no-op and
training runs to completion.
**Fix:** pass a `should_cancel()` callback checked per node;
`threading.Event` in thread mode.

**J4. Root cause of the frontend's "Check console for details" (HIGH).**
`POST /preview` runs the engine synchronously inside the async handler
(blocks the event loop) and then replaces every exception with
`SkyulfException(message="Pipeline preview failed") from None`
(`_routers/preview.py:748-750`) — the real node error is discarded.
**Fix:** `asyncio.to_thread` the engine run; include `str(exc)` in the
raised message. This directly closes frontend README.md §B.6.

**J5. No live progress: `progress` is always 0, `current_step` never written (HIGH).**
Only writers are status flips (`pipeline_execution_service.py:187,133`);
the throttled progress event always emits `progress=0` (`:107-115`).
**Fix:** in the node loop set `progress = i/N*100`, `current_step =
node_id`, emit per node. Unblocks frontend node-body progress strips
(charts report C2).

**J6. Trials chart broken in Celery mode (HIGH).**
`record_trial` fills an in-process dict inside the **worker**
(`trial_buffer.py`); `GET /jobs/{id}/trials` reads the **API process**
buffer (`_routers/jobs.py:115-133`) → always empty with `USE_CELERY=True`.
**Fix:** Redis- or DB-backed trial buffer.

**J7. New Redis connection per event (HIGH).**
`_redis_client_sync()` builds `redis.Redis.from_url` on every
`publish_job_event` (`events.py:53-62`) — fired every boosting iteration
and every 2s log flush.
**Fix:** module-level pooled client.

**J8. Submit-dedupe races + timestamp mix (MED).**
Submit-lock release pops the dict while held (`run_pipeline.py:194`);
`with_for_update(skip_locked=True)` skips locked candidates
(`jobs.py:112`); asyncio locks don't span uvicorn workers. Separately,
aware `datetime.now(UTC)` is stored into naive `DateTime` columns
(`models.py:269-270`) while `_reset_stale_jobs` compares a naive cutoff
(`main.py:201-215`) — orphan reset can misfire.
**Fix:** partial unique index on `(dataset, node, branch_index) WHERE
status IN (queued, running)`; UTC-naive timestamps everywhere + worker
heartbeat check.

**J9. Engine trusts request node order + versioning races (MED).**
`_run_node_loop` iterates `config.nodes` verbatim
(`engine/__init__.py:145`); non-topological saved pipelines → late
"Artifact not found". `version_int=max+1` races on concurrent saves
(`pipeline_versions_service.py:54-65`).
**Fix:** Kahn-sort before running; DB unique constraint on versions.

**J10. Celery hygiene (MED).**
No `task_time_limit`/`soft_time_limit`, no `worker_max_tasks_per_child`,
concurrency unset (`celery_app.py`) → hung fits block workers forever.
**Fix:** add limits + recycling.

**J11. Metadata perf (MED/LOW).**
`_build_node_metadata` re-loads every node's output AND first input
artifact from disk for a one-line summary (`engine/__init__.py:305-316`);
`get_node_summaries` loads 2×200 full jobs incl. JSON blobs per canvas
load (`jobs.py:334-335`).
**Fix:** build summaries from in-memory objects; column projection.

---

## D. Data & model lifecycle (P1)

**D1. Ingestion cancel is cosmetic (HIGH).**
`cancel_ingestion` flips a JSON status (`service.py:142-150`); the task
never checks it, and `_mark_ingestion_completed` overwrites "cancelled"
with "completed" (`tasks.py:104-121`). Unlocked read-modify-write on
`source_metadata` loses updates.
**Fix:** check status before each commit inside the task; optimistic
versioning on metadata.

**D2. Unbounded full-data fetch during ingestion (HIGH).**
`connector.fetch_data()` with no limit (`tasks.py:89`) — a multi-GB S3
file OOMs the worker.
**Fix:** stream/chunk profiling or cap rows.

**D3. Deployment activate race (HIGH).**
SELECT-active → UPDATE-all → INSERT with no locking and no uniqueness on
`is_active` (`deployment/service.py:121-145`) → two concurrent deploys
leave two active rows.
**Fix:** partial unique index `WHERE is_active` + retry.

**D4. Deployed URI guessed, not verified (MED).**
`_resolve_final_deployment_uri` appends `{job_id}.joblib` without an
existence check (`deployment/service.py:84-90`) — deploy succeeds,
predict 500s later.
**Fix:** verify artifact exists before committing the deployment.

**D5. SQLite concurrency misconfiguration (HIGH).**
`StaticPool` = one shared connection serializing the app
(`database/engine.py:62`); the "Enable WAL" comment never executes a
pragma — the pragma logic lives in an unused module;
`DB_SQLITE_BUSY_TIMEOUT_MS` is never applied. Worse, `calculate_drift`
holds a session while reading up to **10GB into RAM**
(`monitoring/router.py:248`, `MAX_UPLOAD_SIZE` default 10GB,
`config/files.py:18`) and runs CPU-bound drift math on the event loop.
**Fix:** connect-event `PRAGMA journal_mode=WAL, busy_timeout`; drop
StaticPool; stream uploads to disk; drift compute via `run_in_threadpool`
or Celery.

**D6. Retention task can never run in Celery (HIGH).**
`cleanup_error_events` uses the FastAPI-lifespan-only
`async_session_factory` (`monitoring/tasks.py:9,26`); in workers it's
`None` → permanent RuntimeError, `error_events` grows unbounded.
**Fix:** build a sync engine inside the task like
`ml_pipeline/tasks.py:42-60`.

**D7. Artifact key collisions + dual-write divergence (MED).**
`LocalArtifactStore._get_path` maps `/`→`_` (`artifacts/local.py:20`) so
`a/b` and `a_b` silently overwrite; same flattening in the factory and
S3 cache paths. data_sources dual-write is best-effort with reads from
primary only (`async_data_sources_crud.py:85-97`); duplicate detection
substring-matches error text.
**Fix:** hash-suffix keys; reconcile strategy or drop dual-write.

**D8. Migrations swallow all errors (MED).**
Bare `except: pass` per ALTER (`database/engine.py:247-257`), no applied-
versions table, Postgres-hostile DDL. (Also flagged in cross-cutting.)
**Fix:** Alembic, or at minimum error-class discrimination + version
ledger.

**D9. Misc (LOW/MED).**
EDA `/decomposition` loads the full dataset uncapped (`eda/router.py:395`)
— DoS vector; EDA cancel race (`eda/tasks.py:222-224`); raw `str(error)`
persisted and served via `/status` (`tasks.py:137`); CPU-bound EDA blocks
the event loop (`eda/tasks.py:172`); **no SQL/API connectors exist yet**
(`service.py:273` TODO) — add timeouts when they land.

---

## E. skyulf-core engine parity & correctness (P1)

These are the same bug *class* as the dual-engine-correctness audit
findings — new, live instances:

**E1. Feature-generation ratio sign flip between engines (HIGH).**
Pandas `_safe_divide` preserves denominator sign
(`preprocessing/feature_generation/_common.py:90-101`); polars clamps
near-zero denominators to *positive* epsilon (`_polars_ops.py:122`).
Same config → different features → silently different model scores per
engine. **Fix:** `.then(pl.sign(den_sum).fill_null(1) * epsilon)`.

**E2. LabelEncoder None/null divergence (HIGH).**
Pandas fit renders `None` as `"None"`, `pd.NA` as `"<NA>"`
(`encoding/label.py:281`); polars maps nulls to `"nan"` (`label.py:51`).
Cross-engine artifacts send those rows to `missing_code` (-1) at scoring.
**Fix:** normalize nulls to `"nan"` in pandas fit/apply.

**E3. KNN/Iterative imputer crash on all-null columns (HIGH).**
sklearn `transform` drops all-missing columns (`keep_empty_features`
never set); write-back fails with ValueError/IndexError
(`imputation/_common.py:105,89`). **Fix:** `keep_empty_features=True` or
explicit fit-time detection + warning.

**E4. SimpleImputer all-null/empty-frame divergence (HIGH).**
All-null columns silently stay null on both engines
(`simple.py:45,87`); empty input: polars records None stats silently,
pandas raises an opaque sklearn error (`simple.py:173`). **Fix:** fit-time
detection with a clear "cannot impute column X: no non-missing values"
error.

**E5. Casting divergence on bad values (HIGH).**
Pandas `coerce_on_error=True` leaves the whole column uncast, unlogged
(`casting.py:369-374`); polars nulls bad values (`casting.py:183`).
Downstream schema divergence per engine. **Fix:** per-value coercion on
pandas (`errors="coerce"`), log skipped values.

**E6. Scalers/dummies on all-null columns (MED).**
Scalers output all-NaN silently (guards test `s != 0`, true for NaN —
`scaling/minmax.py:37` etc.); dummy/OneHot all-null columns vanish from
the schema (`dummy.py:141`, `one_hot.py:134-137`) → train/apply schema
shift. **Fix:** reject/warn at fit; keep placeholder columns.

**E7. Silent no-op transforms & swallowed errors (MED).**
Drift PSI/KL return **0.0 on any exception** — failure reports as "no
drift" (`profiling/drift.py:459-460,495-496`); PowerTransformer returns
input untransformed on failure (`transformations/power.py:85-87`);
GeneralTransformer skips failed columns; elliptic outliers skip filtering;
`pipeline.fit()` swallows evaluation errors into
`metrics["modeling_error"]` (`pipeline.py:250-257`).
**Fix:** return `None`/raise for unknowns; surface failures in step
warnings; `logger.exception` + typed error objects.

**E8. `SkyulfPipeline.save/load` is raw unversioned pickle (HIGH).**
(`pipeline.py:569-579`) — cross-version loads fail opaquely or misbehave,
and it's arbitrary code execution for untrusted artifacts. **Fix:**
envelope `{schema_version, core_version, config, artifacts}` via the
existing `JoblibModelSerializer` seam (`core/serialization.py`).

**E9. DX sharp edges (MED).**
Single-class targets crash with sklearn's raw message from CV
(`cross_validation.py:427-432`) — pre-check `nunique(y) >= 2`.
`FeatureEngineer.transform()` on an unfitted pipeline silently returns
input unchanged (`preprocessing/pipeline.py:59-83`) — raise
`NotFittedError`. No engine pinning: dispatch is per-call
(`dispatcher.py:125`) with mid-pipeline pandas/polars round-trips —
contradicts the "no hidden pandas" README claim; pin at pipeline entry.

---

## M. Missing capabilities (what to build to make things easier)

Feasibility was verified in code for each:

| # | Capability | Status today | Change |
|---|---|---|---|
| M1 | **Per-epoch training events** (live canvas charts) | Only tuning trials + boosting rounds emit (`_node_runners.py:896-923`); plain training emits nothing | Add epoch callback in fit path reusing existing `iteration` JobEvent fields; generalize core's `iteration_callback` (`modeling/base.py:115-131`) |
| M2 | **Run-time estimates** | Per-node durations already persisted (`node_timings`, `strategies.py:35-51`), never aggregated | Service computing median per `step_type`; emit `estimated_duration` on node start |
| M3 | **Structured failure reasons** | Message exists (`"Error in node X: …"` → `JobInfo.error`) but `JobEvent` has no error field | Add `error`/`failed_node` to JobEvent; populate in result/exception writers; combine with J4 |
| M4 | **Drift histograms** | **Mostly exists!** core computes binned ref/current distributions (`skyulf/profiling/drift.py:20-36,280-312`); `/drift/calculate` + `/drift/alerts/{id}` already return them (`monitoring/router.py:568,674`) | Add top-N category counts (categoricals return `None` today, `drift.py:411`); expose in history-list summaries. *Downgrades charts report C4-B from "backend needed" to small work* |
| M5 | **Dataset profile endpoint** | Full `DataProfiler` output already stored in `source_metadata.profile` (`data_ingestion/tasks.py:115`) | Dedicated `GET /sources/{id}/profile` instead of clients digging in metadata |
| M6 | **Registry sparkline data** | Per-version metrics + `created_at` already returned (`model_registry/schemas.py:15`) | Fix `version` int/str inconsistency (`service.py:177,201`); dataset filter; DB-level pagination (`service.py:287-310` loads all jobs into memory) |
| M7 | **`pipeline.explain(X)`** | SHAP implemented (`_explainability/shap_explanation.py`) but unreachable from `SkyulfPipeline` | Wire through |
| M8 | **Model cards** | `export_model_card()` exists (`pipeline.py:539`) but omits final feature schema + training profile | Wire in `infer_output_schema` + `DatasetProfile` |
| M9 | **Ops endpoints** | `/api/config` exists; job queries exist in monitoring | Feature-flag surfacing; queue-depth/disk in `/health/detailed`; usage-stats endpoint for the dashboard |
| M10 | **SQL/API ingestion connectors** | TODO (`service.py:273`) | With timeouts + row caps from day one |

---

## T. Test gaps (skyulf-core)

- No `SkyulfPipeline.save/load` round-trip test (only the joblib seam).
- No single-class-target, empty-frame imputation, or all-null-column
  scaling tests (findings E3/E4/E6).
- `test_engine_parity.py` covers scaler/imputer/WOE **fit only** — no
  apply-time parity for ratio sign (E1), label-encoder nulls (E2),
  casting (E5), or unseen categories. This is exactly the coverage debt
  the growth plan's T5 contract test was meant to close.

---

## Execution proposal

**Wave 0 — stop the bleeding (CRITICAL + HIGH security), ~1 week:**
S1 token auth → S2 CORS → S3 rate limiter → S4 credential encryption →
D5 SQLite WAL/pool + drift upload streaming → D6 retention task fix.

**Wave 1 — job correctness, ~1–2 weeks:**
J1 + J2 (resurrection/siblings), J4 (real preview errors — pairs with the
frontend toast fix), J5 + J6 + J7 (live progress + trials + redis pool),
J3 cooperative cancellation, D3 deployment unique index, D1/D2 ingestion
cancel + fetch caps.

**Wave 2 — core parity & safety, ~1–2 weeks:**
E1–E6 fixes, each with a red-green apply-time parity test (also pays down
the T list); E8 versioned save/load envelope; E7 error surfacing; J9/J10
ordering + celery limits.

**Wave 3 — capabilities, weighted by frontend pairing:**
M4 + M5 + M6 (unblocks charts report C4-B/C1 and node-journey previews) →
M1 + M2 + M3 (unblocks C2, F2/F3 lint & estimates) → M7/M8/M9/M10.

## Relation to other initiatives

- **dual-engine-correctness/**: E1–E6 are new live instances of its bug
  class; this review reinforces shipping its parity-test contract (T5)
  rather than fixing bugs one at a time forever.
- **enterprise-readiness/** Phase 0 (auth/tenancy): S1 confirms zero
  auth today and proposes the minimal token step ahead of full tenancy.
- **frontend-consumer-design/**: J4 closes README.md §B.6 ("Check
  console"); J5 + M1–M3 unblock charts report C2 and node-journey §7;
  M4 downgrades charts C4-B effort; M6 enables charts C1.
- Effort figures are judgement estimates for sequencing only, per repo
  convention.
