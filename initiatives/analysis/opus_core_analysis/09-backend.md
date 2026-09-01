# Skyulf Deep Audit (Opus) — Backend: pipeline execution, API & services

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

**Scope:** `backend/` (27,564 lines) — `ml_pipeline/_execution/`, artifacts, routers,
`model_registry/`, `deployment/`, `data_ingestion/`, `database/`, `config/`, `main.py`.

Two repros in this report were executed against the real `PipelineEngine`, not
inferred from reading.

---

## Findings

### OC-68
### 🟠 High *(escalated from Low)* — Model alias map is task-unaware; a direct API caller silently trains the wrong estimator family

**File:** `backend/ml_pipeline/_execution/engine/_node_runners.py:1157-1183`

> **Merged, not re-filed.** First raised as ⚪ Low in
> [report 08](./08-modeling-tuning.md#oc-68) on the assumption the alias map
> would merely *crash* if reached. The backend audit shows a worse case that does
> not crash, so the finding is escalated and its detail lives here.

```python
alias_map = {
    "logisticregression": "logistic_regression",
    "randomforestclassifier": "random_forest_classifier",
    "random_forest": "random_forest_classifier",   # <- no regressor counterpart
    "ridgeregression": "ridge_regression",
    "ridge": "ridge_regression",
    "randomforestregressor": "random_forest_regressor",
}
```

`"random_forest"` always resolves to the **classifier**, regardless of whether the
node is a `RegressionNode`. Other short forms used elsewhere in this very codebase
(`gradient_boosting`, `decision_tree`, `svm`, `knn`, `xgboost`, `lightgbm`,
`naive_bayes` — the `BASE_ESTIMATORS_CLF/REG` convention in `ensemble.py`) aren't
aliased at all. `NodeConfigModel.params: dict[str, Any]`
(`_internal/_schemas.py:37-41`) has **no enum/Literal validation**, and nothing
cross-checks the resolved calculator's `problem_type` against the node's
`step_type`.

```text
random_forest      -> ('random_forest_classifier', 'classification')   # always classifier
gradient_boosting  -> ERROR: Node 'gradient_boosting' not found in registry
```

**Attack path:** `POST /api/pipeline/run` with `step_type="RegressionNode"`,
`params={"model_type": "random_forest", "target_column": "<continuous col>"}`.
The engine resolves `random_forest_classifier`, never checks it against the
continuous target, and trains a `RandomForestClassifier` on continuous `y` —
sklearn accepts it (treating each unique float as a class label), producing a
nonsensical model reported with accuracy instead of R², with no error surfaced.

**Not reachable from the shipped UI** (which always sends the exact registry id
fetched live from `/registry`), but fully reachable from any direct API client.

**Fix:** Drop the alias map, or make it task-aware (accept `task`/`step_type` and
pick classifier vs regressor). Either way, add a post-resolution assertion that
`calculator.problem_type` matches the node's step type.

**Confidence:** 8/10

---

### OC-69
### 🟠 High — The engine trusts `config.nodes` list order and never verifies it is topologically sorted

**Files:** `backend/ml_pipeline/_execution/_schema_graph.py:49-70`,
`backend/ml_pipeline/_execution/engine/__init__.py:135-171` (`_run_node_loop`),
`backend/ml_pipeline/_execution/_cycle_validation.py`

Both `predict_schemas` and `_run_node_loop` iterate `for node in config.nodes:`
assuming the list is already topologically sorted. `validate_no_cycles()` runs
first but **only detects cycles** — it never verifies or restores topological
order.

A diamond merge fed by two branches of unequal depth produces exactly this. The
frontend's `pipelineConverter.ts` orders nodes by BFS-from-roots, which enqueues a
merge node the instant *any* parent is dequeued, not when *all* parents are — so
the shipped UI can itself emit a fully acyclic but misordered `nodes` list.

```text
Graph: loader -> a1 -> a2 -> a3 --\
       loader -> b ----------------> D (inputs=[a3, b])
nodes list order: loader, a1, b, a2, D, a3     # D before a3, still acyclic

validate_no_cycles:  PASSED
predict_schemas():   D -> None                 # silently unpredictable
engine.run():        pipeline status: failed
                     D -> failed  Artifact not found: a3
```

**Impact:** A valid acyclic pipeline fails with the exact cryptic "Artifact not
found" error that `_cycle_validation.py`'s own docstring says it exists to
eliminate — but it only eliminates it for cycles. `/schema-preview` silently
degrades the merge node to `None`, which the UI renders as "unknown schema"
rather than "your pipeline is misordered."

**Fix:** Have the engine run its own Kahn's-algorithm sort of `config.nodes`
before both `predict_schemas` and `_run_node_loop` — `_kahn_topological_order`
already exists in `graph_utils.py`. Or add an explicit "is this a valid
topological order" check beside `validate_no_cycles` that raises the same clear,
actionable error.

**Confidence:** 9/10

---

### OC-70
### 🟡 Medium — The leakage validator checks for *a* splitter globally, not that *this* training branch is protected

**File:** `backend/ml_pipeline/_execution/_leakage_validation.py:189-267`

`validate_no_preprocessing_before_split` reports `status: "no_split"` only when the
**entire graph** has zero splitters. Otherwise it flags a data-dependent node only
if that node is a topological ancestor of *some* splitter *anywhere* in the graph.

So a multi-branch pipeline where one branch is properly split and a second,
independent branch (same loader, different path) trains on the full dataset with
no splitter of its own gets a clean `"passed"` verdict:

```text
loader -> splitter -> training_A                    (protected)
loader -> StandardScaler(scaler_B) -> training_B    (NOT protected — no splitter)

validate_no_preprocessing_before_split(nodes) =>
{"status": "passed", "messages": [], "splitters": ["splitter"],
 "checked": [{"node_id": "scaler_B", "step_type": "StandardScaler",
              "before_split": false, "violation": false}],
 "exempted": []}
```

**Impact:** `training_B`'s scaler fits mean/std on 100% of the rows with no
held-out test set, yet the Job Details "leakage verdict" modal — which surfaces
this dict verbatim via `pipeline_result.leakage_verdict` — tells the user the
pipeline is clean. This directly contradicts the module's stated purpose and
yields silently over-optimistic metrics for that branch.

> Note the contrast with [report 08](./08-modeling-tuning.md#leakage-audit-table--all-clean):
> `skyulf-core`'s *own* fit/apply boundaries are all leak-free. The gap is in the
> backend's **advisory validator**, which tells users a leaking graph is safe.

**Fix:** For every leaf `training`/`tuning` node, walk *its* ancestors and require
a splitter (or explicit CV mode) on that specific path; flag data-dependent
ancestors of any unprotected training leaf.

**Confidence:** 7/10

---

### OC-71
### 🟠 High — No authentication or authorization anywhere on the API, despite a scaffolded `User`/ownership model

**Files:** `backend/main.py:373-395` (`_include_routers` — no `dependencies=[Depends(...)]`
on any router), `backend/dependencies.py` (no `get_current_user`),
`backend/database/models.py:39-133`, `:151-159`,
`backend/config/mixins/security.py:41-46`

A `User` table exists (password hash, `is_admin`, "Compatible with existing
Flask-Login users"), as does `DataSource.created_by`/`.creator`. But
`DataSource.has_permission()` is a literal placeholder that always returns `True`.
No route in `model_registry/api.py`, `deployment/api.py`, the `ml_pipeline`
routers, or `data_ingestion/router.py` carries an auth dependency — grepping every
router for `current_user` / `get_current_user` / `has_permission` returns **zero
hits**. `AUTH_FALLBACK_*` settings exist and are validated but are never consumed
outside their own validator.

**Attack path:** Any client that can reach the backend (exposed on `0.0.0.0` per
`_DEV_DEFAULTS["HOST"]`) can list and read every dataset, training job, deployed
model and artifact listing (`GET /api/registry/models`,
`GET /api/registry/artifacts/{job_id}`, `GET /api/pipeline/...`) for every user,
and submit, cancel or retrain pipelines — with no credentials.

**Impact:** Complete authn/authz bypass across the entire API surface. The DB
schema anticipates per-user ownership that is never enforced.

**Fix:** Add a real auth dependency (JWT/session), enforce via
`include_router(..., dependencies=[Depends(get_current_user)])`, and make
`has_permission()` (plus equivalents on jobs and deployments) actually compare
`created_by` against the requester.

**Confidence:** 9/10 that no auth exists. Lower confidence on intent — but the
scaffolded `User`/ownership model plus the dead `AUTH_FALLBACK_*` settings
strongly suggest auth was planned and never finished. **Confirm the intended
deployment model before prioritising.**

---

### OC-72
### 🟡 Medium — Insecure-by-default config: unset `FASTAPI_ENV` fails *open* to `DEBUG=True` + wildcard CORS with credentials

**Files:** `backend/config/factory.py:26`, `backend/config/environments.py:16-21`,
`backend/main.py:359-366`

```python
env = os.getenv("FASTAPI_ENV", "development").lower()   # fails open to dev
```

The factory silently falls back to `DevelopmentSettings` when `FASTAPI_ENV` is
unset **or misspelled**, whose defaults are `CORS_ORIGINS=["*"]` and
`HOST="0.0.0.0"`, combined at `main.py:359-366` with `allow_credentials=True`.

Starlette does not send a blanket `*` in that combination — it reflects the
request's actual `Origin` verbatim and sets `Access-Control-Allow-Credentials: true`:

```python
# starlette/middleware/cors.py
if self.allow_all_origins and self.allow_credentials:
    self.allow_explicit_origin(headers, origin)   # reflects ANY origin + credentials
```

**Impact:** Any operator who forgets the env var — there is no fail-closed default —
gets a fully origin-reflecting credentialed CORS policy. There is no session cookie
to steal *today* (see OC-71), but this becomes a live CSRF/data-theft vector the
moment cookie auth is added.

**Fix:** Make the factory fail closed — require `FASTAPI_ENV` explicitly, or
default to `ProductionSettings` and require an opt-in for development. Never
combine wildcard origins with `allow_credentials=True`, even in dev.

**Confidence:** 8/10

---

### OC-73
### ⚪ Low — `DataSource.credentials` is documented as "encrypted in production" but stored as plaintext JSON

**File:** `backend/database/models.py:107`

```python
credentials: Mapped[Any | None] = mapped_column(JSON, nullable=True)
```

The field comment claims encryption "in production," but no encryption or
decryption code references `credentials` anywhere in the connectors or the
ingestion service.

**Impact:** Anyone with DB read access sees connector secrets (DB passwords, API
keys) in the clear. Combined with OC-71, anyone who can reach a future endpoint
that serializes raw `DataSource` rows sees them too.

**Fix:** Implement field-level encryption (e.g. `sqlalchemy-utils` `EncryptedType`
or app-level envelope encryption), or correct the misleading comment and treat
at-rest DB encryption as the only protection.

**Confidence:** 6/10 — no code path currently reads `credentials` back into a
user-facing response, so exposure needs DB access today.

---

## Security checklist

| Class | Checked | Result | file:line |
|---|---|---|---|
| Path traversal (upload/download) | ✅ | **Not found** — `resolve_safe_path` and `_resolve_local_artifact_path`/`_sanitize_artifact_name` both resolve + verify containment via `Path.resolve()`/`os.path.realpath` before use | `data_ingestion/connectors/file.py:39-73`, `ml_pipeline/artifacts/factory.py:126-155`, `artifacts/local.py:18-25` |
| Unsafe deserialization | ✅ | `joblib.load` (pickle-based) used for artifact persistence, but keys are backend-generated, traversal-guarded, and content is only ever written by the same backend — no user-content path reaches it | `artifacts/local.py:33-44`, `artifacts/s3.py:130-144` |
| SQL injection | ✅ | **Not found** — every `text(query)` is either fully parameterized or unreachable dead code (`execute_query`/`execute_update` have zero callers); dynamic `table()`/`column()` builders are unreachable from any route (the live `DELETE /sources/{id}` uses ORM `session.delete`) | `database/async_connection_manager.py:242-266`, `database/data_sources/async_postgres_queries.py:104-120` |
| Authn | ✅ | **Missing entirely** | OC-71 |
| Authz / IDOR | ✅ | **Missing** — `has_permission()` stub always `True` | `database/models.py:151-159` |
| SSRF | ✅ | **Not found** — `_apply_trusted_endpoint` unconditionally strips caller-supplied `endpoint_url`/`aws_endpoint_url`, specifically to block metadata-endpoint SSRF | `data_ingestion/connectors/s3.py:54-67` |
| Secrets in source | ✅ | None in code; DB `credentials` unencrypted | OC-73 |
| CORS | ✅ | Dev default reflects any origin **with credentials**, and the env selector fails open | OC-72 |
| Upload limits | ✅ | Bounded — `MAX_UPLOAD_SIZE` enforced by both a declared `Content-Length` pre-check **and** streamed byte counting during write (spoof-resistant) | `data_ingestion/service.py:391-449` |
| Rate limiting | ✅ | Present — `slowapi` `Limiter`, `200/minute` baseline on undecorated routes, stricter per-route limits elsewhere | `middleware/rate_limiter.py` |
| Async blocking | ✅ | **Not found** — training runs via Celery `.delay()` or threadpool-offloaded `BackgroundTasks`, never inline in an `async def` handler | `_internal/_routers/run_pipeline.py:263-291` |

---

## Contract drift: frontend → backend → skyulf-core

| Wire value from UI | Backend handling | Core node id | Match? |
|---|---|---|---|
| `ClassificationNode`/`RegressionNode` `model_type` (fetched live from `/registry`) | `_node_runners.py:685-688` reads `model_type` directly, no alias needed | exact registry id | ✅ |
| `SegmentationNode` `model_type: "kmeans"` | same path | `kmeans` | ✅ |
| `EnsembleNode` `model_type: "voting_classifier"`/`"stacking_classifier"` | same path | registered in `ensemble.py`, category `"Ensemble"` | ✅ — but see **OC-74** |
| `EnsembleNode.base_estimators: ["random_forest", …]` | never touches the alias map; resolved inside `ensemble.py`'s own `BASE_ESTIMATORS_CLF/REG` factories — a deliberate, documented, separate short-key namespace | resolved via `ensemble.py`, not `NodeRegistry` | ✅ by design |
| Direct-API `model_type: "random_forest"`/`"gradient_boosting"`/`"svm"` (never sent by the UI) | `_get_model_components` alias map | ambiguous → classifier only; most unmapped → `ValueError` | ❌ **OC-68** |

---

## What I checked and found sound

- **Artifact path containment** — read all of `artifacts/local.py`, `s3.py`,
  `factory.py`, `discovery.py`. Every local path construction goes through a
  `realpath` + containment check; folder names are sanitized against `.`/`..`.
- **`ModelRegistryService.get_next_version`** — atomic `UPDATE … RETURNING` with
  retry-on-`IntegrityError` correctly prevents the classic read-then-write version
  race under concurrent job submissions. No double-allocation possible.
- **`validate_no_cycles`** — the Kahn's-algorithm implementation is correct and
  precisely names loop members, pruning nodes that are merely downstream of a loop.
- **`predict_schemas` degradation** — `None` schemas, disconnected/dangling nodes
  and non-seeded loaders all degrade cleanly to `None`. No crash paths found.
- **Upload size enforcement** — confirmed the streamed write aborts and deletes the
  partial file once `MAX_UPLOAD_SIZE` is exceeded, rather than trusting the
  (spoofable) `Content-Length` header alone.

---

## Improvement opportunities (not defects)

- The `alias_map` appears to be entirely dead relative to the current frontend
  contract. Either remove it, or promote it to a documented, test-covered public
  alias contract with one entry per registry id.
- `ModelRegistryService.get_model_versions` duplicates ~40 lines of `ModelVersion`
  construction already in `_train_job_to_version`/`_tune_job_to_version` — the
  docstring itself admits "for brevity, just filtered… in production we should
  query specifically."

---

## Backend infrastructure — findings verified directly by the lead auditor

Surfaced by an agent sweep of `backend/config/`, `middleware/`, `monitoring/`;
**every claim below was re-executed and confirmed by the lead auditor** before
filing.

<a id="oc-130"></a>
### OC-130
### 🟠 High — A typo in `FASTAPI_ENV` silently disables the entire production security posture (fail-open)

**Files:** `backend/config/factory.py:27-32`, `backend/config/base.py:188-189`,
`backend/config/environments.py:16-30`, `backend/main.py:361-363`

`get_settings()` selects the settings class by exact string match, with
`DevelopmentSettings` as the **default for anything unrecognised**:

```python
env = os.getenv("FASTAPI_ENV", "development").lower()
env_map = {"production": ProductionSettings, "testing": TestingSettings}
settings: Settings = env_map.get(env, DevelopmentSettings)()   # <- fail-OPEN
```

`DevelopmentSettings` sets `CORS_ORIGINS=["*"]` and `DEBUG=True`, and
`main.py:362-363` pairs that with `allow_credentials=True`.

**The core problem: the correct spelling fails *loudly*, a typo succeeds
*silently*.** Executed across candidate values:

```
FASTAPI_ENV='production'    -> ValidationError: SECRET_KEY must be explicitly set   (correct: refuses to boot)
FASTAPI_ENV='PRODUCTION'    -> ValidationError: SECRET_KEY must be explicitly set   (correct: .lower() normalises)
FASTAPI_ENV='prod'          -> DevelopmentSettings  DEBUG=True  CORS=['*']          (SILENT FALLBACK)
FASTAPI_ENV='production '   -> DevelopmentSettings  DEBUG=True  CORS=['*']          (SILENT FALLBACK — trailing space)
FASTAPI_ENV=<unset>         -> DevelopmentSettings  DEBUG=True  CORS=['*']          (intended)
```

A deployment that sets `FASTAPI_ENV=prod` — an extremely common abbreviation — or
that picks up a trailing space from a YAML/`.env`/CI variable, boots
**successfully** into full development configuration.

**Consequence 1 — wildcard CORS with credentials is a genuine vulnerability.**
With `allow_origins=["*"]` *and* `allow_credentials=True`, Starlette does not
send a literal `*`; it **reflects the caller's origin**, which defeats the
browser's normal refusal to combine `*` with credentials:

```
dev default CORS_ORIGINS=['*']         ACAO=https://evil.example   ACAC=true
CONTROL prod allow-list                ACAO=None                   ACAC=true

dev config reflects attacker origin: True   with credentials: True
prod config blocks attacker origin : True
```

The control line is what makes this trustworthy: the same probe against the
production allow-list correctly returns no `Access-Control-Allow-Origin`. Any
website could issue credentialed cross-origin reads against the API.

**Consequence 2 — it bypasses a guardrail the codebase deliberately built.**
`base.py:180-196` refuses to start production without an explicit `SECRET_KEY`,
and its docstring states the exact harm:

> *"The default value is a randomly generated token — different per process and
> per restart — which breaks JWT validation across workers and makes all
> existing tokens invalid after a restart."*

But that check keys off the **same** exact-match string
(`os.getenv("FASTAPI_ENV", "development").lower() == "production"`), so the typo
skips it too. Confirmed the default really is per-process random — two runs under
`FASTAPI_ENV=prod` produced `zMmfSQ6mCgWWFkNwwCieJW4j…` and
`O-eJAScDYL1AYSaPAmMaOSzW…`. Under multi-worker gunicorn/uvicorn each worker
signs with a different key, so users are logged out at random as requests
round-robin between workers.

To be fair to the code: the dev secret is **randomly generated, not hardcoded**,
so there is no committed secret to leak — that part is done well. The defect is
purely the fail-open selection.

**Also lost to the same typo:** `DEBUG=False`, the `ALLOWED_HOSTS` allow-list,
and the `_PROD_SECURITY_HEADERS` block (HSTS, `X-Frame-Options: DENY`, CSP,
`nosniff`) — all defined in `environments.py` and all skipped.

**Fix:** fail closed on an unrecognised value rather than defaulting to the most
permissive profile:

```python
env = os.getenv("FASTAPI_ENV", "development").strip().lower()
env_map = {"development": DevelopmentSettings,
           "production": ProductionSettings,
           "testing": TestingSettings}
if env not in env_map:
    raise ValueError(f"Unknown FASTAPI_ENV={env!r}; expected one of {sorted(env_map)}")
settings = env_map[env]()
```

Note `.strip()`, which alone would fix the trailing-space case. Independently,
`main.py` should refuse to combine `allow_origins=["*"]` with
`allow_credentials=True` regardless of environment.

<a id="oc-131"></a>
### OC-131
### ⚪ Low — Drift and diagnostic computations *fail open*, reporting "healthy" when the maths breaks

**Files:** `skyulf-core/skyulf/profiling/drift.py:474-476`,
and the same shape at `profiling/_analyzer/numeric.py` ([OC-113](./15-profiling-analyzers.md#oc-113))

`_calculate_psi` wraps its whole computation in:

```python
except Exception:  # noqa: BLE001 - PSI numeric failure reports no drift (0.0)
    return 0.0
```

`0.0` is not a neutral value here — it is the *strongest possible assertion of
"no drift at all"*. Any numeric failure (an empty reference window, a malformed
histogram, a dtype surprise) is therefore reported to the monitoring layer as a
clean bill of health, and no alert fires.

This is filed ⚪ Low, not higher, because the surrounding guards are genuinely
good and I could not force a failure — see the not-reproduced note below. The
point is the *pattern*, which now has two independent instances:

| Location | On failure returns | Meaning conveyed |
| --- | --- | --- |
| `drift.py:474-476` (PSI) | `0.0` | "no drift" |
| `_analyzer/numeric.py` (VIF, [OC-113](./15-profiling-analyzers.md#oc-113)) | `1.0` (via `max(1.0, …)`) | "no multicollinearity" |

In both cases a *diagnostic* — code whose only job is to raise a warning —
degrades into silence rather than into an honest "could not compute". OC-113 is
🟠 High because there the failure is *reachable and common*; this one is ⚪ Low
because it is well guarded. **Fix:** return `None` (the schemas already permit
optional metrics) and surface "not computed" distinctly from "computed, no
drift".

---

<a id="not-reproduced"></a>
## Not reproduced — recorded so it is not re-filed

### ❌ Non-finite floats breaking JSON in `backend/monitoring/router.py`

An agent flagged that `monitoring/router.py` (~lines 449, 569) might emit bare
`NaN`/`Infinity` tokens — invalid JSON that breaks `JSON.parse`. **This does not
reproduce, for two independent reasons.**

**1. Those call sites are not `json.dumps`.** They are pydantic `model_dump()`
calls whose result is returned from a FastAPI route; the serializer is FastAPI's
response class, not stdlib `json`. (Mis-identifying the serializer is exactly
what made the original [OC-46](./00-validation-log.md#oc-46) wrong.)

**2. FastAPI raises rather than emitting invalid JSON.** Starlette's
`JSONResponse` calls `json.dumps(..., allow_nan=False)`:

```
@app.get('/x') -> {'psi': nan, 'ks': inf, 'ok': 1.5}
RAISED: ValueError  Out of range float values are not JSON compliant: nan
```

So the failure mode would be an HTTP 500, never a malformed body.

**3. The metrics cannot go non-finite anyway.** `_calculate_psi` clips
out-of-range values into the boundary bins and floors zero frequencies at
`0.0001` before `np.log`, so `log(0) = -inf` is unreachable. KS and Wasserstein
were exercised on degenerate inputs and stayed finite:

```
constant ref & curr (all 5.0)    ks=0.0   wass=0.0      non-finite=0
single-row each                  ks=1.0   wass=1.0      non-finite=0
disjoint ranges                  ks=1.0   wass=1000.0   non-finite=0
CONTROL normal vs shifted        ks=0.68  wass=1.911    non-finite=0
```

The control confirms the probe detects real drift, so the all-finite result is
trustworthy rather than a mis-called probe.

<a id="oc-132"></a>
### OC-132
### ⚪ Low — Dead `dropped_features` branch in feature-selection column extraction

**File:** `backend/ml_pipeline/_execution/graph_utils.py:534-537`

`_extract_columns` reads a config key that nothing in the repository ever writes:

```python
if ntype == "feature_selection":
    dropped_feats = params.get("dropped_features")
    if isinstance(dropped_feats, list):
        dropped.extend(dropped_feats)
```

An exhaustive search — **all** file types, excluding `.venv`/`node_modules`/
`.git`/`__pycache__` — finds the string exactly once, at this read site:

```bash
grep -rn "dropped_features" . --exclude-dir=.venv --exclude-dir=node_modules \
     --exclude-dir=.git --exclude-dir=__pycache__ | wc -l
# 1   (graph_utils.py:535 — the read itself)
```

No frontend component, no backend writer, no test fixture, no saved-pipeline
schema produces it. I confirmed the node type string itself is *not* the problem:
`pipelineConverter.ts:302-304` does emit `definitionType === 'feature_selection'`
and forwards `params = node.data` byte-for-byte, so the `ntype` comparison
matches — it is specifically the **key** that never exists.

**Why the branch is unnecessary (and why this is ⚪ Low, not a real bug).** Which
features a selector drops is a *runtime* result, not config — it depends on the
fitted threshold/`k`. The codebase already handles it correctly on the runtime
path, which I verified end to end:

| step | location |
| --- | --- |
| written after execution | `_execution/strategies.py:133` — `final_metrics["dropped_columns"] = all_dropped_columns` |
| read back for job details | `_execution/basic_training_manager.py:82-84` — *"Also check job metrics for runtime dropped columns (e.g. from Feature Selection)"* |
| merged + de-duplicated | `basic_training_manager.py:87` — `list(set(dropped_columns))` |

So feature-selection drops **do** reach `JobInfo`; they simply arrive via
`job.metrics["dropped_columns"]` rather than the config key. The dead branch
costs nothing at runtime but implies a config-time contract that does not exist,
and a maintainer adding a `dropped_features` field to the UI would reasonably
expect it to work.

**Fix:** delete the branch, or add a comment pointing at the metrics path that
supersedes it. Note the naming inconsistency worth resolving either way — the
dead key is `dropped_features` while every live path uses `dropped_columns`.

---

<a id="oc-145"></a>
### OC-145
### 🟡 Medium — A crashed cross-validation is indistinguishable from a disabled one; the job still reports success

**File:** `backend/ml_pipeline/_execution/engine/_node_runners.py:871-907`

`_run_cross_validation_tuned` returns the **same `{}` sentinel** for two
completely different outcomes:

```python
if not tuning_params.get("cv_enabled", False):
    return {}                      # :872  CV deliberately switched off — benign
...
except Exception:
    logger.exception("Cross-validation failed for tuned model")
    return {}                      # :906  CV CRASHED — should be surfaced
```

The caller cannot tell them apart, and consumes the result as:

```python
metrics.update(cv_metrics)         # :591  with {} this is a silent no-op
```

So a crashed cross-validation produces a job that **completes successfully**,
with a saved model, and metrics that simply lack every `cv_*` key. The only
trace is a `logger.exception` in the server log; nothing reaches the API
response or the UI. A user who explicitly enabled CV sees no CV numbers and no
error, and the most natural reading of that is "CV was off" — the exact wrong
conclusion.

This matters more than a normal swallowed exception because **the user asked for
CV specifically**: `cv_enabled=True` is an explicit opt-in, so silently
downgrading to "no CV" discards a requested validation step.

**Contrast with the sibling code path**, which shows the codebase already knows
how to do this properly. Immediately above, at `:567-588`, error-swallowing
during evaluation is controlled by an **explicit named flag**:

```python
if swallow_evaluate_errors:
    try: self._evaluate_and_save_report(...)
    except Exception: logger.exception("Failed to evaluate tuned model")
else:
    self._evaluate_and_save_report(...)       # errors propagate
```

That is a deliberate, reviewable policy decision — the caller chooses. The CV
path hard-codes the swallow with no opt-out.

**Fix:** distinguish the two outcomes. Return `None` for "crashed" and `{}` for
"disabled" (or attach a `cv_status` / `cv_error` field to the job record), so the
API can report *"cross-validation failed"* rather than silently omitting the
metrics. This is the same **fail-open diagnostic** pattern as
[OC-131](#oc-131) (PSI → `0.0`) and
[OC-113](./15-profiling-analyzers.md#oc-113) (VIF → `1.0`): code whose job is to
validate degrades into silence instead of an honest "could not compute".

---

## Backend re-audit — checked and found sound

Additionally verified in the final backend pass:

- **Model version allocation is race-free.** `ModelRegistryService.get_next_version`
  (`model_registry/service.py:42-95`) uses a single atomic
  `UPDATE … SET current_version = current_version + 1 … RETURNING`, backed by a
  dedicated `ModelVersionCounter` table, with an `IntegrityError`-retry loop for
  the first-insert race and a bounded attempt count that raises rather than
  looping forever. Both the model docstring and the method docstring explicitly
  record that this replaced a `SELECT MAX(version)` TOCTOU race — the fix is
  correct and well documented.
- **Pipeline path traversal is properly blocked.** `_pipeline_json_path`
  (`_routers/pipelines_io.py:51-60`) validates against a strict allow-list
  (`^[A-Za-z0-9_-]+$`) *before* constructing any `Path`, and the caller maps the
  resulting `ValueError` to HTTP 400. Executed:

  ```
  '../../etc/passwd' -> REJECTED      '/etc/passwd' -> REJECTED
  'a/../../b'        -> REJECTED      '..'          -> REJECTED
  'a\x00b'           -> REJECTED      'normal_id-123' -> /tmp/store/normal_id-123.json
  ```

  Rejecting on an allow-list before path construction is the right order of
  operations (normalise-then-check schemes are what usually fail here).
- **`swallow_evaluate_errors` is a deliberate policy flag, not a swallowed
  exception.** At `_node_runners.py:567-588` the tuned path wraps
  `_evaluate_and_save_report` in `try/except` while the fixed path lets errors
  propagate — but the behaviour is selected by an explicit named parameter, so
  the caller chooses. **Investigated and dropped as a false positive**; it is the
  counter-example that makes [OC-145](#oc-145) (which hard-codes the swallow with
  no opt-out) a genuine defect rather than a house style.
- **The `except ImportError` handlers are legitimate optional-dependency guards**
  (`preview.py:78,99,134`, `eda/tasks.py:259`), and `realtime/router.py:32`
  catches `WebSocketDisconnect`, which is normal control flow — none of these are
  error-swallowing.


The `backend/ml_pipeline/` and data-layer sweeps were done directly by the lead
auditor after the assigned agents were lost to a session restart. Verified
correct:

- **Model version allocation is genuinely race-free.** `ModelRegistryService.get_next_version`
  (`ml_pipeline/model_registry/service.py:42-95`) allocates via a single
  `UPDATE … RETURNING` statement, so concurrent submissions are serialised by the
  database rather than in application code. The first-time seed path is also
  handled correctly: the `INSERT` is wrapped in `try/except IntegrityError` with a
  `rollback()` + retry, so if a concurrent request wins the primary-key race, the
  loser retries the atomic `UPDATE` against the row the winner just created. The
  loop is bounded by `_MAX_VERSION_ALLOCATION_ATTEMPTS` and raises a descriptive
  `RuntimeError` rather than spinning. `ModelVersionCounter`'s own docstring
  (`database/models.py:355-363`) documents that this replaced *"the previous
  read-then-write via `SELECT MAX(version)`, a classic TOCTOU race"* — i.e. this
  bug class was found and fixed before this audit.
- **The backend/frontend param contract holds for the execution-layer keys.**
  Every key the execution engine reads from `node.params` —
  `cv_type`, `cv_time_column`, `cv_enabled`, `cv_folds`, `cv_shuffle`,
  `cv_random_state`, `tune_threshold`, `run_mode`, `execution_mode`,
  `_merge_strategy`, `reference_column`, `dataset_id`, `sample`, `limit`,
  `hyperparameters`, `metric`, `search_space`, `target_column` — was cross-checked
  against `frontend/ml-canvas/src`, and every one is produced there. The single
  exception is `dropped_features` ([OC-132](#oc-132)), which is dead on both
  sides. This is a notably better result than the *core* node-param layer, where
  [R1](../opus_core_analysis.md#r1) covers 11+ live mismatches.
- **PSI is numerically well guarded.** `skyulf/profiling/drift.py:455-474` clips
  both arrays into the histogram's outer breakpoints *before* binning — with a
  comment explaining that the very scenario drift detection exists to catch is
  `actual` shifting outside `expected`'s range — and floors zero frequencies at
  `0.0001` so `log(0) = -inf` is unreachable.


---

## Round 3 — uncovered backend subpackages

`backend/` is 27.5k lines and had only five findings, all in `ml_pipeline/` and
`config/`. This round targeted the subpackages with **no findings at all**:
`data_ingestion/` (2,280), `database/` (3,287), `realtime/` (485),
`middleware/` (214).

<a id="oc-150"></a>
### OC-150
### 🟠 High — S3 error "sanitiser" is keyword-based and case-sensitive; real AWS credential formats pass through into logs unredacted

**File:** `data_ingestion/connectors/s3.py:31-37` **and** `ml_pipeline/artifacts/s3.py:67-73`
(two byte-identical copies)

The redactor triggers only when the message contains one of four literal
*key names*:

```python
for secret in ("aws_secret_access_key", "aws_access_key_id", "secret=", "key="):
    if secret in message:
        return "redacted sensitive S3 error"
return message          # <-- otherwise the full message is logged verbatim
```

It matches on the **name of the setting**, never on the shape of a credential —
and `in` is case-sensitive. AWS surfaces credentials in error text in formats
that contain none of those four strings, most importantly S3's own 403 response
body and any presigned URL. Note `AWSAccessKeyId=` does **not** contain `key=`
(capital `K`), so the most common form misses by a single character.

**Measured** through the real `S3Connector._sanitize_error` (both controls
behave correctly, so the probe is proven to discriminate):

| error message | redacted? | outcome |
|---|---|---|
| options dict with `aws_access_key_id` *(control +)* | ✅ yes | correct |
| `Generic connection timeout after 30s` *(control −)* | ✅ no | correct |
| S3 403 XML body — `<AWSAccessKeyId>AKIA…</AWSAccessKeyId><SignatureProvided>…` | ❌ **no** | ⛔ key ID + signature logged |
| presigned URL SigV2 — `?AWSAccessKeyId=AKIA…&Signature=…` | ❌ **no** | ⛔ **replayable URL logged** |
| presigned URL SigV4 — `?X-Amz-Credential=AKIA…&X-Amz-Signature=…` | ❌ **no** | ⛔ **replayable URL logged** |

The presigned-URL cases are the serious ones: a presigned URL is a *bearer
credential*. Anyone with log read access can replay it against the object until
it expires — no AWS account required.

**Amplifier:** at both call sites the sanitiser is applied to the exception but
`self.path` is interpolated **raw** —
`logger.error("Failed to fetch data from %s: %s", self.path, self._sanitize_error(e))`
(`s3.py:92`, `:178`). Since `path` is caller-supplied, a presigned URL passed as
the path leaks in full even when the exception itself is redacted.

**Fix:** invert the test — redact by *value shape*, not by key name, and scrub
in place instead of discarding the whole message (all-or-nothing redaction also
destroys the diagnostic, which is why an engineer will be tempted to disable it):

```python
_REDACT = re.compile(
    r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}"          # access key IDs
    r"|(?i:x-amz-signature|signature)=[^&\s]+"      # signatures
    r"|(?i:x-amz-credential)=[^&\s]+"
    r"|(?i:aws_secret_access_key|secret)[\"'=:\s]+[^\s,\"'}]+)"
)
def _sanitize_error(error: Exception) -> str:
    return _REDACT.sub("[REDACTED]", str(error))
```

Apply the same to `self.path` before logging. And **de-duplicate** the two
copies into one shared helper — a security control maintained in two places
will drift, and only one of them will get fixed.

---

<a id="oc-151"></a>
### OC-151
### 🟡 Medium — Trial-buffer cleanup functions are never called; ~111 MB of completed-job chart data is retained for the process lifetime

**File:** `realtime/trial_buffer.py:56-59` (`clear_trials`), `:103-106` (`clear_iterations`)

Both cleanup functions are documented as lifecycle hooks —
*"Drop the job's buffer (called when it can no longer be backfilled)"* — and the
module docstring states the persisted `metrics.trials` list *"takes over once the
job is terminal"*, i.e. the buffer is redundant after completion.

Neither function is called anywhere in the repository:

```
$ grep -rn "clear_trials\|clear_iterations" backend/ --include=*.py
backend/realtime/trial_buffer.py:56:def clear_trials(...)
backend/realtime/trial_buffer.py:103:def clear_iterations(...)
```

Only `record_trial` / `record_iteration` are imported (by
`_node_runners.py:30`). Entries therefore leave only via LRU eviction after
**128 newer jobs**, not at job completion.

**Measured** by filling to the module's own documented bounds
(`_MAX_JOBS=128`, `_MAX_TRIALS_PER_JOB=2000`) and deep-sizing both dicts:

```text
jobs buffered     : 128 trials + 128 iterations
retained at bound : 110.9 MB   (never freed)
clear_* DO work   : job-0 trials -> 0 after call
```

The functions are correct; they are simply unused. This is bounded, so it is not
an unbounded leak — but it is ~111 MB of resident memory in a long-lived API
process holding data the module itself describes as superseded, and it is
retained for jobs that finished days earlier.

**Fix:** call `clear_trials(job_id)` / `clear_iterations(job_id)` where a job
reaches a terminal state and its metrics are persisted (the same place
`metrics.trials` is written). Keeping LRU as the backstop is still right.

---

<a id="oc-152"></a>
### OC-152
### ⚪ Low — Two raw-SQL executors accept caller-built query strings and have zero callers

**File:** `database/async_connection_manager.py:243-268`

`execute_query(query: str, ...)` and `execute_update(query: str, ...)` pass
their argument straight to `text(query)`. Parameters are correctly bound
(`params or {}`), so *values* are safe — but the query string itself is
unconstrained, making these a ready-made injection sink for the first caller who
interpolates a column or table name.

They currently have **no callers anywhere in the repository** — every live query
path uses SQLAlchemy constructs or a parameterised `sa_text` with named binds
(e.g. `async_postgres_queries.py:133`, which is correctly written).

Filed as Low precisely because it is *latent*: it is not a live vulnerability,
it is a loaded footgun sitting in a shared manager class where it looks
blessed.

**Fix:** delete both, or restrict them to an allow-list of known statements.

---

## Checked this round and found sound

* **Dynamic filter columns** — `async_postgres_queries.py:85-111` builds
  `where` clauses from caller-supplied dict *keys* via `column(k)`, and
  `_normalize_filter` (`async_data_sources_crud.py:163`) does **not** allow-list
  them. This is **not** injectable: SQLAlchemy quotes identifiers, so a hostile
  key yields a quoted, non-existent column and an error, not escaped SQL.
* **S3 endpoint SSRF** — `_apply_trusted_endpoint` (`s3.py:53-67`) explicitly
  pops caller-supplied `endpoint_url`/`aws_endpoint_url` and substitutes the
  server-configured value, with a comment naming the metadata-endpoint attack.
  Correct, and a good model for how OC-150 should have been written.
* **Postgres URL building** — `connections/postgres/async_connection.py:52-60`
  percent-encodes user and password via `quote_plus` before assembling the DSN.
* **Trial buffer concurrency** — every mutation and read is under a single
  `threading.Lock`, and `get_trials` returns a deep-copied snapshot rather than
  a live reference. Correct.
* **Rate limiter** — `middleware/rate_limiter.py` keys on
  `get_remote_address` with a `default_limits` safety net for undecorated
  routes. Behind a reverse proxy this would collapse all clients into one
  bucket, but the repository contains **no proxy or deployment config**, so the
  precondition cannot be established. **Deliberately not filed** rather than
  filed on speculation.
* **Naive `datetime.now()` uses** (`strategies.py:199`,
  `pipeline_execution_service.py:63-107`) are local log timestamps and
  elapsed-time deltas compared only against other naive values. Not defects.

---

## Round 4 — `ml_pipeline/` deep pass (execution engine, deployment, services)

Targeted at the 38 `ml_pipeline` files (7,296 lines) that no earlier round had
cited — the merge engine, the deployment/serving path, and the tuning/evaluation
services. Every finding below was **reproduced by execution with a passing
control**; nothing here is filed on reading alone.

<a id="oc-153"></a>
### OC-153 — 🟠 High — Row-count mismatch silently switches merge semantics and duplicates rows

**File:** `backend/ml_pipeline/_execution/engine/_merge.py:338-348`

`_merge_frames` picks its merge mode purely from row counts:

```python
same_rows = all(rc == row_counts[0] for rc in row_counts)
merged = (
    self._merge_frames_columnwise(frames, node_id, strategy, prefix)
    if same_rows
    else self._merge_frames_rowwise(frames, node_id, part_label, prefix, row_counts, col_sets)
)
```

A user who wires two branches into a merge node is expressing a *column-wise*
intent (feature union). If either branch changes the row count — outlier
removal, deduplication, row filtering, any `dropna` — the engine **silently
switches to a row-wise concat** instead. The result is not a wider frame but a
taller one containing duplicate rows.

**Reproduced.** A 5-row dataset merged with its own outlier-filtered branch:

| | rows | cols | duplicate rows | `merge_warnings` surfaced to UI |
|---|---:|---:|---:|---:|
| input A (raw) | 5 | 2 | — | — |
| input B (filtered) | 3 | 2 | — | — |
| **merged output** | **8** | **2** | **3** | **0** |

The three filtered rows appear **twice** in the training frame. Duplicated rows
silently reweight those observations and, if the merge feeds a split node,
place identical rows on both sides of the train/test boundary.

**Why nobody sees it.** `_merge_frames_rowwise` only appends to
`merge_warnings` when the branches' *column sets* differ
(`if any(common_cols != cs for cs in col_sets)`). In this scenario the columns
are identical, so the condition is false and **no UI warning is emitted at
all** — the only trace is one `self.log` line buried in the job log.

**Control.** With equal row counts the same probe takes the column-wise path
and returns `(5, 3)` with 0 duplicates, confirming the probe distinguishes the
two branches rather than always reporting duplication.

**Suggested fix.** Treat a row-count mismatch on a multi-input node as a
condition the user must resolve: either raise, or emit a `merge_warnings` entry
of the same weight as `row_concat_drop`. Silently changing the *shape semantics*
of a merge is not a safe default.

<a id="oc-154"></a>
### OC-154 — 🟠 High — Serving-time feature-order alignment fails open, silently mispredicting

**File:** `backend/ml_pipeline/deployment/service.py:438-442`

The reindex added by fix **F-02** guarantees the model receives columns in
training order — but only when it can verify the set matches:

```python
feature_columns = artifact.get("feature_columns")
if feature_columns and hasattr(X_transformed, "columns"):
    missing = [c for c in feature_columns if c not in X_transformed.columns]
    if not missing:
        X_transformed = X_transformed[feature_columns]
```

When `missing` is non-empty the reindex is **skipped silently** — no warning, no
error — and the frame is handed to `predict()` in whatever order the feature
engineer produced. This is exactly the state F-02 was introduced to prevent, and
the guard disables the protection precisely when alignment cannot be confirmed.

**Reproduced.** A model trained on `a, b, c` and served `c, a, b`:

| Estimator fitted on | Result of misordered predict |
|---|---|
| a **DataFrame** (has `feature_names_in_`) | ✅ raises `ValueError: The feature names should match…` |
| bare **numpy** (no feature names) | ❌ **silently returns 213.00 where the truth is 321.00** |

Control: the same numpy-fitted estimator given the correct order returns
321.00. The failure is silent and the error is arbitrary in size — it scales
with how different the swapped features' coefficients are.

**Suggested fix.** Raise on `missing` (the caller already raises a clear error
for missing *input* columns 15 lines earlier via `_validate_required_columns`),
or at minimum emit a warning. Skipping alignment should not be the fallback.

<a id="oc-155"></a>
### OC-155 — 🟠 High — Legacy predict path zero-fills missing features

**File:** `backend/ml_pipeline/deployment/service.py:457-462`

```python
missing_in_df = set(model_cols) - set(df.columns)
if missing_in_df:
    logger.warning(f"Missing columns in input DataFrame: {missing_in_df}")
    for c in missing_in_df:
        df[c] = 0
```

Absent features are imputed with the literal constant `0` and the request
proceeds. `0` is not a neutral value: for a feature whose training distribution
is centred far from zero (income, age, price, any scaled feature with a
non-zero mean) this is an extreme out-of-distribution input, and the resulting
prediction is returned to the API caller as a normal result. The only signal is
a server-side `logger.warning` the caller never sees.

**Suggested fix.** Return a 4xx naming the missing features, matching how the
bundled path's `_validate_required_columns` already behaves. If silent
imputation is genuinely wanted it must use the *training* imputation statistics
carried in the artifact, not a hardcoded `0`.

<a id="oc-156"></a>
### OC-156 — 🟡 Medium — `roc_auc` threshold objective is mathematically identical to `balanced_accuracy`

**File:** `backend/ml_pipeline/_services/threshold_tuning_service.py:77-92`

`optimize_thresholds` scores **hard, post-threshold class predictions** — the
code says so explicitly at line 197. The `roc_auc` scorer therefore calls
`roc_auc_score` on a binarized 0/1 prediction vector rather than on probability
scores. ROC AUC of a binary hard-label vector reduces exactly to
`(sensitivity + specificity) / 2` — that is, balanced accuracy.

**Reproduced** over 6 random trials (n=200): `roc_auc` and `balanced_accuracy`
agreed to **full float precision in every trial** (e.g. `0.6298714445` vs
`0.6298714445`). Control: `f1` on the same inputs differs (0.634 vs 0.630),
confirming the comparison can distinguish metrics.

Both options are offered separately in `_SUPPORTED_METRICS`, so a user choosing
`roc_auc` — reasonably expecting the standard probability-ranking metric — gets
balanced accuracy under a name that means something else, and picking either of
the two produces byte-identical thresholds.

**Suggested fix.** Either drop `roc_auc` from the threshold-tuning metric list,
or rename it so the UI does not promise ranking-based AUC. Per the
backend↔frontend sync rule, the frontend metric dropdown must be updated in the
same change.

<a id="oc-157"></a>
### OC-157 — ⚪ Low — `first_wins` silently reverses output column order

**File:** `backend/ml_pipeline/_execution/engine/_merge.py:221-236`

`ordered = indexed if strategy == "last_wins" else list(reversed(indexed))`, and
the merged frame is built from a plain dict whose **insertion order becomes the
column order**. Reversing the iteration to implement `first_wins` therefore also
reverses the output columns.

**Reproduced** — merging `A(a, b)` with `B(c, d)`:

| strategy | output column order |
|---|---|
| `last_wins` | `['a', 'b', 'c', 'd']` |
| `first_wins` | `['c', 'd', 'a', 'b']` |

Values are correct under both — only the ordering differs — so this is Low. But
it contradicts the method's own docstring ("*`first_wins` — earlier inputs are
kept; later inputs only add new columns*"), which describes earlier inputs
leading. Column order also reaches positional consumers, which is the same
hazard class as [OC-154](#oc-154).

**Suggested fix.** Resolve *ownership* in reverse for `first_wins` but emit
columns in forward input order — e.g. build `result_cols` in forward order and
let the strategy decide only which frame supplies a contested column.

---

## Round 5 — `data_ingestion/`, `database/`, `eda/`, `data/`

Coverage entering this round: 32 files, 5,237 lines, of which 22 files
(~2,300 lines) had never been cited. Reachability was established by
**importing `backend.main` and inspecting `sys.modules`**, not by grepping for
call sites — which materially changed two conclusions (see the note at the end).

<a id="oc-158"></a>
### OC-158 — 🟡 Medium — The two JSON serializers disagree; the sync one silently nulls legitimate strings

**File:** `backend/data_ingestion/serialization.py:369, 435-446`

`JSONSafeSerializer._handle_special_string_values` stringifies **any** object and
returns `None` on an exact match against a literal list:

```python
if str(obj) in ["nan","NaN","NaT","<NA>","inf","-inf","infinity","-infinity"]:
    return None
```

It is registered *before* `_handle_basic_types` in the handler chain, so a plain
`str` never reaches the passthrough handler. Any legitimate string equal to one
of those eight tokens is silently converted to `null`.

**Reproduced**, and the sync/async pair **disagree**:

| input | `JSONSafeSerializer` (sync) | `AsyncJSONSafeSerializer` |
|---|---|---|
| `"nan"`, `"NaN"`, `"NaT"`, `"<NA>"` | ❌ `None` | ✅ preserved |
| `"inf"`, `"-inf"`, `"infinity"`, `"-infinity"` | ❌ `None` | ✅ preserved |
| `"Nan"`, `"NAN"`, `"Inf"`, `"nano"`, `"Nancy"`, `"naan"` | ✅ preserved | ✅ preserved |

**8 of 15** test strings are destroyed by the sync path and **0 of 15** by the
async path. The async serializer simply has no equivalent handler in its chain.
Control: `"Nancy"`/`"naan"`/`"nano"` survive, confirming the match is exact
rather than substring — so the probe distinguishes the two behaviours.

These are not contrived values. `NaT` and `<NA>` are what pandas renders for
missing values, so any dataset round-tripped through a text export arrives with
them as **literal strings**, and `"nan"` is a common CSV null token. A
categorical column carrying them loses those categories entirely, and the loss
is indistinguishable from a genuine missing value downstream.

**Currently latent — the module has no production callers.** `backend.main`
imports 132 backend modules; `data_ingestion/serialization.py` is **not one of
them**, and no file under `backend/` imports it (the only textual references are
two comments in `config/mixins/files.py`). Its 603 lines are exercised solely by
three test files.

That combination is the reason this is filed as Medium rather than Low: the
module is **dead in production but alive in coverage**, so the test suite reports
confidence in 603 lines nobody runs, and the bug sits ready for the first caller
that wires the sync serializer up — the natural choice from a synchronous
context. Same shape as [OC-143](./17-file-coverage.md#oc-143).

**Suggested fix.** Delete the handler (the async chain proves it is unnecessary),
or restrict it to non-`str` inputs so it only catches `float('nan')`-style
objects it was presumably meant for. Then decide whether the sync serializer
should exist at all.

<a id="oc-159"></a>
### OC-159 — ⚪ Low — `delete()`/`update()` compile to WHERE-less statements on an empty filter

**File:** `backend/database/data_sources/async_sqlite_queries.py:129-146`
(byte-equivalent logic in `async_postgres_queries.py`)

Both build their WHERE clause by iterating the filter dict:

```python
stmt = delete(tbl)
for k, v in filter_dict.items():
    stmt = stmt.where(column(k) == v)
```

An empty dict produces **no iterations and therefore no WHERE clause**.
`_normalize_filter` in the CRUD layer maps `None` to `{}` without a guard, so an
empty or omitted filter is silently interpreted as "every row".

**Reproduced** by compiling the statements:

| filter | compiled SQL |
|---|---|
| `{"id": 7}` | `DELETE FROM data_sources WHERE id = :id_1` |
| `{}` | **`DELETE FROM data_sources`** |
| `{"id": 7}` | `UPDATE data_sources SET name=:name WHERE id = :id_1` |
| `{}` | **`UPDATE data_sources SET name=:name`** |

**Low because the call path is dead, not because the statement is harmless.**
The module *is* imported (via `database/data_sources/__init__.py`, so it loads
with the app), but `delete()` and `update()` have **no production call sites** —
only `tests/unit/test_async_data_sources_crud_extra.py`. The blast radius is
therefore currently zero, and the finding is a landmine rather than a live
defect. Same category as [OC-152](#oc-152).

**Suggested fix.** Raise on an empty filter in `_normalize_filter`. A destructive
helper should require an explicit `delete_all=True` to operate table-wide, never
infer it from an absent argument.

## Round 5 — checked and found sound

* **No dynamic SQL anywhere in `database/` or `data_ingestion/`.** Every
  `text()` call is a static literal (`SELECT 1`, `SELECT last_insert_rowid()`).
  Identifiers go through SQLAlchemy `column()`/`table()`, which quote them.
* **Session lifecycle** in `adapter.py:123-127` and `engine.py:134-138` pairs
  `rollback()` with `close()` in `except`/`finally`. Correct.
* **`eda/`** (`router.py`, `tasks.py`) and **`data/catalog.py`** were already
  covered by earlier rounds; re-checked, nothing new.
* **`repository.py:268` `get_by_file_hash` fails open to `None`** — a query error
  is indistinguishable from "no such file", which for a dedup lookup means a
  duplicate row. Noted rather than filed: it has no production callers either,
  and it does log a warning. Belongs to the recurring fail-open family
  (OC-113 VIF→1.0, OC-131 PSI→0.0, OC-145 CV→`{}`, OC-150 S3 redaction).

### Method note — why reachability was measured, not grepped

Grepping for call sites suggested `async_data_sources_crud` was dead. Importing
`backend.main` and reading `sys.modules` showed it **is** loaded, via the
package `__init__` re-export — dead *call path*, live module. The same check
also prevented filing `database/connections/postgres/` as dead: it is absent
from the import-time set only because it is **config-gated** behind a
PostgreSQL primary, not because it is unused. Both distinctions changed how the
findings above are worded.
