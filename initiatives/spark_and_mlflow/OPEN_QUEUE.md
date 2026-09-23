# Spark ve MLflow — Open Queue

Updated: 2026-09-23. **SM-00–SM-16 and SM-24a/25/26 complete; SM-15L next. Target: 0.9.0.**
First deliverable: pandas/Polars training and local batch on Databricks, then a
working local-engine Bundle. Spark expansion follows that Bundle. SM-15L is
reopened for monthly UC output; SM-18 and continuous streaming remain parked.

Read [HANDOFF.md](HANDOFF.md), [the integration plan](04-databricks-integration-plan.md)
and [the separate SM-20 Bundle plan](05-sm20-bundle-plan.md) before starting SM-15L.
Historical completion evidence is preserved in
[OPEN_QUEUE_tamamlanmakaydi.md](OPEN_QUEUE_tamamlanmakaydi.md).
SM-26 added a local MLflow package; no Bundle was generated or cloud job started.

Durumlar: READY = başlanabilir; WAIT = önceki görev bekleniyor;
LATER = son aşama; ACTIVE = yürütülüyor; BLOCKED = somut dış engel;
DONE = kanıtla tamamlandı. SM-00 commit: `105a6fe4`.
DEFERRED = user postponed this work; PARKED = do not implement until resumed.

| Sıra | Görev | Bağımlılık | Durum | Kısa bitiş ölçütü |
| --- | --- | --- | --- | --- |
| SM-00 | Baseline ve Spark test ortamı | — | DONE | Güncel master: 231 baseline; 2 Spark smoke; [kanıt](BASELINE.md) |
| SM-01 | Execution/capability kuralları | SM-00 | DONE | Immutable config; registry operation desteği; aşağıda kanıt |
| SM-02 | Spark engine ve conversion sınırı | SM-01 | DONE | Gerçek Spark adapter; conversion korumaları; aşağıda kanıt |
| SM-03 | Dispatcher, schema, row keys | SM-02 | DONE | Keyed Spark FE giriş/preflight; aşağıda kanıt |
| SM-04 | Versioned portable state | SM-03 | DONE | Tagged state, limit/version/schema kontrolü; aşağıda kanıt |
| SM-05 | Native SimpleImputer | SM-04 | DONE | Mean/constant train-only fit, Spark apply; aşağıda kanıt |
| SM-06 | Native StandardScaler | SM-05 | DONE | Population variance, dört flag ve numeric parity; aşağıda kanıt |
| SM-07 | Uçtan uca FE kapısı | SM-06 | DONE | Üç engine çapraz fit/apply; kayıt/yükleme; aşağıda kanıt |
| SM-08 | Ortak inference bundle | SM-07 | DONE | Raw/features ayrımı; model metadata round-trip; aşağıda kanıt |
| SM-09 | Native FE + worker model | SM-08 | DONE | Spark tahmini local ile key bazında aynı; aşağıda kanıt |
| SM-10 | Worker Python FE + model | SM-09 | DONE | Batch-safe portable FE; window gibi yollar açık ret |
| SM-11 | Classification ve ölçek kapısı | SM-10 | DONE | Class/proba/threshold parity; transport, packaging and scale evidence |
| SM-12 | Opsiyonel MLflow tracking | SM-11 | DONE | Off bağımsız; gerçek run lifecycle ve izolasyon |
| SM-13 | MLflow model packaging | SM-12 | DONE | Temiz ortamda pyfunc yükleme ve parity |
| SM-14 | Registry/Unity Catalog adapter | SM-13 | DONE | Explicit publish; alias/version pinning; local evidence |
| SM-15 | Monthly batch + Delta sink | SM-14 | DONE | 43 local Delta/contract/admission tests; evidence below |
| SM-16 | Gerçek Databricks kapısı | SM-15 | DONE | Selected serverless workflow passed; live evidence below |
| SM-15L | Local predictions -> UC Delta monthly sink | SM-24a | READY | Score only requested month; keep prior months; Spark only for bounded I/O; real UC replay/concurrency proof |
| SM-17 | Complete existing Spark node/model coverage | SM-20a | LATER | Resume only after the first local Bundle; preserve all inventoried gaps |
| SM-18 | Backend/Canvas and legacy artifact bridge | SM-17 | PARKED | Capability UI/API, DAG contracts and existing deployment compatibility |
| SM-26 | Local pandas/Polars pipeline MLflow packaging | SM-16, existing local persistence | DONE | Fitted FE/model/engine, schema and thresholds preserved; MLflow 3.16.1 isolated-process parity; Spark/HTTP scopes rejected |
| SM-25 | Local-first config / SDK preflight | SM-26 | DONE | Immutable local config; pinned artifact load, bounded frame and actionable preflight; no node/model allowlist |
| SM-24a | Local training and bounded batch prediction | SM-25 | DONE | Two UC source tables, five live cross-job models, 62-ID FE audit and bounded monthly reads; [evidence](08-sm24a-live-validation-report.md) |
| SM-20a | First local-engine Bundle/template | SM-24a, verified SM-15L | WAIT | Generate, validate, deploy and run pandas/Polars train/monthly batch with pinned MLflow/UC model |
| SM-24b | Optional Databricks Jobs API operations | SM-20a | LATER | Add dynamic submit/status/cancel only if Bundle jobs are insufficient |
| SM-24d | Hard transport budget for wide UC rows | SM-24a | LATER | Add a proven paged/size-limited source adapter when exact transfer-byte enforcement is required; current Spark iterator bounds accepted decoded rows and frame memory only |
| SM-22 | Optional model validation and controlled promotion | SM-20a, current evaluation/registry | LATER | Pinned candidate/champion comparison, explicit version promotion and rollback |
| SM-27 | Full-history local rescore and selectable Bundle mode | SM-20a, SM-15L | LATER | Score a pinned, bounded source snapshot into a new prediction generation; validate and activate explicitly; preserve prior generation |
| SM-28 | Monthly retraining and challenger workflow | SM-20a, SM-22 | LATER | Label-aware schedule, leakage-safe training, candidate/champion report, explicit promotion, and independent pinned scoring job |
| SM-19 | Optional live HTTP / SQL ai_query / endpoint operations | SM-20a, compatible pyfunc package | LATER | Add only after serving parity; streaming remains parked |
| SM-21 | Optional Databricks feature tables / online lookup | SM-20a; SM-19a for online serving | LATER | Point-in-time lookups and optional online freshness; declare any Spark dependency |
| SM-23 | Optional monitoring and inference observability | SM-20a, relevant batch/serving adapter | LATER | Existing Skyulf metrics + optional Databricks monitoring/inference tables |
| SM-24c | Spark batch workflow adapter | SM-20a, existing Spark sink | LATER | Expose tested Spark runner after first local Bundle |
| SM-20b | Spark Bundle enhancement | SM-24c, selected SM-17 slices | LATER | Add tested Spark engine choice while preserving local variant |

IDs remain stable; suffixes distinguish the first local Bundle from its later
Spark extension. SM-27 and SM-28 add the selectable full-rescore and scheduled
retraining workflows after the first Bundle; neither blocks SM-20a. Endpoint,
feature lookup and monitoring work also remain optional. Prefer completing the
local Bundle extensions before broad Spark expansion.

## Local-first path to the first Bundle

These tasks were identified in the reference review; the latest user direction
moves the first local Bundle before Spark and other optional integrations.
Full task scope, proposed code areas, reference sources and acceptance checks are
in [04-databricks-integration-plan.md](04-databricks-integration-plan.md).

1. **SM-26:** fitted pandas/Polars pipeline -> MLflow package -> same local predictions.
2. **SM-25:** config, pinned local artifact loading, limits and preflight.
3. **SM-24a:** local training and bounded local batch prediction in Databricks.
   Its [live validation report](08-sm24a-live-validation-report.md) records two
   source tables, five cross-job models and every preprocessing registration ID.
4. **SM-15L:** convert only final local predictions for guarded Spark Delta publication; each new month adds results while earlier months remain unchanged.
5. **SM-20a:** generate, validate, deploy and run the first local-engine Bundle using the [two-month rehearsal](05-sm20-bundle-plan.md).
6. **After SM-20a:** SM-27 adds explicit `full_rebuild` alongside the default
   `period_update`; SM-22 then SM-28 add candidate/champion validation and
   optional monthly retraining. These are separate from monthly scoring.
7. **Later:** optional SM-24b/24d/19/21/23 adapters; Spark SM-24c/SM-17, then SM-20b Bundle enhancement.

SM-26 is the explicit packaging task added after the local-first MLflow discussion.
It reuses the fitted local pipeline rather than requiring every FE node to gain a
portable Spark codec first. Each advertised local pipeline/model variant needs
save/load/prediction evidence. This does not make its artifact Spark-compatible.
Backend joblib dictionary bridging stays in parked SM-18. Local prediction
and UC Delta publication remain separate tasks; SM-15L is now on the first
Bundle's critical path, with an explicit small-data memory limit. Spark can
perform UC table I/O here without becoming the FE or model execution engine.

Pandas/Polars training -> Spark inference is available only for the currently
compatible FE/model bundle contract. Both Spark modes remain capability-gated.
Delaying SM-17 does not enable arbitrary local pipelines or all registered models.
SM-25 extracts the supported-workflow usability subset from later SM-17j; wider
native model/engine coverage waits until the first local Bundle passes.

## Later SM-17 scope and completion contract

The user requested coverage of the existing nodes and models, not closure by
listing the rest as unsupported. Missing combinations stay open unless the user
explicitly defers them. This expansion starts after SM-20a; none of its missing
features is marked DONE. No implicit collection or algorithm swap.

| Task | Status | Deliverable |
| --- | --- | --- |
| SM-17-00 | LATER | Expand the [100-ID inventory](NODE_SUPPORT.md) into per-option fit/apply/state/model/runtime coverage and a registry coverage guard |
| SM-17a | LATER | Column/cast/cleaning/date/math/interaction/polynomial and basic scaler native paths |
| SM-17b | LATER | Remaining imputation, categorical encoders, binning/ranges and robust/quantile state |
| SM-17c | LATER | Feature selection, transforms and outliers; fit/apply distinction and row alignment |
| SM-17d | LATER | Group/window/history and target/WOE OOF semantics; deterministic splits and leakage gates |
| SM-17e | LATER | Text/vectorization/embeddings and geo; sparse/vector contracts and worker dependencies |
| SM-17f | LATER | Split/resampling/inspection; training-only row changes, stable membership and bounded previews |
| SM-17g | LATER | Existing model-family bundle/inference coverage: sklearn variants, XGBoost/LightGBM, ensembles/calibration and clustering |
| SM-17h | LATER | Explicit native Spark model training/transform, artifact kind and MLflow persistence; all existing model IDs retain decisions/tasks |
| SM-17i | LATER | Evaluation/CV/tuning/thresholds and explainability/SHAP contracts; parallel trials versus distributed fit; bounded execution |
| SM-17j | LATER | Easier config/SDK entry point, preflight and independent platform/FE/model/sink/tracking choices; no Canvas or DAB work |
| SM-17k | LATER | Family regression gates and selected Databricks runtime validation; no missing requested coverage silently marked DONE |

Family tasks depend on SM-17-00 and their relevant preceding contracts. Model
work need not wait for every independent FE family, but no task should claim
support before its artifact, inference and runtime requirements pass.

Details, reference-repo comparison and parked endpoint tasks:
[Spark/Databricks gap review](reports/2026-09-22-spark-databricks-gap-review.md).

## Detay planlar

- Completed SM-16 evidence and scope: [PLATFORM_VALIDATION.md](PLATFORM_VALIDATION.md).
- SM-15 currently provides the Spark writer only. SM-15L is now required for
  SM-20a's local monthly UC output and must establish its own transaction and
  runtime evidence. The first template precedes Spark expansion.
- Latest user direction supersedes the prior Spark-first and template-last order:
  deliver a local pandas/Polars Bundle, then expand Spark and add a tested Spark
  option. SM-18 and continuous streaming remain parked.

- SM-00–SM-07: [Core/Spark](01-core-spark-plan.md)
- SM-08–SM-11: [Inference](02-inference-plan.md)
- SM-12–SM-20: [MLflow/batch/son teslimatlar](03-mlflow-batch-delivery-plan.md)
- Current integration lane: [Local-first integration](04-databricks-integration-plan.md)
  and [SM-20 Bundle rehearsal](05-sm20-bundle-plan.md).
- Ortak kurallar: [Mimari](ARCHITECTURE.md), [Doğrulama](VALIDATION.md)

SM-17 requires the full tracked scope above. Some algorithms may need an explicit
alternative backend rather than an equivalent native Spark implementation; that
decision must not hide missing support. SM-19 streaming remains optional and parked.

## SM-24a — 2026-09-23 local and Databricks validation, DONE

- Added explicit bounded pandas/Polars training and a pinned UC Delta monthly
  reader. Spark filters/projects/limits the source; saved local FE/model runs in
  one Python process. This stage returns predictions and diagnostics, not a
  Delta prediction table. The [local guide](../../docs/user_guide/databricks_local_sdk.md)
  includes code and the decoded-byte boundary.
- Five pandas/Polars model packages trained and registered in isolated schema
  `workspace.skyulf_sm24a_20260923`; an independent serverless scoring job
  matched both 80-row months by key. Replay and negative cases passed. All 30
  additional inference-eligible preprocessing registrations passed on both
  engines in a separate live audit, making 41 live-evidenced eligible IDs in
  the [62-ID matrix](07-sm24a-node-matrix.csv). Train-only, inspection,
  optional and unsupported IDs remain explicit.
- An unknown registry version failed preflight. A disposable alias moved to
  R1 version 2 after preparation; the prepared predictor remained on version 1
  and returned the expected 80 rows. The alias was deleted. The full
  [report](08-sm24a-live-validation-report.md) lists runs, versions, resources
  and test results.
- A follow-up [scaler and outlier audit](09-sm24a-scaler-outlier-audit.md)
  registered 10 pandas/Polars MLflow models with held-out metrics and passed
  a separate two-month scoring replay. Four filtering outlier nodes reject
  shortened prediction batches; Winsorize clips and preserves rows.
- A [200-row held-out metrics audit](10-sm24a-heldout-metrics-report.md)
  logged regression MAE/RMSE/R2 and classification accuracy/F1 in the same
  MLflow runs as five new test models. All five registered versions passed a
  separate two-month scoring replay. R2/R3 had negative held-out R2; those
  example models are not approved for production promotion.
- The Spark iterator does not expose actual wire-byte counts. `max_bytes`
  bounds accepted decoded rows and local-frame memory; a very wide row can
  arrive before rejection. [SM-24d](OPEN_QUEUE.md) retains the stronger
  transport-budget option for wider or larger workloads. Next: SM-15L.

## SM-25 — 2026-09-23 local SDK validation, DONE

- Added frozen `LocalWorkflowConfig` with explicit runtime, pandas/Polars engine,
  bounded caller frame or declared future UC source, local/portable artifact
  selection, optional MLflow store URIs and return/declared future Delta sink.
  Config parsing performs no remote call and rejects embedded URI credentials.
- Local preflight reports actionable config, source, sink, artifact, engine,
  node-contract and model-contract issues. Optional read-only registry preparation
  resolves an alias once, loads the concrete version, and checks package digest.
  The SDK has no separate FE/model allowlist; a fitted SkyulfPipeline controls
  local prediction. An optional representative-frame probe exercises that
  actual path before submission. Inference input is capped by rows and bytes.
- After the allowlist correction, **20 local SDK tests passed**, including
  pandas/Polars MinMaxScaler + random-forest replay. The combined isolated
  MLflow SDK, local-package and registry gate passed **39/39**.
  The broader Core unit run completed **3927 passed, 71 skipped** before that
  narrow correction; Ruff, repository Ty and strict MkDocs passed after it. No
  Databricks job, UC source adapter or Delta writer was added in SM-25.
- [Local-engine SDK guide](../../docs/user_guide/databricks_local_sdk.md).
  Next: SM-24a local training and bounded batch workflow.

## SM-26 — 2026-09-23 local packaging validation, DONE

- Added a versioned, trusted local pipeline artifact with recorded pandas/Polars
  fit engine, raw/model feature schemas, model class, exact dependency versions,
  payload checksum, classification classes and explicit tuned-threshold choice.
  Legacy pipeline pickles still load; those without a recorded engine require a
  refit before local MLflow packaging.
- Added MLflow pyfunc packaging and a synthetic input example/signature for the
  supported scalar inputs. Its pandas boundary converts back to the recorded
  Polars engine when needed. Whole-frame local is the only declared execution
  scope; HTTP row-local and Spark scopes are rejected. Existing explicit registry
  publication and pinned-version resolution now carry the local payload digest.
- Isolated MLflow 3.16.1 integration: **16 local artifact/pyfunc tests passed**
  for categorical encoding, binning, null input, classes/probabilities, tuned
  thresholds, cross-engine requests, and a fresh isolated Python subprocess
  load. The 0.9.0 wheel was built from the changed source and installed into the
  isolated environment; `python -I` resolved the new module from site-packages.
  Existing MLflow packaging and registry compatibility: **47 passed, 1 skipped**.
  Pipeline unit regression: **385 passed, 2 skipped**. Focused Ruff and ty checks
  passed. No Databricks job or Delta writer was added in this task.
- [MLflow model guide](../../docs/user_guide/mlflow_models.md) lists the tested
  model families and the wheel, serving and Spark limits. This task handed off
  to SM-25 and then SM-24a.

## SM-16 — 2026-09-22 local and live validation, DONE

- Baseline `d43e74ca`. Added a trusted pinned registry-bundle loader and a
  templates-free `databricks_batch_smoke.py` probe with pandas/Polars training,
  an independent numerical oracle and both existing Spark inference modes.
- Combined local registry/worker gate: **29 passed**, no skips, MLflow 3.16.1 /
  PySpark 4.0.3 / Python 3.12.3. Base regression: **77 passed, 28 optional skips**.
  Ruff/format, repository Ty and strict MkDocs passed. Wheel built locally;
  checksum, exact commands and remaining gates are in
  [PLATFORM_VALIDATION.md](PLATFORM_VALIDATION.md).
- Databricks CLI **1.17.0**, profile `skyulf`, approved isolated schema
  `workspace.skyulf_sm16_20260922`. Live regression parity passed both Spark
  modes in run `447606109645160`; restricted-principal model allow/deny and
  10k/50k synthetic parity passed in run `2921374308246`.
- Checkpoint `311547fc` adds serverless identifier handling and shared Delta
  admission. Local gates passed 75 inference tests and 18 admission/batch tests.
  The reusable monthly Delta probe passed 5 real Delta tests; live Delta/alias
  run `810044894558535` reached an unsupported `REFRESH TABLE` command.
  Narrow refresh/cache compatibility fixes passed 48 real Delta tests; corrected
  run `783094949884769` passed monthly replacement/replay, alias pinning and
  full worker package-content checks. `databricks.yml` remains absent.
- SM-16 is **DONE** for the selected serverless workflow: winner job
  `973709879452231` committed only after the contender verified held-owner
  rejection. Contender `404334394214907` then failed a test-only permission-code
  assertion; final restricted job `977447944071613` passed actual MODIFY denial,
  unchanged data/version and owner release. The failed run remains documented.
- SM-15L was deferred immediately after SM-16; the later local-first Bundle
  request reopens it after SM-24a. No implementation of that local sink is
  included in SM-16, and real UC access remains an acceptance requirement.

## SM-15 — 2026-09-22 validation record

- Baseline `4a613cb5`; branch `090`. The delivery commit containing this record
  adds `skyulf.integrations.databricks`, three batch test modules and a real
  Delta fixture, the optional Delta dependency profile/CI lane, and the English
  [monthly batch guide](../../docs/user_guide/databricks_batch.md).
- `run_batch` pins a concrete Delta source version and verifies snapshot
  availability at the explicit cutoff. Both existing Spark inference modes
  feed a precreated Delta target. A committed manifest records request identity,
  model digest, installed code version and counts. Source history newer than
  `as_of`, mismatched bundle digest/runtime and ambiguous contracts fail.
- Atomic `replaceWhere` preserves other periods. Empty deletion requires opt-in.
  Run receipts prevent duplicate publication and prevent an old retry undoing
  a newer recomputation. Changed requests under one run ID and stale target
  versions fail. Local OS admission locks are tested across processes; both
  runner and public sink reject local locks on distributed masters.
- Final required Linux/WSL lane: **43 passed**, including **18 real Delta
  integration cases** and **25 contract/admission cases**, no skips, 111.95s.
  Python **3.12.3**, PySpark **4.0.3**, Delta Python/JVM **4.0.0**, Java 17;
  pandas **2.3.3**, Polars **1.44.2**, Arrow **25.0.1**, sklearn **1.9.1**.
  Initial exploratory tests used newer Python-package patch versions; the final
  gate was rerun after aligning the environment with `requirements-delta.txt`.
- Cases include a pinned old source after a newer commit, a non-UTC Spark
  session, exact period boundaries, null/wrong metadata/count rejection with
  target version unchanged, stale/new/old retries and committed empty deletion.
  Admission permission denial uses fault injection; it is not live UC evidence.
- Base regression: **105 passed, 26 skipped** with MLflow/Delta absent, plus
  existing sklearn interchange and Windows physical-core warnings. Final narrow
  Windows contract/admission run: **25 passed**. Ruff check/format, repository
  Ty and strict MkDocs passed; the guide's Mermaid parsed with the real parser.
  Repository pre-commit hooks passed, including the synchronized optional
  dependency lock. GitHub Actions was added but has not run remotely.
- Windows Hadoop filesystem support could not run real Delta I/O, so the
  required transaction gate used an isolated Linux/WSL environment. No runtime
  authentication or Databricks job was performed. Preserve pre-existing temp
  folders and editor-created Databricks configuration outside this commit.
- Remaining scope: the cutoff proves snapshot availability only, not upstream
  point-in-time feature joins. Model registry identity is bound by the caller;
  bundle digest is checked. Receipt/history retention bounds the retry window.
  A shared local lock directory protects cooperative local writers on one host;
  distributed admission requires a separate implementation with ownership held
  throughout the commit. Expiring leases without sink fencing are unsupported.
  These platform concerns and the carried registry-to-Spark gate belong to SM-16.

Reproducible commands (Linux with Java 17 and an isolated environment):

```bash
uv venv .venv-delta
uv pip install --python .venv-delta/bin/python -r requirements-delta.txt
SKYULF_REQUIRE_DELTA=1 .venv-delta/bin/python -m pytest \
  skyulf-core/tests/integrations/test_batch_contract.py \
  skyulf-core/tests/integrations/test_batch_admission.py \
  skyulf-core/tests/integrations/test_delta_publish.py -q -o addopts=
```

Exact local invocation used the prepared WSL environment and cached official jars:

```powershell
wsl -d Ubuntu -- bash .cache/sm15-linux-run.sh -m pytest skyulf-core/tests/integrations/test_batch_contract.py skyulf-core/tests/integrations/test_batch_admission.py skyulf-core/tests/integrations/test_delta_publish.py -q -p no:cacheprovider -o addopts= --basetemp /tmp/sm15-delta-pinned-final --tb=short
.venv/Scripts/python.exe -m pytest skyulf-core/tests/spark/test_inference_bundle.py skyulf-core/tests/unit/test_pipeline_inference_schema.py skyulf-core/tests/integrations -q -o addopts= --basetemp .cache/sm15-base-final --tb=short
.venv/Scripts/ty.exe check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py
.venv/Scripts/python.exe -m mkdocs build --strict --site-dir .cache/sm15-docs
```

The local shell wrapper exports Java 17, its Python worker executable is selected
by the fixture, and `SKYULF_DELTA_JARS` points to the cached Delta 4.0.0 jars.
The portable command above downloads those jars through Delta's normal helper.

## SM-13 — 2026-09-22 validation record

- Added optional MLflow `pyfunc` packaging in
  `skyulf/integrations/mlflow/model.py`. `log_model` serializes the existing
  `InferenceBundle`, records a manifest-derived named signature and synthetic
  input example, and uploads the model directory to the explicitly supplied
  run ID through `MlflowClient`. `tracking_uri` can be passed when the run was
  created by a client-bound tracker; no global active run is used to select the
  destination.
- Raw and features bundles preserve local prediction results after MLflow
  save/load for pandas and Polars-trained pipelines. Classification tests cover
  class order, probabilities and tuned thresholds. Positional NumPy/list input
  is rejected. Dtypes without an exact MLflow column representation are rejected
  during signature creation rather than silently widened.
- Baseline `67315253`; this delivery commit includes implementation, tests,
  English guide and validation records. Python 3.12.10 / MLflow 3.16.1:
  **31 passed** (23 model, 8 tracking) with the wheel subprocess gate enabled.
  Base bundle/schema/integration regression: **80 passed, 7 skipped** with
  existing numeric/protocol warnings. Base tracking/model alone: **4 passed,
  5 skipped**; MLflow is absent. Ruff, repository Ty/pre-commit and strict
  MkDocs passed. Only MLflow 3.16.1 was exercised.
- The isolated consumer environment reused installed dependencies and replaced
  editable Skyulf with the final 0.9.0 wheel. Python `-I` loaded the model from
  a temporary working directory, asserted imports from site-packages and
  matched producer predictions. This verifies independent package imports,
  not a fresh network dependency install.
- MLflow aligns named columns, ignores extras and safely casts compatible
  request types before bundle validation; direct `predict_local` still requires
  exact names/order/dtypes. This transport boundary has an explicit regression.
  Tests also exclude the producer's uv project and temporary source paths and
  preserve an unrelated active run on a different tracking store.
- Scope boundary: registry/Unity Catalog resolution, G2 runner evidence,
  Databricks jobs, Spark UDF/endpoint adapters and Delta batch publication stay
  in SM-14–SM-16. SM-14 is now the next task.

Reproducible verification (consumer Python must contain the final built wheel):

```powershell
$env:SKYULF_MLFLOW_WHEEL_PYTHON = (Resolve-Path .cache/sm13-clean-env/Scripts/python.exe).Path
.cache/sm12-mlflow-env/Scripts/python.exe -m pytest skyulf-core/tests/integrations/test_mlflow_tracking.py skyulf-core/tests/integrations/test_mlflow_model.py -q -o addopts= --basetemp .cache/sm13-delivery --tb=short
.venv/Scripts/python.exe -m pytest skyulf-core/tests/spark/test_inference_bundle.py skyulf-core/tests/unit/test_pipeline_inference_schema.py skyulf-core/tests/integrations -q -o addopts= --basetemp .cache/sm13-base-regression --tb=short
```

## SM-12 — 2026-09-22 validation record

- Added the optional `skyulf.integrations.mlflow` adapter, the `mlflow` package
  extra, `requirements-mlflow.txt`, and the English tracking guide. Importing
  the adapter does not import MLflow; disabled tracking is a no-op with no
  client construction or network access.
- `TrackingConfig` and `track_run` use an explicit `MlflowClient` and run ID.
  They do not mutate MLflow's process-global active run, so concurrent contexts
  remain isolated and a caller-owned run stays open. Explicit metrics, params,
  tags, and an opt-in config artifact/digest are supported.
- Successful contexts terminate `FINISHED`; body exceptions terminate `FAILED`.
  The default `raise` policy propagates tracking failures. `warn` preserves the
  body result and exposes `run.tracking_error`; an outer runner may propagate
  that value into its own result metadata.
- Base verification (`.venv`): **4 passed, 4 skipped** for the integration
  tests; MLflow is absent in that environment. Optional verification in an
  isolated MLflow 3.16.1 environment: **8 passed**. Ruff and ty checks passed
  for the new Python files.
- Scope boundary: this task adds run tracking only. MLflow model packaging,
  registry/Unity Catalog, Databricks batch delivery, endpoints, and templates
  remain SM-13 onward.

## SM-14 — 2026-09-22 validation record

- Added `skyulf/integrations/mlflow/registry.py` and
  `tests/integrations/test_mlflow_registry.py`. Publication accepts only the
  `runs:/...` URI produced by the packaging adapter, uses explicit tracking and
  registry clients, and leaves alias promotion to the caller.
- Resolution requires exactly one alias or version, pins an alias to a concrete
  `models:/name/version` URI, and returns the packaged signature and bundle
  digest. The version's recorded source URI is used for artifact metadata so
  separate tracking and registry stores work without process-global MLflow state.
- The isolated MLflow 3.16.1 lane passed **10 tests**. It covers local SQLite
  registration, alias movement, separate stores, missing model, access and
  dependency failures, non-run publication rejection, and Unity Catalog
  three-part-name validation. The base environment skips the optional module
  because MLflow is absent.
- Scope boundary: live Unity Catalog/Databricks validation and the G2 Spark
  runner remain SM-16 carry-forward work. SM-15 is now READY.

## Bir görevi kapatma kaydı

Her DONE satırına aynı dosyada aşağıdaki bilgileri içeren tarihli kayıt ekle:

```text
Task ID / tarih / incelenen commit / değişiklik commit'i veya uncommitted
Değişen dosyalar ve sağlanan davranış
Çalıştırılan exact komutlar / passed-failed-skipped / runtime sürümleri
Olumsuz senaryo kanıtı (unsupported, leakage, retry vb.)
Bilinen sınırlamalar / sonraki görev
```

Komutlar çalıştırılmadan kutular işaretlenmez. Spark lane'inde tamamı skip olmuş
suite DONE kanıtı değildir. Platform erişimi gerektiğinde BLOCKED nedeni somut
olarak yazılır; yerel test sonucu gerçek UC/Databricks sonucu yerine geçmez.

## Devam oturumu için kısa talimat

Önce README, mimari ve bu kuyruğu oku. Güncel git durumunu ve ilgili kaynakları
doğrula; kullanıcı değişikliklerini koru. İlk READY görevi ACTIVE yap, kendi
test döngüsüyle tamamla ve kanıtı yaz. Sonraki bağımlılığı aç. İlk local Bundle
kapısından önce Spark veya endpoint işine başlama. SM-00 yalnız test/runtime
hazırlığıydı; güncel destek ve sınırlar tamamlanan görevlerin kayıtlarında belirtilir.
