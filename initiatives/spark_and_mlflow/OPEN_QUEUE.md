# Spark ve MLflow — Open Queue

Updated: 2026-09-24. **SM-00 through SM-16, SM-15L/15I/24a/25/26, SM-22a/b/c/28a/28b, SM-20a/20R/20S and SM-27/29 complete for their documented scopes. Target: 0.9.0.**
The pre-Spark local Bundle lifecycle gate passed live: selectable rescoring,
automatic champion selection, failed-score recovery and serialized handoff.
This is functional completion for the selected workflow. Production must
still enforce one exclusive alias writer; both test jobs used a human owner,
and live contention was exercised for score requests, not concurrent trains.
See [the combined live evidence](35-sm27-sm29-live-validation-report.md). The
current priority is the approved SM-30 through SM-43 local Bundle improvement
program below, starting with challenger semantics. Broad Spark expansion
waits for its local acceptance gate (SM-43a). SM-15L
provides explicit-period UC output. SM-15I adds automatic new-row scoring
for the first Bundle; SM-18 and streaming remain parked.

Read [HANDOFF.md](HANDOFF.md), [the integration plan](04-databricks-integration-plan.md)
and [the pre-Bundle lifecycle plan](06-prebundle-model-lifecycle-plan.md) before SM-20a.
The [SM-20 Bundle plan](05-sm20-bundle-plan.md) remains the packaging gate.
Historical completion evidence is preserved in
[OPEN_QUEUE_tamamlanmakaydi.md](OPEN_QUEUE_tamamlanmakaydi.md).
SM-26 added a local MLflow package; SM-15L and SM-15I validated UC writers.
The first local-engine Bundle passed generated-project validation and live jobs;
see [SM-20a evidence](21-sm20a-local-bundle-validation-report.md).

Durumlar: READY = başlanabilir; WAIT = önceki görev bekleniyor;
LATER = son aşama; ACTIVE = yürütülüyor; BLOCKED = somut dış engel;
DONE = kanıtla tamamlandı. SM-00 commit: `105a6fe4`.
DEFERRED = user postponed this work; PARKED = do not implement until resumed.
SUPERSEDED = replaced by a later user-directed scope; not a completed feature.

## Active priority: local Bundle improvements before Spark

The user approved the comparison findings and clarified that a newly trained
contender remains challenger even when it fails promotion. The user also
explicitly approved separating manual/automatic promotion from score model
selection. These are new requirements; prior DONE rows retain their original
scope and do not imply the improvements below already exist.

Read [the improvement program](37-local-bundle-improvement-program.md) and
[the independent selection/approval design](38-model-selection-and-approval-design.md).
Each task includes implementation locations and concrete acceptance checks.
**SM-30, company-compatible tags and SM-31 are committed; SM-32 is ACTIVE.**
Commit `fcfcee31` passed 116 tests and all applicable commit hooks.
The first SM-32 library policy slice passed 78 tests; see
[policy separation progress](43-sm32-policy-separation-progress.md).
The [manual approve library action](45-sm32-manual-approval-progress.md) now
rechecks saved candidate evidence without training; reject/rollback, contender
history and Bundle wiring remain open. [36 misplaced Core test files](44-core-test-relocation.md)
were relocated and verified before this continuation. See
[the implementation evidence](40-sm30-challenger-validation-report.md): 104
tests passed, including real MLflow lifecycles with pandas and Polars.
See [the clean live rehearsal](41-company-tags-and-clean-live-validation.md).
SM-31 passed 110 tests and its deployed thin score notebook passed live; see
[the extraction evidence](42-sm31-refactor-plan-and-evidence.md).
SM-43a remains the later combined acceptance for the full improvement program.
Keep two jobs and reuse Core services. Manual approval must act
on an existing candidate without retraining or reuploading the model.

| Order | Task | Dependencies | Status | Acceptance / deliverable |
| --- | --- | --- | --- | --- |
| SM-30 | Challenger nomination and visible evaluation status | Existing SM-22c/29 | DONE | 104 local tests and live run 607409241163605 passed; company tags, both engines, retained contender, replacement, error and rollback verified |
| SM-31 | Thin template and reusable workflow/publication services | SM-30 | DONE | 44-line notebook, two Core services, 110 tests, wheel installation/import, strict generation validation and live score run 383572337148835 passed |
| SM-32 | Independent score selection and promotion policy | SM-31 | ACTIVE | Library's four combinations and saved-evidence approve implemented; remaining: reject/rollback via one writer, score handoff without fit, previous_challenger history, matching Bundle choices and live proof |
| SM-33 | Validated config, migration and runtime parameters | SM-32 | WAIT | Per-run version/action/evidence inputs; task/column/key setup; migrate old mode and reject config/job-graph disagreement |
| SM-34 | Independent score/train schedules and training windows | SM-33 | WAIT | Editable paused score/train cron/timezone, explicit lookback/holdout/label windows; scoring needs no manual dates or retraining |
| SM-35 | Multi-metric quality gates and clear thresholds | SM-33 | WAIT | One selection metric plus optional guardrails; task/domain validation, first-model gate, failed-gate explanations and no probability-threshold confusion |
| SM-36 | Core tuning, model search and optional explainability | SM-35 | WAIT | Existing Core search/trials and FE reused with budgets, protected holdout, MLflow evidence and selected-model inference parity |
| SM-37 | Production identities and enforced writer ownership | SM-32, SM-33 | WAIT | Per-target run_as/permissions/hosts/roots; one lifecycle writer; scoring cannot move aliases; actual denial and lifecycle queue evidence |
| SM-38 | Operational limits, retry/recovery and run summaries | SM-34, SM-37 | WAIT | Configurable timeouts/retries/notifications; source/model/count/no-op summaries; score-only recovery and no blind alias retry |
| SM-39 | Reproducible packaging and per-target compute | SM-33 | WAIT | Central compatible wheel/runtime pins, clean install/load, configurable policy/worker/cost settings and strict target validation |
| SM-40 | Generated-project tests and generic CI/CD | SM-37, SM-38, SM-39 | WAIT | Standalone project tests/build/validate and optional company adapter; explicit deployment approvals; no default chargeable PR runs |
| SM-41 | CDF recovery, full refresh and generation retention | SM-31, SM-33, SM-38 | WAIT | Explicit CDF-expiry/source-change recovery; preserve active outputs/grants and rollback generations; update/delete policy never silently inferred |
| SM-42 | Complete first-run examples and operator guide | SM-30 through SM-41 | WAIT | pandas/Polars regression/classification examples; manual/auto and append/full instructions, config-vs-runtime settings, scheduling and recovery |
| SM-43a | Combined personal-workspace acceptance | SM-42 | WAIT | Representative real-data FE/models, both engines, manual/auto, retained challenger, new rows, no-op, failure/rollback/queue; exact run/resource evidence |
| SM-43b | Company-target production readiness gate | SM-43a; confirmed company environment | WAIT | Approved identities/UC/policy compute/dependency/CI/data checks in the actual company target; no inference from personal serverless tests |

The tasks are sequenced by this table when dependencies allow. SM-43b needs
actual company settings and access; it must not prevent unrelated local work.
No new live resources or company deployments are authorized merely by a queue
status. Existing explicit live authorizations retain their original scope.

## Approved extensions after the local improvement gate

The user explicitly requested serving, A/B, ai_query and online feature lookup.
These existing IDs are required backlog work, implemented after SM-43a in the
listed order, while each feature remains opt-in for generated projects.
See [the delivery contract](39-serving-and-feature-lookup-delivery-plan.md).

| Task | Status | Dependency / scope |
| --- | --- | --- |
| SM-19a | LATER | SM-43a; HTTP serving, fitted artifact parity, readiness and load/cold-start checks |
| SM-19b | LATER | SM-19a; SQL ai_query invocation, named inputs, privileges and failure behavior |
| SM-19d | LATER | SM-19a; A/B/canary routing, endpoint update/rollback; batch rollout separately explicit |
| SM-21a | LATER | SM-43a; UC feature lookup, keys and point-in-time correctness |
| SM-21b | LATER | SM-21a/19a; optional online publication, freshness and serving lookup |
| SM-23a | LATER | SM-43a; batch quality/drift/delayed-label reporting and optional dashboards; SM-38 covers basic operations first |
| SM-23b | LATER | SM-19a; endpoint inference tables and version-aware model-performance monitoring |
| SM-19c | PARKED | Continuous streaming remains outside the current user-approved implementation sequence |

SM-17/24c/20b remain later Spark enhancements after SM-43a. SM-18 stays parked.

## Earlier implementation and live evidence

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
| SM-15L | Explicit-period local predictions -> UC Delta | SM-24a | DONE | Score requested period; preserve rows outside it; Spark only for bounded I/O |
| SM-15I | Automatic incremental local scoring | SM-15L | DONE | No date/source-version input; 80+80 synthetic and 200+100 real taxi rows; no-op replay; [core evidence](13-sm15i-live-validation-report.md), [real-data proof](15-sm15i-real-nyctaxi-live-report.md) |
| SM-17 | Complete existing Spark node/model coverage | SM-20a | LATER | Resume only after the first local Bundle; preserve all inventoried gaps |
| SM-18 | Backend/Canvas and legacy artifact bridge | SM-17 | PARKED | Capability UI/API, DAG contracts and existing deployment compatibility |
| SM-26 | Local pandas/Polars pipeline MLflow packaging | SM-16, existing local persistence | DONE | Fitted FE/model/engine, schema and thresholds preserved; MLflow 3.16.1 isolated-process parity; Spark/HTTP scopes rejected |
| SM-25 | Local-first config / SDK preflight | SM-26 | DONE | Immutable local config; pinned artifact load, bounded frame and actionable preflight; no node/model allowlist |
| SM-24a | Local training and bounded batch prediction | SM-25 | DONE | Two UC source tables, five live cross-job models, 62-ID FE audit and bounded monthly reads; [evidence](08-sm24a-live-validation-report.md) |
| SM-22a | Pinned candidate/champion validation | Current evaluation/registry | DONE | Local comparison/report and isolated UC metrics gate passed; [local evidence](16-sm22a-local-validation-report.md), [live evidence](17-sm22a-live-metrics-report.md) |
| SM-22b | Explicit promotion and rollback | SM-22a | DONE | Version-checked alias changes, prior/new-version receipt, conflicts and permission tests; [live evidence](18-sm22b-live-validation-report.md) |
| SM-22c | Challenger and previous-champion aliases | SM-22b | DONE | Validated challenger staging; guarded three-alias promotion/rollback; legacy receipts and partial-write tests; [local and UC evidence](19-sm22c-lifecycle-alias-validation-report.md) |
| SM-28a | Label-aware retraining service | SM-22a, SM-22b, SM-22c, SM-24a | DONE | Pinned training snapshot, temporal holdout, candidate registration/comparison; no automatic promotion; [evidence](20-sm28a-label-aware-retraining-report.md) |
| SM-20a | First local-engine Bundle/template | SM-24a, SM-15I, SM-22b, SM-22c, SM-28a | DONE | Generated/deployed Polars Bundle; train/compare/stage/promote and 2+2 incremental score/no-op live; [evidence](21-sm20a-local-bundle-validation-report.md) |
| SM-20P | Company-shaped Bundle draft | SM-20a | SUPERSEDED | Replaced by the generic four-target SM-20R direction; no company workspace deployment |
| SM-20R | Reset personal tests; generic minimal Bundle | SM-20a | DONE | Ten test jobs and three schemas removed; four target shapes validated, personal dev deployed with three jobs; one source, one prediction, one internal control, one UC model; Polars 600+50-row scoring and no-op replay; [live evidence](24-sm20r-clean-generic-bundle-validation-report.md) |
| SM-20S | Two-job, table-minimal local Bundle | SM-20R | DONE | First score creates only prediction output; no admission tables; 650+1 rows, replay no-op, final two jobs/two tables; [live evidence](26-sm20s-two-job-live-validation-report.md) |
| SM-28b | Optional monthly retraining schedule | SM-20S, SM-28a | DONE | Two-job Bundle with optional paused train schedule, pinned monthly label window/version and explicit activation; [plan](27-sm28b-monthly-retraining-plan.md), [validation](28-sm28b-monthly-retraining-validation-report.md) |
| SM-24b | Optional Databricks Jobs API operations | SM-20a | LATER | Add dynamic submit/status/cancel only if Bundle jobs are insufficient |
| SM-24d | Hard transport budget for wide UC rows | SM-24a | LATER | Add a proven paged/size-limited source adapter when exact transfer-byte enforcement is required; current Spark iterator bounds accepted decoded rows and frame memory only |
| SM-27 | Full-history local rescore and selectable Bundle mode | SM-20a, SM-15L | DONE | Live append retained 160 v1 rows and added ten v2 rows; full view exposes 170 v2 rows, retains v1 and its SELECT grant; [plan](29-sm27-model-change-scoring-plan.md), [live evidence](35-sm27-sm29-live-validation-report.md) |
| SM-29 | Gated automatic champion and score handoff | SM-22a/b/c, SM-28a/b, SM-27 | DONE | Live first champion, v2 promotion, tied v3 rejection, two-job handoff, score-failure recovery, queue/no-op and restricted alias-write denial passed; [design](31-sm29-auto-champion-design.md), [live evidence](35-sm27-sm29-live-validation-report.md) |
| SM-19 | Optional live HTTP / SQL ai_query / endpoint operations | SM-20a, compatible pyfunc package | LATER | Add only after serving parity; streaming remains parked |
| SM-21 | Optional Databricks feature tables / online lookup | SM-20a; SM-19a for online serving | LATER | Point-in-time lookups and optional online freshness; declare any Spark dependency |
| SM-23 | Optional monitoring and inference observability | SM-20a, relevant batch/serving adapter | LATER | Existing Skyulf metrics + optional Databricks monitoring/inference tables |
| SM-24c | Spark batch workflow adapter | SM-20a, existing Spark sink | LATER | Expose tested Spark runner after first local Bundle |
| SM-20b | Spark Bundle enhancement | SM-24c, selected SM-17 slices | LATER | Add tested Spark engine choice while preserving local variant |

SM-28b added no second score writer; shared-target admission remains a gate
before any later scoring workflow writes the same prediction table.
SM-20R replaces SM-20P after the user's reset and generic-template direction.
Company profile/policy execution remains unverified. SM-22a/b/c and SM-28a
are prerequisites for the first Bundle. SM-28b only
wires their optional monthly schedule after the Bundle exists. SM-27 remains
post-Bundle and independent. Endpoint, feature lookup and monitoring remain
optional; broad Spark expansion follows the local Bundle.

## SM-27/SM-29 closure record - 2026-09-24

Implementation inspected: `68f64935`; approved test plan: `556b63f3`.
The selected Polars/serverless generated Bundle passed first-champion
initialization, improved v2 promotion, tied v3 rejection, both scoring
policies, failed-score recovery, queued no-op requests and a restricted
principal's real alias-write denial. Final rows: append 160 v1 + ten v2;
full view 170 v2; retained v1 generation unchanged. Exactly two persistent
test jobs and no admission table. The English
[live report](35-sm27-sm29-live-validation-report.md) records every run URL,
expected negative runs, exact commands, wheel hash, runtime configuration
and scope limits. Strict Bundle validation, wheel build, local pandas/Polars
precheck, live-evidence assertions and strict docs build passed. The prior
94-test implementation gate is preserved in the SM-29 local report.
No library fix was required by this rehearsal. Source updates/deletes,
company targets, policy compute and broad Spark execution remain outside
this evidence. Production must still enforce one serialized alias writer.

The subsequent [reference comparison and readiness review](36-bundle-reference-comparison-and-readiness.md)
separates this completed functional scope from remaining production adoption
work: manual approval UX, runtime parameters, independent score scheduling,
workflow simplification, identities/operations and generated-project CI.
The user subsequently approved that work; the new task definitions and the
corrected challenger semantics are recorded in SM-30 through SM-43 above.

## Local-first path to the first Bundle

These tasks were identified in the reference review; the latest user direction
moves the model lifecycle services before the first local Bundle. Spark and
other optional integrations follow the Bundle.
Full task scope, proposed code areas, reference sources and acceptance checks are
in [04-databricks-integration-plan.md](04-databricks-integration-plan.md).

1. **SM-26:** fitted pandas/Polars pipeline -> MLflow package -> same local predictions.
2. **SM-25:** config, pinned local artifact loading, limits and preflight.
3. **SM-24a:** local training and bounded local batch prediction in Databricks.
   Its [live validation report](08-sm24a-live-validation-report.md) records two
   source tables, five cross-job models and every preprocessing registration ID.
4. **SM-15L:** explicit-period local predictions reach guarded Delta. The live
   test used scripted period and source-version values; it is not automatic.
5. **SM-15I:** add automatic first-snapshot and later new-row scoring with a
   committed source-version receipt and append-safe output. See the
   [incremental plan](12-sm15i-incremental-scoring-plan.md).
6. **SM-22a/b/c:** compare concrete candidate/champion versions on one labeled
   holdout, stage a validated `@challenger`, then explicitly promote with
   `@previous_champion` and guarded rollback.
7. **SM-28a:** train a candidate from a pinned, label-aware snapshot and
   temporal holdout; compare it without automatic promotion.
8. **SM-20a:** generate, validate, deploy and run the first local-engine Bundle
   using the [two-run rehearsal](05-sm20-bundle-plan.md) without manual dates
   or source versions. Package the already-tested lifecycle services. **DONE:**
   [generated-project and live evidence](21-sm20a-local-bundle-validation-report.md).
9. **After SM-20a:** SM-28b wires optional monthly retraining; SM-27 adds
   explicit `full_rebuild` alongside default `incremental_append`. SM-15L's
   `period_update` remains an explicit backfill.
10. **Later:** optional SM-24b/24d/19/21/23 adapters; Spark SM-24c/SM-17, then SM-20b Bundle enhancement.

SM-26 is the explicit packaging task added after the local-first MLflow discussion.
It reuses the fitted local pipeline rather than requiring every FE node to gain a
portable Spark codec first. Each advertised local pipeline/model variant needs
save/load/prediction evidence. This does not make its artifact Spark-compatible.
Backend joblib dictionary bridging stays in parked SM-18. Local prediction
and UC Delta publication remain separate tasks; SM-15I is now on the first
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
- SM-15 and SM-15L provide Spark and explicit-period local UC writers.
  SM-15I is required for automatic new-row output in SM-20a. The first
  template precedes Spark expansion.
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

## SM-20a closure record - 2026-09-24

Inspected branch `090` at the SM-28a baseline and added the custom template,
generated-workflow tests, Databricks Bundle guide and isolated rehearsal example.
`databricks bundle init skyulf-core/templates/databricks --config-file ... --output-dir ...`,
`databricks bundle validate --strict -t dev --profile skyulf`,
`databricks bundle deploy -t dev --profile skyulf` and Bundle `run` commands for
train/compare/stage/promote/score all succeeded in the test workspace. The
score job passed 2 initial + 2 new rows, then returned a no-op with target
version unchanged. `python -m pytest tests/integration/test_sm20a_bundle_template.py`
passed four tests; Ruff lint/format and `mkdocs build --strict` passed. The
first local wheel upload and a test-only alias verifier failed, were diagnosed,
corrected and rerun; [the report](21-sm20a-local-bundle-validation-report.md)
retains those failure details and live run links. This Bundle uses Polars for
the live test, and remote forced-write failure was not repeated. SM-28b is next;
SM-27, broad Spark work and endpoints remain later.

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
