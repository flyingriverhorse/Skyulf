# MLflow, Batch ve Son Teslimatlar Implementation Plan

> **For agentic workers:** Use `executing-plans`. Read
> [ARCHITECTURE.md](ARCHITECTURE.md) and [OPEN_QUEUE.md](OPEN_QUEUE.md).
> The first local Bundle is SM-20a; Spark enhancement follows it.

**Goal:** Optional tracking/registry, local monthly UC output and a first local Bundle;
Spark becomes a later Bundle option.
**Architecture:** Core sözleşmeleri platform I/O'dan ayrılır. MLflow pyfunc,
SM-08 bundle'ını yükler; Databricks runner aynı inference girişini kullanır.
**Tech Stack:** MLflow, Unity Catalog, pandas/Polars, Databricks Jobs/Bundle;
Spark/Delta adapters extend the Bundle later.
**Spec:** [ARCHITECTURE.md](ARCHITECTURE.md).

## Global constraints

- Tracking ve registry kapalıyken core çalışır; implicit autolog/global callback yok.
- Tracking URI ve registry URI ayrı; local ortamda UC erişimi de mümkündür.
- Her run başlangıcında alias concrete version'a çözülür; batch boyunca değişmez.
- Kimlik doğrulama ortam/profile/secret provider üzerinden; artifact/config içine token yok.
- Endpoint ve template için önce çalışan runner; template iş mantığı içermez.
- Yeni source/test yolları `skyulf-core/` köküne göre; platform kaynakları repo köküne göre.

## SM-12 — Açık kapsamlı opsiyonel MLflow tracking

**Bağımlılık:** SM-11. **Oluştur:**
`skyulf/integrations/__init__.py`, `skyulf/integrations/mlflow/__init__.py`,
`skyulf/integrations/mlflow/tracking.py`, `tests/integrations/test_mlflow_tracking.py`.
**Değiştir:** `skyulf-core/setup.py` ve mevcut requirements eşleri (repo köküne göre).
**Üretir:** `TrackingConfig(enabled=False, tracking_uri=None, experiment_name=None,
failure_policy="raise")`; `track_run(config, *, run_name)` context manager.
Yield edilen run nesnesi `log_metrics`, `log_params`, `set_tags` sunar;
disabled durumda aynı arayüzle no-op ve network/import yok.

- [x] MLflow kurulu olmayan base ortamda import ve disabled eğitim testi yaz.
- [x] URI/experiment seçimi açık client/run_id kapsamına bağlı olsun; eşzamanlı
  iki çalışmanın run id'si karışmasın. Kullanıcının aktif run'ını sessizce kapatma.
- [x] Exception'da run FAILED; başarıda FINISHED. Varsayılan `raise`; opt-in
  `warn` durumunda model sonucu korunur ve tracking kaybı `run.tracking_error`
  üzerinde görünür; dış runner bunu kendi sonuç metadata'sına taşır.
- [x] Explicit param/metric log; params limitinde tam config artifact + digest;
  veri satırları ve secret içerebilen bütün config otomatik loglanmaz.

```python
def test_disabled_tracking_never_constructs_client(monkeypatch):
    """Local training must remain independent of the tracking service."""
    from skyulf.integrations.mlflow import tracking
    def forbidden(*args, **kwargs):
        raise AssertionError("MLflow client was constructed")
    monkeypatch.setattr(tracking, "_make_client", forbidden)
    with tracking.track_run(tracking.TrackingConfig(), run_name="offline") as run:
        run.log_metrics({"rmse": 0.5})
    assert run.enabled is False
```

`_make_client` bu görevde lazy client fabrikası olarak oluşturulur.
**Komut:** `python -m pytest skyulf-core/tests/integrations/test_mlflow_tracking.py -q`
**Kabul:** Base dependency isolation; local geçici tracking store'da gerçek run lifecycle.

**Validation record (2026-09-22):** `skyulf.integrations.mlflow.tracking` keeps
MLflow lazy and client-bound. Base tests passed with MLflow absent; the isolated
MLflow 3.16.1 environment passed all 8 tracking tests, including SQLite
FINISHED/FAILED lifecycle, concurrent run-ID isolation, caller active-run
preservation, explicit config digest/artifact, and warn-mode failure handling.

## SM-13 — MLflow model packaging and independent loading

**Dependency:** SM-12. **Files:** `skyulf/integrations/mlflow/model.py`,
`tests/integrations/test_mlflow_model.py`.
**API:** `log_model(bundle, *, run_id, artifact_path, tracking_uri=None) -> model_uri`;
`SkyulfPythonModel.load_context` loads the existing bundle and `predict`
delegates to SM-08 `predict_local` after MLflow signature enforcement.

- [x] Real pyfunc save/load preserves raw/features predictions for pandas and
  Polars-trained pipelines, class probabilities, thresholds, index and dtypes.
  Named columns reach the estimator in its recorded feature order.
- [x] Record the manifest-derived signature, a synthetic zero-like row, pinned
  requirements and wheel version. Never auto-capture production examples,
  notebook closures, Spark sessions or the producer's uv project.
- [x] Log through an explicit client/run ID, including when an unrelated fluent
  run is active on a different store; preserve that run and global URI.
- [x] Load from a wheel-installed interpreter with Python `-I`, assert import
  from site-packages, and compare predictions with the producer's bundle.

**Transport boundary:** MLflow's named signature may reorder columns, ignore
extras and perform safe casts before the adapter receives the frame. This is
explicitly tested and documented; direct `predict_local` still requires exact
schema/order. Positional arrays/lists and missing names fail. Bundle input
types without an exact MLflow representation (`int8`, `int16`, unsigned ints)
are rejected before contacting a store; bool/int32/int64/float32/float64 work.

**Validation (2026-09-22, Python 3.12.10, MLflow 3.16.1):** **31 passed** for
tracking and model tests with the wheel subprocess gate enabled (23 model,
8 tracking). Base bundle/schema/integration regression: **80 passed, 7 skipped**;
optional dependencies are absent there. Scoped Ruff, repository Ty/pre-commit
and strict MkDocs checks pass. Exact commands and evidence are in
[OPEN_QUEUE.md](OPEN_QUEUE.md#sm-13--2026-09-22-validation-record).
The isolated consumer environment reused installed dependencies and replaced
editable Skyulf with the final 0.9.0 wheel; this verifies independent imports,
not a fresh network dependency install. Only MLflow 3.16.1 was exercised.

**Acceptance:** G3a is complete for local pyfunc packaging. The registry-to-G2
item originally listed here requires SM-14 and is carried forward below;
real Unity Catalog/Databricks evidence stays in SM-16. No Spark UDF was added.

## SM-14 — Registry ve Unity Catalog adapter'ı

**Bağımlılık:** SM-13. **Oluştur:** `skyulf/integrations/mlflow/registry.py`,
`tests/integrations/test_mlflow_registry.py`.
**İncele:** `skyulf/core/model_registry.py`; in-memory object-ref arayüzünü remote
URI çözümleme ile eşdeğer sayma, eski metotların return tipi değişmesin.
**Üretir:** `resolve_model(name, *, alias=None, version=None) -> ResolvedModel`;
ResolvedModel(name, version, model_uri, signature, digest).
`register_model(model_uri, name)` explicit publish işlemidir; tahmin sırasında çağrılmaz.

- [x] Carry-forward from SM-13: the SM-16 preparation probe loads the bundle
  selected by a concrete registry version and exercises both G2 Spark modes
  against an independent gold oracle in local SQLite/Spark. Live UC remains SM-16.
- [x] Alias/version aynı anda verilirse ve ikisi de yoksa açık config hatası;
  missing model/permission failure ile dependency failure ayrı hatalar olsun.
- [x] Resolve an alias once and carry the concrete version URI through the job.
  Local registry tests are complete; the live UC integration fixture remains in SM-16.
- [x] Validate the UC three-part model name, signature and separate registry URI
  with a local client; live UC access remains in the SM-16 platform gate.
- [x] Alias değiştirme otomatik başarı yan etkisi olmasın; model kaydı ve
  promotion ayrı explicit operasyonlar olsun. Testten sonra yalnız test-owned
  kaynaklar temizlenir; var olan alias değiştirilmez.

```python
def test_registry_rejects_ambiguous_reference():
    """A run must have exactly one model selection rule before resolving it."""
    import pytest
    from skyulf.integrations.mlflow.registry import resolve_model
    with pytest.raises(ValueError, match="alias.*version"):
        resolve_model("catalog.schema.model", alias="champion", version="3")
```

**Komut:** `python -m pytest skyulf-core/tests/integrations/test_mlflow_registry.py -q`
**Kabul:** G3b: local registry gerçek test; UC smoke için SM-16 platform kanıtı gerekir.

**Validation record (2026-09-22):** `register_model` publishes only
`runs:/...` artifacts through an explicit MLflow client and does not assign an
alias. `resolve_model` requires exactly one alias or version, resolves aliases
once, downloads the version's recorded source URI, and returns the concrete
`models:/name/version` URI with the packaged signature and bundle digest. Local
SQLite tests cover missing-model, permission, dependency, separate
tracking/registry stores, alias movement, and Unity Catalog three-part-name
validation. MLflow 3.16.1 isolated lane: **10 passed**. Base environment:
registry module skipped because MLflow is absent. Live Unity Catalog and Spark
runner validation remain SM-16 carry-forward items.

## SM-15 — Monthly batch and Delta publication

**Dependency:** SM-14. **Files:**
`skyulf/integrations/databricks/{__init__,_contracts,admission,batch,delta}.py`,
`tests/integrations/{conftest,test_batch_contract,test_batch_admission,test_delta_publish}.py`.
**API:** `BatchSpec` pins aware period boundaries, `as_of`, keys, source version,
target name/expected version, model name/concrete version/digest, installed code
version, run ID and inference mode. `run_batch(spark, spec, *, source, bundle,
options, admission) -> BatchResult` consumes an existing Delta table by name.

- [x] Validate UTC half-open periods, explicit calendar timezone and cutoff;
  reject naive/nonexistent local times. Test a non-UTC Spark session and DST.
- [x] Read a concrete Delta `versionAsOf` and verify its commit is no later than
  `as_of`. Retain table ID/version, model digest, runtime version and counts in
  a committed manifest. Reject incorrect digests/runtime and unavailable history.
- [x] Use atomic `replaceWhere`, preserving other periods. Reject invalid output
  timestamps/metadata/counts before writing; empty deletion requires `allow_empty`.
- [x] Enforce explicit table-wide admission and expected target version. Replays
  return the original receipt without undoing newer intentional recomputation;
  changed requests under the same run ID fail. Return success only after receipt
  verification. Test real Delta transactions and cross-process OS lock contention.
- [x] Reject append/key MERGE. Require a concrete model version for every run,
  including backfills. Intentional replacement uses a new run ID and explicitly
  reviewed target version; no automatic conflict overwrite.

**Temporal boundary:** `as_of` proves snapshot availability, not correctness of
upstream historical feature joins. The runner rejects a snapshot committed after
the cutoff. The source producer remains responsible for point-in-time joins;
there is no generic timestamp-column filter claiming to reconstruct history.
Model name/version association is caller-provided; the supplied bundle's digest
is verified. MLflow/UC downloads remain separate adapter/platform work.

**Admission boundary:** `LocalTableLock` coordinates local Spark drivers on one
host using a common lock directory keyed by Delta table ID. It is rejected by
both public entry points on distributed masters. The provider protocol requires
non-expiring ownership through the commit; it has no sink fencing token support.
Distributed admission remains SM-16. Writers bypassing the authority are outside
the single-writer guarantee. Delta-detected concurrency failures still propagate.

**Acceptance:** G4 local Delta period isolation, retry and conflict evidence.
The fixture performs real Delta I/O; permission denial at the admission boundary
is a fault-injection test, not evidence of live UC permissions. Exact versions,
commands and results are recorded in [OPEN_QUEUE.md](OPEN_QUEUE.md).

## SM-16 - Real Databricks validation

**Dependency:** SM-15. **Status:** DONE on 2026-09-22 for the selected serverless
Spark Connect workflow. Evidence: [PLATFORM_VALIDATION.md](PLATFORM_VALIDATION.md)
and its linked machine-readable live report.

- [x] Implement shared, non-expiring Delta publish admission. Verify ownership
  through commit and receipt validation, actual target-write denial and two-job
  contention. The initial contender probe failed only on its permission-code
  assertion; its preceding contention checks and the final successful restricted
  probe jointly establish the result. Preserve that failed-run record.
- [x] Record workspace/profile, test namespace/resources, serverless environment,
  Python/Spark/MLflow/Arrow versions and concrete run IDs/URLs.
- [x] Install the built wheel in clean jobs without replacing Databricks Spark.
  Check all 225 installed worker package files against the r3 content digest;
  verify current source files also match the tested wheel.
- [x] Compare native FE and Python-pipeline predictions with independent gold
  values. Measure 10k/50k synthetic runtime and scoped memory observations.
  Notebook/client Python lifetime RSS is not cluster or inference-worker peak.
- [x] Verify UC version pinning after alias movement, monthly replacement/replay,
  empty protection and preservation of other periods using actual transactions.
- [x] Use one-time CLI jobs and scripts. No DAB template or root databricks.yml.

**Acceptance:** G4 passed for serverless Connect 4.2.0 / Python 3.12.3 /
MLflow 3.16.1, client environment 4.10. The aggregate evidence is generated from
actual job outputs after checking success states, expected results, matching
model/table/trial identities and the wheel checksum. This is not a production
benchmark or blanket certification of all engines, node families or runtimes.

## SM-15L — Local-engine Delta publication

**Dependency:** SM-24a. **Status:** WAIT; reopened by the later local-first Bundle request.
The SM-15 delivery covers only the Spark runner. A Delta destination must not
imply Spark inference; support for pandas/Polars publication needs its own sink.

- [ ] Design explicit local source/target references and snapshot provenance;
  do not pass filesystem paths into the Spark-only table-name contract.
- [ ] Prefer a bounded local-predictions -> Spark DataFrame bridge for UC Delta
  publication. Use explicit output schema and preserve pandas/Polars prediction
  values, row keys, labels and nulls. SQL Connector is an alternative, not required.
- [ ] Reuse guarded `publish_replace_period` logic where valid, but separate
  the local publication request from `BatchSpec`'s Spark inference modes.
- [ ] Match period, UTC/schema, empty-result, receipt, retry and concurrency
  guarantees with real UC transactions, including stale concurrent runs.
- [ ] Validate selected compute, storage, table protocol and permissions.
  A direct delta-rs path write does not establish UC managed-table access.
- [ ] Expose only tested engine/sink combinations before SM-20a generates them.

This is planned work, not an available `run_batch(engine="polars")` feature.

## SM-17 - Complete existing Spark node and model coverage

**Dependency:** SM-20a. **Status:** LATER, after the first local Bundle.
The scope below is retained for Spark enhancement. Current local integration
work uses fitted pandas/Polars packages; see [the current plan](04-databricks-integration-plan.md).
The user's 2026-09-22 direction supersedes the earlier limited-family acceptance:
cover existing nodes and model paths, track every remaining gap, and do not close
SM-17 merely because unsupported combinations are listed.

Starting artifacts:
- [NODE_SUPPORT.md](NODE_SUPPORT.md): 100 literal source registrations, including
  aliases and optional model dependencies; all assigned to an owning task.
- [Gap review](reports/2026-09-22-spark-databricks-gap-review.md): source findings,
  template-service comparison, training/inference distinctions and runtime constraints.
- [OPEN_QUEUE.md](OPEN_QUEUE.md): ordered SM-17-00 through SM-17k deliverables.

- [ ] SM-17-00: expand every registration to per-configuration fit/apply/codec,
  row/context and model-output contracts, with an inventory coverage guard.
- [ ] SM-17a-f: implement the complete FE/data families in the queue, retaining
  existing pandas/Polars behavior. Test exact versus approximate statistics,
  category order, missing/unseen values, sparse/vector output, partition boundaries,
  row membership, deterministic split and train-only/OOF state as applicable.
- [ ] SM-17g: extend explicit model serialization/prediction adapters to existing
  model families. The current sklearn-only, single-output, probability-capable
  regression/classification bundle is not a general model support guarantee.
- [ ] SM-17h: introduce explicit Spark-native distributed estimator training and
  transform, artifact/version metadata and MLflow persistence. Preserve the local
  estimator path; never silently substitute an algorithm or collect all training data.
- [ ] SM-17i: evaluation, CV/tuning, thresholds and explainability/SHAP for the
  declared model backends. Distinguish parallel trials from one estimator learning
  on distributed data; define explicit bounded/local or native explanation paths
  and fail clearly when unavailable. No implicit full-data collection for SHAP.
- [ ] SM-17j: config-driven SDK entry point and runtime preflight, with independent
  platform, FE/model engine, sink and optional tracking/registry selections.
- [ ] SM-17k: real per-family tests, artifact round trips, full pipeline gates and
  selected Databricks evidence. Supported rows require actual matching evidence.

Each implementation slice must identify exact APIs/files and tests before coding.
Native models may require a new artifact kind; maintain old bundle compatibility.
A Spark ML model and similarly named sklearn estimator are not automatically
numerically equivalent. Every source model ID needs an explicit disposition.
If equivalent native execution is infeasible, explain the limitation and keep
that work open for an explicit backend decision or user-approved deferral.

**Acceptance:** G5 closes when the requested tracked coverage is implemented and
verified, or any remaining exclusions have been explicitly agreed with the user.
A rejection is the correct current runtime behavior for missing capabilities,
but does not by itself satisfy the requested feature-completion task.

## SM-18 — Backend/Canvas ve custom feature kullanımı

**Status: PARKED by the user on 2026-09-22. Do not implement until resumed.**

**Bağımlılık:** SM-17. **Mevcut:**
`backend/ml_pipeline/_execution/engine/{__init__,_feature_eng,_artifacts,_node_runners}.py`,
`backend/deployment/service.py`,
`frontend/ml-canvas/src/core/registry/init.ts`,
`frontend/ml-canvas/src/core/utils/pipelineConverter.ts`.
**Oluştur:** `tests/integration/test_spark_pipeline_capabilities.py`;
ilgili frontend registry/config testleri.

- [ ] Core capability'yi API metadata'sına aktar; UI desteklenmeyen node/engine
  bağlantısını neden göstererek engellesin. Backend aynı kontrolü bağımsız yapsın.
- [ ] Core sequential pipeline ile Canvas DAG ayrı: branch merge key/cardinality,
  intermediate artifact policy ve executor lifecycle için entegrasyon testi yaz.
  Bütün ara Spark frame'lerini joblib'e koyma; lineage/ref ve bounded preview kullan.
- [ ] Legacy backend model+feature_engineer artifact adapter'ını threshold/label
  testleriyle ekle. Eski deployment davranışı değişmesin.
- [ ] Config-driven FE ve paketlenmiş custom module referansını destekle;
  generated config/editable Python ayrımı anlaşılır olsun. Arbitrary kod sandbox'ı
  veya kod editörü bu görevin gereği değildir.

**Örnek assertion:** API'ye unsupported Spark node gönderildiğinde 4xx +
node/operation/reason alınır ve job submission çağrısı yapılmaz; frontend aynı
node için disabled nedeni gösterir. Bunlar backend integration ve UI testlerinde
gerçek request/render ile doğrulanır.
**Kabul:** G6: SDK/Canvas supported yollar tutarlı; ilgili pytest, tsc/vitest/lint
ve `npm run build` kanıtı kayıtlı.

## SM-19 — En son inference erişim biçimleri

**Status: LATER - optional after the first local Bundle.**

**Bağımlılık:** SM-25 and the current compatible pyfunc contract; no SM-18 dependency. **Oluştur:**
`skyulf-core/examples/serving_prediction.py`,
`skyulf-core/examples/sql_batch_prediction.sql` (repo köküne göre).
**Mevcut:** `backend/deployment/service.py`, bundle/MLflow model adapter'ları.

- [ ] Önce HTTP serving: tek satır/microbatch schema, label/proba/threshold,
  local model ile parity ve dependencies ortamını doğrula. Serving'de SparkSession
  var sayma; context gerektiren model history olmadan kabul edilmez.
- [ ] Sonra SQL ai_query: uygun endpoint/signature ile tablodan çağrı; hata
  davranışı, maliyet/rate limit ve batch'e uygunluk hedef platformda test edilir.
  ai_query ayrı eğitim engine'i veya zorunlu yeni endpoint değildir.
- [ ] Streaming isteğe bağlı alt teslimat: checkpoint, watermark/state, retry,
  model version değişimi ve sink idempotency tasarım/testi tamamlanmadan advertise etme.

**Örnek assertion:** Aynı gold dataset için HTTP ve SQL sonucu, key + model version
üzerinden G2 batch sonucuyla eşleşir; missing feature 4xx/schema hatasıdır.
**Kabul:** G7: yalnız doğrulanmış erişim biçimleri supported; streaming ayrı durum.

## SM-20 — Template ve Databricks Bundle

**Status: WAIT - SM-20a builds the local Bundle before Spark; SM-20b adds Spark later.**

**Dependencies:** SM-20a needs SM-26, SM-25, SM-24a and verified SM-15L.
SM-20b needs selected validated Spark adapters; SM-19 only if endpoints are selected.
See [the dedicated SM-20 plan](05-sm20-bundle-plan.md) for generated resources
and the two-month UC source/output-table rehearsal.
**Planned files:** `skyulf-core/templates/databricks/` and
`tests/templates/test_databricks_template.py`; template dosyaları
`databricks.yml`, `resources/batch_job.yml`, `README.md`, `src/run_batch.py`.

- [ ] Ask for platform, FE engine, artifact store and output only when the
  combination has passed preflight. The first Databricks Bundle pins an
  MLflow/UC model version; a tracking-off or registry-off Bundle needs its own
  tested durable artifact handoff before it is offered.
- [ ] Generated run_batch yalnız test edilmiş runner'ı çağırır; FE config ve
  custom feature module açık düzenleme noktalarıdır. İş mantığı kopyalanmaz.
- [ ] Aylık schedule timezone, period/as_of parametreleri ve dev hedefi üret;
  secret value üretme. Schedule ile backfill data period'ünü karıştırma.
- [ ] İlk iki fixture: Databricks+pandas+MLflow/UC ve
  Databricks+Polars+MLflow/UC; ayrıca tracking-off yerel kullanım testi.
  Generated config/import ve bundle validate sonrasında ilk local Bundle için
  gerçek deploy/run, UC table parity ve replay kanıtı topla.
- [ ] SM-20b'de yalnız doğrulanmış Spark kombinasyonunu ikinci seçenek olarak
  ekle; local engine seçeneği korunur ve ayrı Spark bundle run testi yapılır.

**Örnek assertion:** Generate → config load → runner preflight; engine seçimi
manifestte korunur, endpoint seçilmediyse endpoint resource üretilmez.
**Kabul:** G8: template çalışan core/runner'ları paketler; endpoint/batch seçimi
isteğe bağlı; gerçek deployment kanıtı config validation'dan ayrı yazılır.

## Selected Databricks follow-ups discovered during the reference review

The user has selected these integrations for task definition before implementation.
Their current dependencies, sequence, proposed code areas and acceptance checks
are in [04-databricks-integration-plan.md](04-databricks-integration-plan.md).
SM-26 local pipeline packaging is READY; SM-25, SM-24a, SM-15L and SM-20a
follow. SM-24b is optional later. SM-18 and SM-19c remain PARKED. Spark SM-17/SM-24c/SM-20b
follow the first working local Bundle. This planning update starts no deployment.

### SM-19a/b/c/d - Endpoint detail

- [ ] SM-19a: live HTTP endpoint create/update/readiness and inference parity using
  a pinned compatible model package; dependency build, request schema, ACL and
  timeout/error tests. No implicit SparkSession inside serving.
- [ ] SM-19b: SQL ai_query against the same existing endpoint; named feature struct,
  return schema, CAN QUERY, row-level failures and capacity/rate handling. Preserve
  model/version identity; SQL invocation is not a separate estimator backend.
- [ ] SM-19c (PARKED): optional streaming with explicit checkpoint, history/state and model
  update contracts. Serverless continuous-trigger support must be checked separately.
- [ ] SM-19d: endpoint operations: scaling/cold start, A/B routing, rollback,
  bounded retry/backoff, inference logging and real HTTP/SQL/load tests.

### SM-21 - Feature tables and optional online lookup

- [ ] Optional Databricks Feature Engineering adapter, separate from core FE nodes.
- [ ] Explicit primary/timestamp keys, point-in-time joins and training-set lineage.
- [ ] Packaged lookup metadata and batch score semantics, with same-model parity.
- [ ] Optional online publication/freshness and identity-only serving inputs;
  absence/staleness behavior, privileges and offline/online parity tests.

### SM-22 - Validation and controlled promotion

Current order: SM-22a comparison and SM-22b promotion/rollback precede
SM-28a candidate retraining and the first local Bundle. SM-28b later wires
an optional schedule. See [the lifecycle plan](06-prebundle-model-lifecycle-plan.md).

- [ ] Candidate/champion evaluation reports using existing Skyulf metrics and
  model output contracts; configured minimum quality and regression thresholds.
- [ ] Explicit version-pinned promotion, rollback and audit metadata. Evaluating
  a model must not implicitly move production aliases or create a first champion.
- [ ] Reproducible validation dataset/split identity and promotion permission tests.
- [ ] Keep monthly retraining orchestration separate (SM-28): labels, temporal
  cutoff, candidate training and validation precede any explicit promotion;
  scoring resolves and pins the selected version once per run.

### SM-23 - Monitoring and inference observability

- [ ] Reuse current Skyulf quality/drift logic; optional Databricks monitoring
  adapters and inference-table integration rather than duplicated metrics.
- [ ] Delayed labels, classification/regression output schemas, model version,
  aggregation windows, freshness and observability coverage.
- [ ] Payload retention/redaction and monitoring/refresh permissions are explicit.

### SM-24 - Job operation helpers

- [ ] Validated period/backfill parameters, timezone, source snapshot, stable run
  identity and resource scope, independent of a generated DAB project.
- [ ] Bounded timeout/retry policies, publication receipts, orphan-claim recovery
  instructions and structured logs; no automatic takeover of a live owner.
- [ ] Optional job schedule/notification configuration and test/prod separation.
- [ ] Evaluate append/MERGE or other sink policies only as explicit separately
  tested extensions; current replace-period guarantees do not transfer implicitly.

### SM-25 - Supported-workflow SDK configuration and preflight

**Dependency:** SM-26. **Status:** WAIT; follows the local artifact contract.
See [the detailed integration plan](04-databricks-integration-plan.md).
This extracts only the usability subset for current compatible bundles from
parked SM-17j; it does not implement broader native node/model coverage.

- [ ] Independent runtime/engine/sink/tracking/registry configuration without secrets.
- [ ] Resolve bundle metadata and pinned model identity once; retain explicit
  source snapshot, period, target version and retry identity.
- [ ] Preflight unsupported FE/model/runtime combinations before submission.
- [ ] Short pandas/Polars training -> local Databricks batch examples first;
  Spark batch examples follow SM-20a.
- [ ] No remote mutations during validation; optional integration dependencies.

### SM-26 - Local pandas/Polars pipeline MLflow packaging

**Dependency:** SM-16 and existing standalone local persistence. **Status:** READY.
First in the revised local-first integration lane; see
[04-databricks-integration-plan.md](04-databricks-integration-plan.md) for code
areas, artifact boundaries and acceptance criteria.

- [ ] SM-26a: audit local fitted pipeline/engine/model semantics and define a
  versioned local artifact contract distinct from the portable Spark bundle.
- [ ] SM-26b: MLflow pyfunc package/load with no refit, explicit engine/input
  conversion, signatures, original feature order and required dependencies.
- [ ] SM-26c: original/save/load/clean-environment parity for pandas and Polars,
  including FE beyond the current portable subset; per-model eligibility evidence.
- [ ] Classify local whole-frame, row-local serving and distributed eligibility
  independently; unsupported serving/Spark behavior remains an explicit rejection.

Native Spark expansion follows the first local Bundle in SM-17; backend legacy
artifact bridging remains parked SM-18. Local UC Delta publication is SM-15L
on the critical path to SM-20a.
