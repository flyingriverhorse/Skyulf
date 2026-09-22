# MLflow, Batch ve Son Teslimatlar Implementation Plan

> **For agentic workers:** Use `executing-plans`. Read
> [ARCHITECTURE.md](ARCHITECTURE.md) and [OPEN_QUEUE.md](OPEN_QUEUE.md).
> SM-18–SM-20 start only after the core and platform gates.

**Goal:** Opsiyonel tracking/registry, aylık tablo çıktısı ve son aşamada ürünleştirme.
**Architecture:** Core sözleşmeleri platform I/O'dan ayrılır. MLflow pyfunc,
SM-08 bundle'ını yükler; Databricks runner aynı inference girişini kullanır.
**Tech Stack:** MLflow, Unity Catalog, Spark/Delta, Databricks Jobs; en son DAB.
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

- [ ] Carry-forward from SM-13: run the G2 Spark runner on the same bundle
  downloaded through a concrete registry URI. If an MLflow Spark UDF is added,
  reuse the existing prediction contract rather than duplicating FE.
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

## SM-16 — Gerçek Databricks doğrulaması

**Bağımlılık:** SM-15. **Oluştur:** `skyulf-core/examples/databricks_batch_smoke.py`,
`initiatives/spark_and_mlflow/PLATFORM_VALIDATION.md` (uygulama sırasında).
**Üretir:** Bir wheel ile gerçek FE → registry load → Spark inference → test Delta table.

- [ ] Implement and verify shared publish admission for the selected runtime.
  SM-15's local file locks are rejected on distributed masters. The provider
  must hold exclusive ownership through commit and receipt verification;
  expiring leases require a new sink fencing design before they can be used.
  Include actual UC write-permission and concurrent-publisher evidence.
- [ ] Kullanılan DBR/Python/Spark/MLflow/Arrow, compute türü, izinli catalog/schema
  ve test resource isimlerini kaydet. Workspace bilgisi yoksa yalnız bu görev WAIT
  olur; local testler Databricks geçti sayılmaz.
- [ ] Wheel'i build et; temiz job ortamında kur. DBR'ın Spark'ını körlemesine
  pip extra ile değiştirme. OSS Spark, DBR ve Connect ayrı dependency profilleri.
- [ ] Native FE ve Python pipeline yollarını aynı küçük gold dataset ile
  karşılaştır; büyük sentetik dataset'te bellek/runtime ölç. UC alias pinning
  ve monthly retry testini test catalog'unda çalıştır.
- [ ] İlk doğrulama job/script ile; DAB template üretme. Serverless/Connect ancak
  ayrıca test edilirse supported ilan edilir; classic başarılarını genelleme.

```python
def test_platform_evidence_has_concrete_versions(platform_evidence):
    """A successful job without runtime and model identity is not reproducible."""
    assert platform_evidence["model_version"]
    assert platform_evidence["wheel_version"]
    assert platform_evidence["input_snapshot"]
    assert platform_evidence["job_run_id"]
    assert platform_evidence["prediction_parity_passed"] is True
```

Bu kontrol gerçek job çıktısına uygulanır; elle doldurulmuş dict platform testi değildir.
**Kabul:** G4 platform kapısı: run URL/id, sürümler, parity ve tekrar yazma kanıtı.

## SM-17 — Kalan node aileleri ve context desteği

**Bağımlılık:** SM-16. **Kaynak:** rapordaki aile matrisi;
`skyulf/preprocessing/{encoding,feature_selection,vectorization,transformations}`
ve diğer ailelerin mevcut modülleri; capability tablosu.
**Oluştur:** `NODE_SUPPORT.md`, `tests/spark/test_node_support_matrix.py`;
her kabul edilen aile için `tests/spark/test_<family>_parity.py`.

- [ ] İlk genişletme: column/cast/date/arithmetic, minmax/maxabs ve selection apply.
  Fit local-only olsa da seçilmiş kolonları Spark apply destekleyebilir.
- [ ] İkinci genişletme: median/quantile ve kategorik encode. Exact/approximate
  algoritma ayrı config; category order, unseen/null, büyük state tablo lookup
  ve hash algoritması korunmadan native etiketi verme.
- [ ] Üçüncü: group fitted lookup, rolling/lag ve custom FE. Group/order/tie-break,
  history cutoff, geçmiş veri erişimi ve partition boundary testleri zorunlu.
- [ ] Dördüncü: OOF target/WOE. Her fold'da train-only fit ve out-of-fold train
  representation testleri. Dataset split/hash kuralı engine değişince değişmesin.
- [ ] SMOTE, yüksek boyutlu vectorizer/embedding, SHAP/CV/tuning ve native model
  training için native/explicit bounded local/unsupported kararlarını kaydet.
  Bağımsız algoritma portları aile bazlı ayrı testli alt görevler olur; SM-17
  bütün node'ların desteklendiği iddiasıyla kapatılamaz.

```python
def test_declared_support_has_evidence(support_matrix):
    """An advertised engine-operation pair must point to a passing parity case."""
    for entry in support_matrix:
        if entry["status"] == "supported":
            assert entry["fit_test"] or entry["operation"] == "apply"
            assert entry["apply_test"] or entry["operation"] == "fit"
            assert entry["runtime_evidence"]
```

`support_matrix` fixture registry/capability anahtarlarını NODE_SUPPORT ile
eşleştirir; eski tarihli test yolu tek başına evidence değildir. Her alt aile
uygulanmadan exact dosya/test/API dilimine ayrılır; unsupported aileler kullanıcıya
açıkça gösterilir. İlk release için tüm ailelerin native olması şart değildir.
**Kabul:** G5: ilan edilen support matrix gerçek testlerle tutarlı, diğerleri açık ret.

## SM-18 — Backend/Canvas ve custom feature kullanımı

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

**Bağımlılık:** SM-18. **Oluştur:**
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

**Bağımlılık:** SM-19. **Oluştur:** repo kökünde `templates/databricks/` ve
`tests/templates/test_databricks_template.py`; template dosyaları
`databricks.yml`, `resources/batch_job.yml`, `README.md`, `src/run_batch.py`.

- [ ] Sorular: platform, FE engine, tracking, registry, batch/HTTP/SQL; birbirinden
  bağımsız ve capability validation ile. Local + MLflow/UC geçerli; Databricks +
  pandas/Polars küçük veri yolu geçerli. Unsupported kombinasyon açıklanır.
- [ ] Generated run_batch yalnız test edilmiş runner'ı çağırır; FE config ve
  custom feature module açık düzenleme noktalarıdır. İş mantığı kopyalanmaz.
- [ ] Aylık schedule timezone, period/as_of parametreleri ve dev hedefi üret;
  secret value üretme. Schedule ile backfill data period'ünü karıştırma.
- [ ] İki fixture üret: local+Polars+tracking off ve Databricks+Spark+MLflow/UC.
  Config syntax, import smoke ve bundle validate çalıştır; kullanıcıya ait ortamda
  deploy/run ayrıca gerçek smoke ve yetkilendirme sınırıyla yapılır.

**Örnek assertion:** Generate → config load → runner preflight; engine seçimi
manifestte korunur, endpoint seçilmediyse endpoint resource üretilmez.
**Kabul:** G8: template çalışan core/runner'ları paketler; endpoint/batch seçimi
isteğe bağlı; gerçek deployment kanıtı config validation'dan ayrı yazılır.
