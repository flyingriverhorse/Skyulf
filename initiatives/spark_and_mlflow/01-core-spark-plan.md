# Core Spark Implementation Plan

> **For agentic workers:** Use the `executing-plans` skill to implement this
> plan task-by-task. Read the spec and queue first; do not start later tasks
> whose acceptance gates are incomplete.

**Goal:** Gerçek Spark fit/apply ve küçük state'in engine'ler arası taşınması.
**Architecture:** Mevcut Calculator/Applier ve engine-keyed dispatcher korunur.
Spark lazy/distributed işlemleri local dataframe sözleşmesinden ayrılır.
**Tech Stack:** Python >=3.12, pandas, Polars, PySpark SQL, pytest.
**Spec:** [ARCHITECTURE.md](ARCHITECTURE.md).

## Global constraints

- Örtük ham veri collect yok; sınırlı aggregate state collection açıkça ölçülür.
- Spark için unsupported node'lar işlem başlamadan hata verir.
- Mevcut local API ve varsayılan değişmez; yeni yollar opt-in olur.
- Spark session artifact içinde bulunmaz; key/target kolonları feature değildir.
- Yeni dependency kurulumları `uv pip`; mevcut manifest/requirements eşlenir.
- Commit yalnız kullanıcı isteğiyle; her görev sonu kanıt queue'ya işlenir.

Her görevde önce belirtilen davranış testi yazılır, fail nedeni incelenir, en dar
uygulama yapılır, aynı test ve ilgili eski local testler çalıştırılır. Örneklerde
yazılan yeni API isimleri bu planın teslim edeceği isimlerdir; bugün var sayılmaz.
Yeni test kökü: `skyulf-core/tests/spark/`; fixture dosyası `conftest.py`.

## SM-00 — Baseline ve test ortamı

**Bağımlılık:** Yok. **Mevcut:** `engines/registry.py`, `preprocessing/dispatcher.py`,
`skyulf-core/setup.py`, `skyulf-core/pyproject.toml`, mevcut requirements ve CI config.
**Oluştur:** `skyulf-core/tests/spark/conftest.py`,
`skyulf-core/tests/spark/test_runtime_smoke.py`,
`initiatives/spark_and_mlflow/BASELINE.md` (bu görev uygulanırken).
**Arayüz:** `spark` pytest fixture; local[2] session, UTC timezone;
SM-01–SM-11 bu fixture'ı tüketir.

- [x] HEAD, working tree, Python/Java, paket sürümleri, registry default ve
  mevcut fit/apply/save/load API'lerini BASELINE.md'ye kaydet. Tarihsel rapor
  sayılarını yeni ölçüm gibi kopyalama; eşzamanlı değişiklikleri ayır.
- [x] Local regression için mevcut engine/pipeline/artifact testlerini çalıştır.
  Önceden var olan failure'ları hata ve nedenleriyle kaydet.
- [x] Ayrı Spark test ortamı oluştur; Python >=3.12 ile uyumlu Spark/Java çiftini
  resmî sürüm dokümanıyla sabitle. Base dependency'ye PySpark ekleme; `spark`
  extra ve test dependencies'i aynı sürüm aralığına hizala.
- [x] Fixture session'ı teardown'da kapatsın. Base suite Spark yoksa skip edebilir;
  Spark release job'ı import başarısızlığında fail etsin, toplu skip başarı sayılmasın.
- [x] Aşağıdaki testi yeni ortamda çalıştır ve environment bilgileriyle kaydet.

```python
def test_spark_executes_multiple_partitions(spark):
    """Prove that the Spark test lane executes an action on partitioned data."""
    from pyspark.sql import functions as F
    frame = spark.range(100).repartition(2)
    total = frame.agg(F.sum("id").alias("total")).first()["total"]
    assert total == 4950
```

**Komut:** `.venv-spark/Scripts/python.exe -m pytest skyulf-core/tests/spark/test_runtime_smoke.py -q`
**Kabul:** Spark action gerçekten çalışır; local baseline ve runtime sürümleri kayıtlı.

## SM-01 — Execution ve capability sözleşmesi

**Bağımlılık:** SM-00. **Oluştur:** `skyulf/core/execution.py`,
`skyulf/core/capabilities.py`, `tests/unit/test_execution_capabilities.py`,
`tests/unit/test_execution_contract.py` (ilk iki yol
`skyulf-core/` altında; bu planda tüm `skyulf/` ve `tests/` yolları aynı köke göre).
**Değiştir:** `skyulf/registry.py`, `skyulf/config_validation.py`.
**Tüketir:** Mevcut node registration anahtarları ve operation metadata.
**Üretir:** ARCHITECTURE'daki FrameSpec, ExecutionOptions,
`require_capability(...)`; `UnsupportedExecutionError(ValueError)`.

- [x] Mevcut registry anahtarlarını dolaşan testte Spark desteği ilan edilmemiş
  bütün fit/apply kombinasyonlarının açıklayıcı hatayla reddedildiğini sabitle.
- [x] Capability kaydına engine + operation + config strategy + row/context etkisi
  ekle. Alias'lar canonical node ile aynı kararı vermeli; ikinci node registry yok.
- [x] FrameSpec duplicate key isimlerini/target-key çakışmasını; options sıfır/negatif
  limitleri ve bilinmeyen engine'i reddetsin. Frame içeriği kontrolü SM-03'te.
- [x] Yeni runtime config'de yazım hatasının sessizce kabul edilmediğini test et.
  Eski config davranışını topluca değiştirme.

```python
def test_unimplemented_spark_fit_is_explicit():
    """An unsupported fit must not fall back to collecting a local frame."""
    import pytest
    from skyulf.core.capabilities import require_capability, UnsupportedExecutionError
    with pytest.raises(UnsupportedExecutionError, match="spark"):
        require_capability("KNNImputer", "fit", "spark", config={})
```

**Komut:** `python -m pytest skyulf-core/tests/unit/test_execution_contract.py skyulf-core/tests/unit/test_execution_capabilities.py -q`
**Kabul:** Destek olmayan yol herhangi bir Spark action veya estimator fit çağırmaz.

## SM-02 — Spark engine ve materialization sınırı

**Bağımlılık:** SM-01. **Oluştur:** `skyulf/engines/spark_engine.py`,
`tests/spark/test_engine_contract.py`. **Değiştir:**
`skyulf/engines/{registry,protocol,__init__}.py`, `skyulf/engines/sklearn_bridge.py`.
**Üretir:** SparkEngine; `EngineName.SPARK`; açık distributed frame sözleşmesi.

- [x] Gerçek Spark frame'in `get_engine(frame)` sonucunu ve local default'un
  değişmediğini test et. Paket yokken core importu başarılı kalmalı.
- [x] Bilinen pyspark frame için engine kurulu/kayıtlı değilse erken dependency
  hatası ekle; bilinmeyen local objelerin mevcut fallback davranışını ayrı koru.
- [x] `len/shape/to_numpy/toPandas` gerektiren local protocol'ü Spark'a sahte
  implement etme. Local-only materialization metodları açıklayıcı hata versin;
  schema/columns/select gibi yetenekleri ayrı dar protocol üzerinden sağla.
- [x] SklearnBridge Spark veriyi reddetsin. Session yaratma, session singleton
  tutma veya RDD tabanlı wrapper ekleme.

```python
def test_spark_engine_does_not_replace_local_default(spark):
    """Input detection must leave unrelated local jobs on their existing engine."""
    from skyulf.engines import get_engine
    before = get_engine().name
    assert get_engine(spark.range(2)).name == "spark"
    assert get_engine().name == before
```

**Komut:** `python -m pytest skyulf-core/tests/spark/test_engine_contract.py -q`
**Kabul:** Import isolation, gerçek frame detection, yasak materialization ve
local engine regression testleri geçer.

## SM-03 — Dispatcher, schema ve key-aware FE girişi

**Bağımlılık:** SM-02. **Oluştur:** `skyulf/preprocessing/_spark.py`,
`tests/spark/test_frame_contract.py`. **Değiştir:**
`skyulf/preprocessing/{dispatcher,base,pipeline}.py`, `skyulf/core/schema.py`,
`skyulf/utils.py`, `skyulf/data/dataset.py` (yalnız ilgili input sınırı).
**Tüketir:** FrameSpec, ExecutionOptions, SparkEngine.
**Üretir:** FeatureEngineer'ın opt-in `frame_spec`/`execution_options` desteği;
ayrı X/y Spark frame'i almayan, tek frame'de target/key koruyan fit/transform yolu.

- [ ] Repartition edilmiş frame'de key/target'ın aynı satırda kaldığı testi yaz.
  Duplicate/null key, eksik kolon ve nested/Decimal unsupported hata testleri ekle.
- [ ] Spark input hazırlığını engine-keyed dispatcher mapping'e ekle; local
  tuple ve wrapper dönüşlerini değiştirme. Spark destek preflight bütün adımları
  kontrol etsin; ilk node çalışıp sonraki unsupported node'da geç kalınmasın.
- [ ] Default istatistik toplama Spark'ta count/NumPy conversion başlatmasın.
  İstenen distributed aggregate metrics ayrı seçilir; local RSS cluster belleği
  gibi raporlanmaz. Satır sayısı bilinmiyorsa unknown kalır.
- [ ] Feature listesi key/target'ı dışlasın. Kolon adlarında nokta/backtick için
  doğru quoting ve collision testleri yaz; uydurma internal key ekleme.

```python
def test_frame_spec_rejects_duplicate_key_names():
    """Ambiguous key declarations must fail before Spark starts work."""
    import pytest
    from skyulf.core.execution import FrameSpec
    with pytest.raises(ValueError, match="row_keys"):
        FrameSpec(row_keys=("event_id", "event_id"))
```

**Komut:** `python -m pytest skyulf-core/tests/spark/test_frame_contract.py -q`
**Kabul:** Identical keys/targets sonrası dönüşüm; preflight hatasında action yok;
local FeatureEngineer regression testleri geçer.

## SM-04 — Versioned küçük state codec'i

**Bağımlılık:** SM-03. **Oluştur:** `skyulf/core/portable_state.py`,
`tests/spark/test_portable_state.py`. **Değiştir:**
`skyulf/core/artifacts.py`, `skyulf/pipeline/seal.py` (gereken ortak canonicalization).
**Üretir:** `encode_state(node_type, params, *, max_bytes) -> bytes` ve
`decode_state(payload: bytes) -> tuple[str, dict]`; envelope v1.

- [ ] StandardScaler ve SimpleImputer mevcut artifact dict fixture'larını
  lossless round-trip test et; boş state/no-op davranışını ayrı kapsa.
- [ ] NaN/Infinity, integer/string kategori değerleri ve ordered_columns için
  tagged representation; semantic hash'te repr/pickle fallback olmaması testi yaz.
- [ ] Bozuk payload, bilinmeyen version/node, duplicate columns ve aşırı büyük
  payload'ı decode/worker yüklemesinden önce reddet.
- [ ] Eski pipeline pickle save/load testlerini değiştirmeden geçir. Bu codec
  generic Python object serializer değildir; estimator nesnesi kabul etmez.

```python
def test_state_rejects_future_version():
    """A newer artifact must not silently acquire older transformation semantics."""
    import pytest
    from skyulf.core.portable_state import decode_state
    with pytest.raises(ValueError, match="version"):
        decode_state(b'{"format_version":999}')
```

**Komut:** `python -m pytest skyulf-core/tests/spark/test_portable_state.py -q`
**Kabul:** Round-trip değer/kolon sırası; limit/version reddi; eski persistence korunur.

## SM-05 — SimpleImputer native Spark

**Bağımlılık:** SM-04. **Değiştir:** `skyulf/preprocessing/imputation/simple.py`,
`skyulf/preprocessing/_spark.py`, capability kayıtları.
**Oluştur:** `tests/spark/test_simple_imputer.py`.
**Tüketir/üretir:** Mevcut SimpleImputer artifact anahtarları; mean/constant
Spark fit ve apply. Median/most_frequent Spark fit bu görevde unsupported kalır.

- [ ] Local ve Spark mean/constant fixture'ları: train/test dağılımları farklı,
  null/NaN, tamamen boş kolon, boş seçim ve geçersiz tip; öğrenme yalnız train'den.
- [ ] Mean fit için seçilen kolonlarda bounded aggregate; constant için veri
  istatistiği gerekmiyorsa action yok. Missing counts gerekiyorsa aynı aggregate'e
  birleştir. Mevcut local boş kolon politikasını baseline'dan aynen koru.
- [ ] Apply yalnız Spark expressions kullansın; keys/target'a dokunmasın.
  Capability stratejiye göre karar versin; median'ı mean'e dönüştürme.
- [ ] Apply planında Python UDF olmadığını, row/key setinin korunduğunu doğrula.

```python
def test_spark_mean_uses_only_training_data(spark):
    """Held-out values must not influence the learned replacement value."""
    from skyulf.preprocessing.imputation.simple import SimpleImputerCalculator, SimpleImputerApplier
    train = spark.createDataFrame([(1.0,), (None,), (3.0,)], "x double")
    held_out = spark.createDataFrame([(1, 100.0), (2, None)], "id long, x double")
    state = SimpleImputerCalculator().fit(train, {"columns": ["x"], "strategy": "mean"})
    output = SimpleImputerApplier().apply(held_out, state).orderBy("id").collect()
    assert [row.x for row in output] == [100.0, 2.0]
```

Bu küçük node testi yanında key/target validation ve üç-engine parity
FeatureEngineer girişinden de test edilir.
**Komut:** `python -m pytest skyulf-core/tests/spark/test_simple_imputer.py -q`
**Kabul:** Üç engine'de aynı fill state ve key'le eşleşen sonuç; ham collect yok.

## SM-06 — StandardScaler native Spark

**Bağımlılık:** SM-05. **Değiştir:** `skyulf/preprocessing/scaling/standard.py`,
`skyulf/preprocessing/_spark.py`, capability kayıtları.
**Oluştur:** `tests/spark/test_standard_scaler.py`.
**Tüketir/üretir:** Mevcut mean/var/scale/columns/with_mean/with_std artifact'i.

- [ ] Dört with_mean/with_std kombinasyonu, sabit kolon, null/NaN, boş veri,
  tek satır, büyük offset/küçük varyans ve kolon sırası için parity testleri yaz.
- [ ] Dağıtık population variance ile ddof=0 semantiğini koru. NaN'ın aggregate
  davranışını local node'la eşleştir. Spark ML StandardScaler'ı aynı isim diye
  doğrudan kullanma. O(kolon) sonuç topla; stabil varyans kullan.
- [ ] Apply `(x - mean) / scale` ifadelerini flag'lere göre kur; sıfır scale
  ve gerekli olmayan None state'i mevcut node davranışına göre ele al.
- [ ] Floating tolerance küçük normal fixture'da rtol=1e-10/atol=1e-12;
  adversarial sayısal fixture'da gerekçeli ayrı tolerance, sonuç farkını gizleme.

```python
def test_spark_scaler_uses_population_variance(spark):
    """Sample variance would change every nonconstant scaled prediction."""
    import pytest
    from skyulf.preprocessing.scaling.standard import StandardScalerCalculator, StandardScalerApplier
    frame = spark.createDataFrame([(1, 1.0), (2, 3.0)], "id long, x double")
    state = StandardScalerCalculator().fit(frame, {"columns": ["x"]})
    output = StandardScalerApplier().apply(frame, state).orderBy("id").collect()
    assert state["var"] == pytest.approx([1.0])
    assert [row.x for row in output] == pytest.approx([-1.0, 1.0])
```

Bu assertion'lar her engine ve çapraz state yoluna genişletilir; flag ve
adversarial numeric fixture'lar ayrıca uygulanır.
**Komut:** `python -m pytest skyulf-core/tests/spark/test_standard_scaler.py -q`
**Kabul:** Population variance semantiği ve flag/NaN parity; native apply planı.

## SM-07 — İlk uçtan uca FE ve compatibility kapısı

**Bağımlılık:** SM-06. **Oluştur:** `tests/spark/test_feature_pipeline.py`,
`skyulf-core/examples/spark_feature_engineering.py`.
**Değiştir:** `skyulf/preprocessing/pipeline.py`,
`skyulf/pipeline/_pipeline.py` yalnız public integration için gerekirse.
**Üretir:** Kolon seçimi + mean imputer + scaler; state export/import ile
yerel/Spark fit ve apply. Model training burada Spark-native ilan edilmez.
Public state API'si `FeatureEngineer.export_state() -> bytes` ve
`FeatureEngineer.from_state(payload, *, frame_spec=None, execution_options=None)`
olarak eklenir. Ordered step config + codec payload dizisi saklanır; keys ve
runtime tercihi yeni çağrıdan alınır, SparkSession kaydedilmez.

- [ ] Frame projection ile key/target/feature listesini oluştur; yeni bir node
  tipi gereksizse sırf demo için registry node'u ekleme.
- [ ] Her destekli node için pandas/Polars/Spark fit × üç apply engine'i test et;
  sıralı output yerine key join ile karşılaştır. State kaydet/yüklemeyi araya koy.
- [ ] Repartition(1/2/7), ters girdi sırası ve ikinci transform çağrısında
  öğrenilmiş state'in değişmemesini doğrula.
- [ ] Baseline local pipeline ve leakage suite'leri + Spark suite çalıştır.
  Büyük veri testinde input collect yasak, sadece bounded aggregate izinli;
  terminal test collect'i sınırlı fixture için ayrı işaretlenir.

```python
def test_feature_state_applies_after_repartition(spark, fitted_feature_engineer):
    """Saved state must preserve keyed features across partition boundaries."""
    import pytest
    from skyulf.core.execution import FrameSpec, ExecutionOptions
    from skyulf.preprocessing.pipeline import FeatureEngineer
    restored = FeatureEngineer.from_state(fitted_feature_engineer.export_state(),
        frame_spec=FrameSpec(row_keys=("id",)),
        execution_options=ExecutionOptions(engine="spark"))
    frame = spark.createDataFrame([("a", 1.0), ("b", 3.0)], "id string, x double")
    first = {r.id: r.x for r in restored.transform(frame.repartition(1)).collect()}
    second = {r.id: r.x for r in restored.transform(frame.repartition(7)).collect()}
    assert first == pytest.approx({"a": -1.0, "b": 1.0})
    assert second == pytest.approx(first)
```

`fitted_feature_engineer` fixture'ı x=[1,3] eğitim verisinde gerçek mean imputer
ve scaler fit eder; bu görevde conftest'e eklenir. Aynı key karşılaştırması
pandas/Polars fit edilmiş state için de uygulanır.
**Komut:** `python -m pytest skyulf-core/tests/spark -q`
**Kabul:** G1: çalışan native FE, taşınabilir küçük state, local regresyon yok.
Sonraki dilim [inference](02-inference-plan.md); henüz endpoint veya template yok.
