# Distributed Python Inference Implementation Plan

> **For agentic workers:** Use `executing-plans`; finish SM-07 and read
> [ARCHITECTURE.md](ARCHITECTURE.md) before executing these tasks.

**Goal:** Native Spark FE + Python model ve uyumlu Python FE + model yolları.
**Architecture:** Aynı model bundle iki giriş aşamasını açıkça ayırır. Spark
veriyi dağıtır; Python modeli iterator batch'lerinde tahmin üretir.
**Tech Stack:** PySpark DataFrame/Arrow, pandas/NumPy, mevcut sklearn modelleri.
**Spec:** [ARCHITECTURE.md](ARCHITECTURE.md).

## Global constraints

- Model eğitimi mevcut local API'dedir; burada distributed training sözü yok.
- Worker'da SparkSession, credentials, tracking mutation veya tablo yazma yok.
- Sıra yerine row_keys; retry-safe saf tahmin; context-dependent pipeline erken ret.
- Core dependency yükü ve legacy artifact davranışı korunur; paketler versioned.
- Bütün yeni source/test yolları aşağıda `skyulf-core/` köküne göredir.

## SM-08 — Ortak inference bundle

**Bağımlılık:** SM-07. **Oluştur:** `skyulf/inference/__init__.py`,
`skyulf/inference/bundle.py`, `tests/spark/test_inference_bundle.py`.
**İncele/değiştir:** `skyulf/pipeline/_pipeline.py`, `skyulf/core/serialization.py`;
backend legacy artifact okuyucusuna bu görevde yeni format dayatma.
**Üretir:**

```text
build_bundle(pipeline, *, input_stage, feature_order) -> InferenceBundle
save_bundle(bundle, path) -> None
load_bundle(path) -> InferenceBundle
predict_local(frame, bundle) -> pandas.DataFrame
```

`InferenceBundle`: format_version, input_stage(raw|features), feature_order,
input/output schema, FE state, estimator payload, classes/positive label,
threshold provenance, package/runtime requirements ve semantic digest.
`probability_columns` sınıf sırasıyla eşleşen ordered tuple alanıdır;
regression için boş tuple'dır.
`predict_local` sonucu regression için `prediction`; classification için
`prediction` ve sınıf sırası manifestte tanımlı probability kolonlarıdır.

- [x] Mevcut pipeline'dan bundle oluştur; raw ve features girişlerini ayrı
  fixtures ile kaydet/yükle. `features` modunda scaler ikinci kez uygulanmamalı.
- [x] İsim doğru ama kolon sırası farklı, eksik/ekstra kolon, dtype uyuşmazlığı,
  bilinmeyen version ve model boyut sınırı için failure testleri yaz.
- [x] Model payload ile manifest checksum'ını doğrula; dependency manifesti
  credentials ve session taşımasın. Pickle tabanlı payload yalnız güvenilir
  kaynaklardan yüklenir; manifest checksum'ı güvenilirlik kanıtı değildir.
- [x] Eski standalone ve backend artifact formatları için explicit adapter
  tasarla; ilk bundle yalnız doğrulanmış standalone girdiyi kabul etsin.
  Legacy backend adapter SM-18'de mevcut threshold/label davranışıyla test edilir.

```python
def test_bundle_round_trip_predictions(fitted_regression_pipeline, tmp_path, raw_frame):
    """Packaging must preserve the prediction function, including preprocessing."""
    import numpy as np
    from skyulf.inference.bundle import build_bundle, save_bundle, load_bundle, predict_local
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    before = predict_local(raw_frame, bundle)["prediction"]
    save_bundle(bundle, tmp_path / "model")
    after = predict_local(raw_frame, load_bundle(tmp_path / "model"))["prediction"]
    np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-12)
```

`raw_frame`, x/z kolonları ve farklı scale içeren küçük pandas verisidir.
SM-09 ile bu fixture ve `fitted_regression_pipeline`, ortak
`tests/spark/conftest.py` dosyasındadır; pipeline fixture'ı imputer +
scaler + mevcut linear regression ile
local fit edilmiş gerçek SkyulfPipeline. Veri üretimi seed=17 ve held-out
satırlarla sabitlenir; mock estimator bu testin yerine geçmez.
Local giriş yalnız feature kolonlarıdır; ekstra id/target reddedilir. SM-09
Spark verisine ayrı id ekler; training fixture iki engine için ortak kullanılır.
Kolon adı/sırası modelin NumPy girdisinden tahmin edilmez; başarılı pipeline fit
metadata-only raw/model schema kaydeder. Eski standalone pickle için adapter
`SkyulfPipeline.load` → `build_bundle`; schema yoksa refit gerekir. Backend
dict adapter'ı SM-18'e kadar açıkça reddedilir. Ayrıntılar [bundle rehberinde](../../docs/user_guide/inference_bundles.md).
**Komut:** `python -m pytest skyulf-core/tests/spark/test_inference_bundle.py -q`
**Kabul:** Kayıt/yükleme, feature sırası ve raw/features giriş ayrımı doğrulanmış.

2026-09-21: SM-08 tamamlandı. Komutlar, runtime sonuçları ve kapsam sınırları
[tamamlanma kaydında](OPEN_QUEUE.md). Dağıtık inference kapısı G2 henüz kapanmadı.

## SM-09 — Native Spark FE ardından Python model

**Bağımlılık:** SM-08. **Oluştur:** `skyulf/inference/spark.py`,
`tests/spark/test_native_features_inference.py`.
**Üretir:** `predict_spark(frame, bundle, *, frame_spec, options, mode)`;
`mode="native_features"` ham girdiye önce native FE uygular, worker'a yalnız
model feature'ları + keys gönderir. Sonuç Spark DataFrame'dir.

- [x] Gerçek Spark action ve key-based karşılaştırma testi yaz. Model training
  local fixture'da açıkça yapılsın; Spark fit çağrısı veya gizli conversion olmasın.
- [x] Native FE'yi G1 yoluyla çalıştır; Python modelini mapInPandas iterator
  başına bir kez yükle. Output schema önceden bundle'dan gelsin; key dtype korunsun.
- [x] Feature order ve input_stage doğrulamasını driver'da yap. Geniş kolonları
  gereksiz yere worker'a gönderme; session/connection closure capture etme.
- [x] Repartition ve Arrow batch boyutu değişince aynı key için aynı tahmini,
  boş partition ve boş dataframe'de doğru boş schema'yı doğrula.

```python
def test_native_path_matches_local(spark, raw_frame, regression_bundle):
    """Partitioning must not change a row-independent model's predictions."""
    import numpy as np
    from skyulf.core.execution import FrameSpec, ExecutionOptions
    from skyulf.inference.bundle import predict_local
    from skyulf.inference.spark import predict_spark
    frame = spark.createDataFrame(raw_frame.assign(id=range(len(raw_frame)))).repartition(3)
    actual = predict_spark(frame, regression_bundle,
        frame_spec=FrameSpec(row_keys=("id",)),
        options=ExecutionOptions(engine="spark", python_batch_rows=2),
        mode="native_features").toPandas().sort_values("id")
    expected = predict_local(raw_frame, regression_bundle)
    np.testing.assert_allclose(actual["prediction"], expected["prediction"], rtol=1e-10)
```

`regression_bundle`, SM-08 pipeline'ından `build_bundle` ile üretilen fixture'dır.
Testteki toPandas yalnız küçük test sonucunu karşılaştırmak içindir; üretim
inference yolunda driver collection olmaması ayrıca denetlenir.
**Komut:** `python -m pytest skyulf-core/tests/spark/test_native_features_inference.py -q`
**Kabul:** G2a: Spark FE native, model dağıtık Python; FE bir kez uygulanır.

2026-09-21: SM-09 tamamlandı. 393 gerçek Spark lane testi, 10050 base core testi
ve rehber örneği geçti; komutlar ve sınırlar [tamamlanma kaydındadır](OPEN_QUEUE.md).

SM-09'un ilk runner'ı raw regression bundle kabul eder. Prepared-feature giriş,
Python FE worker modu, classification ve streaming açık ret verir. Numeric
model schema değişimleri action öncesinde reddedilir. Integer/boolean model
feature'ları Arrow hassasiyetini korumak için non-nullable Spark schema ister;
key'ler ayrı non-null/unique değer kontrolünden geçer. Model chunk satır limiti
Arrow transport limiti değildir. Detaylar [bundle rehberindedir](../../docs/user_guide/inference_bundles.md).

## SM-10 — Worker'da Python FE + model

**Bağımlılık:** SM-09. **Değiştir:** `skyulf/inference/spark.py`, bundle
capability preflight. **Oluştur:** `tests/spark/test_python_pipeline_inference.py`.
**Üretir:** Aynı fonksiyonda `mode="python_pipeline"`; sadece batch-independent
ve row-preserving fit edilmiş pipeline desteği.

- [ ] Imputer/scaler modelinin SM-09 ile aynı tahmini verdiğini test et.
  pandas/Polars training kaynağı farklı bundle'lar kullan.
- [ ] Rolling/lag, batch'ten yeniden group statistics hesaplayan custom step,
  row drop ve resampling pipeline'larını action'dan önce reddet.
- [ ] FeatureEngineering fit worker'da çağrılmasın; yalnız frozen apply/predict.
  Model/FE state mutation testi ve iki ardışık action sonucu kontrolü ekle.
- [ ] Custom step yalnız package import yolu + capability beyanı ile kabul
  edilsin; bilinmeyen callable otomatik batch-safe sayılmasın.

```python
def test_python_pipeline_rejects_window_context(spark, rolling_bundle):
    """A rolling window must not reset invisibly at Arrow batch boundaries."""
    import pytest
    from skyulf.core.execution import FrameSpec, ExecutionOptions
    from skyulf.core.capabilities import UnsupportedExecutionError
    from skyulf.inference.spark import predict_spark
    with pytest.raises(UnsupportedExecutionError, match="context"):
        predict_spark(spark.range(6), rolling_bundle,
            frame_spec=FrameSpec(row_keys=("id",)),
            options=ExecutionOptions(engine="spark"), mode="python_pipeline")
```

`rolling_bundle` fixture'ı mevcut rolling node ile fit edilmiş, window context
metadata'sı taşıyan geçerli bundle'dır; bozuk manifest ile ret üretme.
**Komut:** `python -m pytest skyulf-core/tests/spark/test_python_pipeline_inference.py -q`
**Kabul:** G2b: iki inference yolunda parity; temporal karşı örnek sessizce çalışmaz.

## SM-11 — Classification ve dağıtık yürütme kapısı

**Bağımlılık:** SM-10. **Oluştur:** `tests/spark/test_prediction_contract.py`,
`tests/spark/test_worker_isolation.py`,
`skyulf-core/examples/spark_batch_inference.py` (repo köküne göre).
**Değiştir:** Bundle/prediction output contract yalnız mevcut model semantiğiyle.

- [ ] String binary labels, çok sınıf, positive label, probability sütun sırası,
  saved threshold ve override precedence için local/Spark key parity testleri.
- [ ] Spark'ın lazy olması nedeniyle assertion'lar action çağırmalı. İki
  repartition ve en az iki Arrow batch limitinde aynı tahmini doğrula.
- [ ] Worker'ın wheel'i kullanması ve eksik dependency'nin anlaşılır hatası için
  ayrı-process testi ekle. Model her satır için yüklenmesin; iterator seviyesinde
  yükleme helper'ını ölç. Task sayısına eşit tam bir global yükleme sayısı varsayma.
- [ ] SM-09'un nullable integer/boolean model-feature schema sınırını yeniden
  değerlendir. Arrow → pandas null dönüşümünde büyük integer hassasiyeti
  korunmadan desteği genişletme; float'tan integer'a geri cast çözüm değildir.
  Destek açılmayacaksa mevcut erken ret ve kullanıcı dokümanını koru.
- [ ] Büyük sentetik veri üzerinde driver RSS, worker peak ve wall time ölç;
  row count/feature width/partition sayısını kaydet. Driver belleğinin input
  boyutuyla doğrusal büyümemesini araştır; sabit hızlanma oranı vaat etme.
- [ ] Güncel core suite ve optional-dependency import testini çalıştır;
  bilinen failure varsa neden/izolasyon kanıtıyla release kararına yaz.

```python
def test_probability_columns_follow_class_manifest(classification_bundle, raw_frame):
    """A class-order mismatch would silently reverse business decisions."""
    import numpy as np
    from skyulf.inference.bundle import predict_local
    output = predict_local(raw_frame, classification_bundle)
    columns = classification_bundle.probability_columns
    assert list(output[columns].columns) == list(columns)
    np.testing.assert_allclose(output[columns].sum(axis=1), 1.0, atol=1e-7)
```

Classification fixture SM-08 fixture'larına eklenir; manifest class sırası
estimator.classes_ ile karşılaştırılır. Sum testi tek başına yeterli değildir:
Spark probability değerleri local estimator.predict_proba ile sınıf bazında
karşılaştırılır; threshold label sonucu ayrıca assertion alır.
**Komut:** `python -m pytest skyulf-core/tests/spark -q`
**Kabul:** G2: iki yol, regression/classification ve worker packaging kanıtlı.
Endpoint yok; [MLflow/batch](03-mlflow-batch-delivery-plan.md) aşamasına geçilir.
