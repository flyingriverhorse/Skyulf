# Spark ve MLflow — Open Queue

Güncelleme: 2026-09-21. **SM-05 tamamlandı. Sonraki görev: SM-06. Hedef: 0.9.0.**
Bu dosya kısa çalışma sırasıdır; detaylar bağlantılı planlarda.

Durumlar: READY = başlanabilir; WAIT = önceki görev bekleniyor;
LATER = son aşama; ACTIVE = yürütülüyor; BLOCKED = somut dış engel;
DONE = kanıtla tamamlandı. SM-00 commit: `105a6fe4`.

| Sıra | Görev | Bağımlılık | Durum | Kısa bitiş ölçütü |
| --- | --- | --- | --- | --- |
| SM-00 | Baseline ve Spark test ortamı | — | DONE | Güncel master: 231 baseline; 2 Spark smoke; [kanıt](BASELINE.md) |
| SM-01 | Execution/capability kuralları | SM-00 | DONE | Immutable config; registry operation desteği; aşağıda kanıt |
| SM-02 | Spark engine ve conversion sınırı | SM-01 | DONE | Gerçek Spark adapter; conversion korumaları; aşağıda kanıt |
| SM-03 | Dispatcher, schema, row keys | SM-02 | DONE | Keyed Spark FE giriş/preflight; aşağıda kanıt |
| SM-04 | Versioned portable state | SM-03 | DONE | Tagged state, limit/version/schema kontrolü; aşağıda kanıt |
| SM-05 | Native SimpleImputer | SM-04 | DONE | Mean/constant train-only fit, Spark apply; aşağıda kanıt |
| SM-06 | Native StandardScaler | SM-05 | READY | Population variance ve flag parity |
| SM-07 | Uçtan uca FE kapısı | SM-06 | WAIT | Üç engine çapraz fit/apply; kayıt/yükleme |
| SM-08 | Ortak inference bundle | SM-07 | WAIT | Raw/features ayrımı; model metadata round-trip |
| SM-09 | Native FE + worker model | SM-08 | WAIT | Spark tahmini local ile key bazında aynı |
| SM-10 | Worker Python FE + model | SM-09 | WAIT | Batch-safe pipeline; window gibi yollar açık ret |
| SM-11 | Classification ve ölçek kapısı | SM-10 | WAIT | Class/proba/threshold parity; worker kanıtı |
| SM-12 | Opsiyonel MLflow tracking | SM-11 | WAIT | Off bağımsız; gerçek run lifecycle ve izolasyon |
| SM-13 | MLflow model packaging | SM-12 | WAIT | Temiz ortamda pyfunc yükleme ve parity |
| SM-14 | Registry/Unity Catalog adapter | SM-13 | WAIT | URI/signature; alias → sabit version |
| SM-15 | Aylık batch + Delta sink | SM-14 | WAIT | Dönem izolasyonu, retry ve conflict kontrolü |
| SM-16 | Gerçek Databricks kapısı | SM-15 | WAIT | Job/run kanıtı; UC load; iki inference yolu |
| SM-17 | Kalan node aileleri | SM-16 | WAIT | Aile bazlı port; her kayıt supported/unsupported |
| SM-18 | Backend/Canvas ve custom FE | SM-17 | LATER | Capability UI/API; DAG/artifact uyumu |
| SM-19 | HTTP/SQL erişimi; optional streaming | SM-18 | LATER | Desteklenen erişimlerde batch ile aynı tahmin |
| SM-20 | Template / Databricks Bundle | SM-19 | LATER | Seçilebilir working runner; validate ve smoke |

## Detay planlar

- SM-00–SM-07: [Core/Spark](01-core-spark-plan.md)
- SM-08–SM-11: [Inference](02-inference-plan.md)
- SM-12–SM-20: [MLflow/batch/son teslimatlar](03-mlflow-batch-delivery-plan.md)
- Ortak kurallar: [Mimari](ARCHITECTURE.md), [Doğrulama](VALIDATION.md)

SM-17 başlangıcında 17a column/cast/date, 17b quantile/category,
17c group/window, 17d OOF alt görevlerine ayrılır. Her alt görevin kendi
test/review kapısı vardır. Tüm node'ların native olması zorunlu değildir;
gerçekten desteklenen kapsam ilan edilerek G5 kapatılır.
SM-19 streaming isteğe bağlıdır; HTTP/SQL bittiğinde streaming unsupported
olarak açıkça kaydedilebilir, template'i belirsiz süre bloke etmez.

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
test döngüsüyle tamamla ve kanıtı yaz. Sonraki bağımlılığı aç. Endpoint veya
template'e önden başlama. SM-00 test/runtime hazırlığıdır; native node uygulaması
ve MLflow entegrasyonu henüz yok.

## SM-00 — 2026-09-21 tamamlanma kaydı

- Başlangıç HEAD: `2397c11648c27645a4ee332412a503c394c1c2f2`; değişiklikler uncommitted.
- Test fixture/smoke, optional extra/requirements, Spark CI lane ve 0.9.0
  Unreleased changelog oluşturuldu. Detaylı ortam ve komutlar [BASELINE.md](BASELINE.md).
- Local baseline: 206 passed, 5 mevcut numeric warning.
- PySpark 4.0.3 + Temurin Java 17.0.20.1 + Python 3.12.10: 2 passed, temiz kapanış.
- Spark olmayan base ortam: 2 skipped; zorunlu Spark modu: beklenen 2 setup error,
  exit 1. Eksik dependency başarılı CI gibi raporlanmıyor.
- Ruff check/format ve scoped ty geçti. GitHub Actions ve Databricks çalıştırılmadı.
- Sonraki görev SM-01; bu kanıt native Spark node desteği anlamına gelmez.

## SM-01 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `105a6fe4`; bu kayıt doğrulanmış SM-01 çalışma ağacına aittir.
- Yeni `core/execution.py`: frozen FrameSpec ve ExecutionOptions, strict
  `from_config`, pozitif integer limitleri; engine default'unu değiştirmez.
- Yeni `core/capabilities.py`: immutable operation/engine/config desteği ve
  structured UnsupportedExecutionError. NodeRegistry mevcut Calculator sınıfında
  declaration tutar; alias aynı sınıfı paylaşır, subclass açıkça yeniden ilan eder.
- Yeni testler `tests/unit/test_execution_contract.py` ve
  `tests/unit/test_execution_capabilities.py`; Spark gerektirmeyen kurallar bu
  yüzden planlanan spark/ dizini yerine unit/ altındadır.
- İlk 48 test eksik modüllerle fail oldu; uygulama sonrası geçti. Tam core:
  `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests -q
  --tb=short --disable-warnings -p no:cacheprovider --basetemp .cache/sm01-core`
  → **9850 passed, 84 skipped, 1077 warnings**, 127.52 saniye; 3 snapshot geçti.
- Bağımsız inceleme bloklayan sorun bulmadı. Ek alias/subclass, type-exact
  selector, invalid registration ve optional import regresyonlarından sonra:
  iki yeni unit dosyası + test_engines_registry.py + test_core_seams.py
  → **94 passed**. Bu son eklemeler test kapsamıdır; ürün kodu değişmedi.
- Tam repo ty, scoped Ruff/format ve yeni core modüllerinde CCN<=8 kontrolü geçti.
  `mkdocs build --strict` başarılı; docs/user_guide/spark.md kontrat örnekleri
  doğrudan çalıştırıldı. Kullanıcı belgeleri uygulamayla birlikte güncellendi.
- Spark engine/node desteği henüz etkin değil. Existing local pipeline preflight'ı
  bu yeni API'ye bağlanmadı; SM-03'te yapılacak. Sıradaki görev SM-02.

## SM-02 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `b9eef16a`; bu kayıt SM-02 çalışma ağacına aittir.
- SparkEngine ve ayrı DistributedDataFrame sözleşmesi eklendi. Gerçek dataframe
  tanıma, literal kolon seçimi ve native frame erişimi var; varsayılan local
  engine korunur. Eksik Spark dependency/registration local fallback'e gitmez.
- Adapter session oluşturmaz; `len/shape/to_pandas/to_numpy/to_arrow` ve
  SklearnBridge üzerinden Spark feature/target dönüşümü açık hata verir.
  `from_pandas/create_dataframe` çağırana kendi SparkSession'ını kullanmasını söyler.
- TDD: ilk 12 gerçek Spark testi başarısızdı; adapter sonrası runtime dahil
  14 test geçti. Son iki lazy-operation/namespace testi ve review düzeltmesiyle:
  `JAVA_HOME=.cache/spark-jdk/jdk-17.0.20.1+1`, `SKYULF_REQUIRE_SPARK=1`;
  `.venv-spark/Scripts/python.exe -m pytest skyulf-core/tests/spark
  skyulf-core/tests/unit/test_spark_optional_import.py
  skyulf-core/tests/unit/test_engines_registry.py -q --tb=short -o addopts=
  -p no:cacheprovider --basetemp .cache/sm02-spark-reviewed`
  → **46 passed**, 11.07 saniye; JVM temiz kapandı.
- PySpark 4.0.3 / Python 3.12.10 / Java 17 profili kullanıldı. Base ortamda
  subprocess import engeli PySpark/MLflow yokken core importunu ve local seçimi
  doğrular. Aynı test Spark kurulu ortamda da dependency yokluğunu simüle eder.
- Bağımsız review yalnız dependency hata mesajındaki dağıtım adını düzeltti:
  `skyulf-core[spark]`. İlk tam core koşusu bu düzeltme sırasında eski toplanmış
  test beklentisi/yeni subprocess kaynağı nedeniyle 1 failed verdi; son kaynakla
  odaklı test **3 passed**. Son kaynakla tam core tekrarı:
  `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests -q
  --tb=short --disable-warnings -o addopts= -p no:cacheprovider
  --basetemp .cache/sm02-core-reviewed`
  → **9875 passed, 84 skipped, 1077 warnings**, 163.31 saniye.
- Ruff check/format ve tam repo ty geçti. Yeni adapter max CCN=3.
  `mkdocs build --strict` geçti; Spark rehberindeki runtime/adapter örnekleri
  gerçek oturumda çalıştırıldı. Rehber ve 0.9.0 Unreleased changelog güncel.
- Sınır: native FE node, Spark dispatcher veya model inference etkin değil.
  Databricks/Connect/uzak CI çalıştırılmadı. Sonraki görev SM-03.

## SM-03 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `2d8cd8e5`; bu kayıt SM-03 çalışma ağacına aittir.
- FeatureEngineer `frame_spec` / `execution_options` ile Spark'a opt-in olur.
  Yeni `_spark.py` native yürütmeyi local profiler'dan ayırır. Dispatcher input
  hazırlığı engine-keyed mapping kullanır. `SkyulfSchema` Spark tiplerini okur.
- Key/target tek frame'de kalır; key null/duplicate ve desteklenmeyen dtype/schema
  kontrolleri var. Bütün node fit/apply capability'leri action öncesi kontrol
  edilir. Her node sonrası protected projection `exceptAll` karşılaştırması
  key/label değişimini, row drop/expand'i reddeder. Validation sonucu limit(1)'dir;
  full scan/shuffle maliyeti rehberde yazılıdır. Native node desteği ilan edilmez.
- Otomatik feature listesi key/target dışındadır. Literal dot/backtick kolonlar,
  case-insensitive ambiguity ve internal aggregate alias collision testlidir.
  Local get_data_stats Spark'ı açıkça reddeder; Spark FE metriklerinde row count,
  fit_time ve peak_memory unknown (`None`). Driver süre ölçümü ayrı isimlidir.
- Planlanan base.py/data/dataset.py değişikliği gerekmedi: mevcut local profiler'a
  girilmeden routing yapılır; tuple/SplitDataset Spark input sınırda reddedilir.
  Local frame/tuple/split API'leri korunur; explicit engine seçimi her split'i denetler.
- TDD: ilk Spark contract koşusu **10 failed**; uygulama sonrası **10 passed**.
  Ek güvence testlerinde case-insensitive isim hataları yeniden üretildi ve
  düzeltildi. Bağımsız review internal alias bulgusunu verdi; son recheck temiz.
- Son Spark + dispatcher/optional import komutu:
  `JAVA_HOME=.cache/spark-jdk/jdk-17.0.20.1+1`, `SKYULF_REQUIRE_SPARK=1`;
  `.venv-spark/Scripts/python.exe -m pytest skyulf-core/tests/spark
  skyulf-core/tests/unit/test_spark_optional_import.py
  skyulf-core/tests/unit/test_preprocessing_dispatcher.py -q --tb=short -o addopts=
  -p no:cacheprovider --basetemp .cache/sm03-spark-reviewed`
  → **69 passed**, 61.59 saniye; PySpark 4.0.3/Python 3.12.10/Java 17, temiz kapanış.
- İlk full core **9875 passed, 107 skipped, 1077 warnings**, 170.97 saniye.
  Ardından explicit local engine/split seçimindeki iki hata testle üretildi;
  fix sonrası yeni 9 test + dispatcher/pipeline/base/optional import **124 passed**.
  Son full core komutu:
  `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests -q
  --tb=short --disable-warnings -o addopts= -p no:cacheprovider
  --basetemp .cache/sm03-core-final`
  → **9884 passed, 107 skipped, 1077 warnings**, 170.30 saniye.
- Ruff/format, tam repo ty ve `mkdocs build --strict` geçti. Spark rehberindeki
  beş Python örneği aynı gerçek session'da çalıştırıldı; beklenen unsupported
  capability örneği de doğrulandı. Changelog/rehber birlikte güncellendi.
- Sınır: built-in native node, portable state codec, distributed model inference,
  explicit aggregate metrics ve Databricks/Connect testi henüz yok. SM-04 sırada.

## SM-04 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `44b5241e`; bu kayıt SM-04 çalışma ağacına aittir.
- `core/portable_state.py`: UTF-8 tagged JSON v1 encode/decode; explicit encode
  byte bütçesi ve parse öncesi decode limiti (varsayılan 8 MiB). StandardScaler ve
  mean/constant SimpleImputer gerçek artifact sözlükleri, boş no-op state dahil.
- Integer/string/bool/null ayrımı, non-finite float ve signed zero, kolon sırası
  korunur. Unknown version/node/fields/tags, duplicate JSON/map keys/columns,
  inconsistent state, unsupported Python objects ve aşırı nesting reddedilir.
  Checksum canonical semantic content üzerinden SHA-256; imza değildir.
- `core/artifacts.py` StandardScaler mean/var alanları mevcut disabled-flag
  davranışına uygun `None` kabul edecek şekilde düzeltildi. İlgili scaling testi
  mean-enabled koşulda non-null assertion ile gerçek beklentiyi açıkça doğrular.
- `pipeline/seal.py` değişmedi: eski fingerprint ve pickle formatını değiştirmek
  gerekmedi. Portable checksum kendi versioned vocabulary'sine sahiptir.
  API'de bulunmayan producer/schema bilgisi uydurulmadı; mimari metadata sınırı
  güncellendi. Future bundle bu bilgiyi gerçek input/execution bağlamından alacak.
- TDD ilk koşu eksik modül nedeniyle fail; ilk uygulama **44 passed**. UTF-16
  kabulü ve review'un compact Unicode byte-limit bulgusu ayrı red/green testlerle
  düzeltildi. Review son recheck'te limit sınırlarını bağımsız doğruladı.
- Odaklı komut: `.venv/Scripts/python.exe -m pytest
  skyulf-core/tests/spark/test_portable_state.py
  skyulf-core/tests/unit/test_pipeline_seal.py skyulf-core/tests/unit/test_pipeline.py
  skyulf-core/tests/unit/test_pipeline_threshold_fingerprint.py
  skyulf-core/tests/integration/test_scaling.py -q --tb=short -o addopts=
  -p no:cacheprovider --basetemp .cache/sm04-reviewed`
  → **188 passed**, 5 mevcut numeric warning, 4.69 saniye. Eski save/load testleri
  ve fingerprint golden testleri değişmeden geçti. İlk odaklı koşunun Windows
  default temp erişim hatası workspace içindeki basetemp ile giderildi.
- Spark profile: `.venv-spark/Scripts/python.exe -m pytest
  skyulf-core/tests/spark/test_portable_state.py -q --tb=short -o addopts=
  -p no:cacheprovider --basetemp .cache/sm04-spark-codec`
  → **61 passed**, 1.22 saniye. Codec JVM kullanmaz; bu sonuç Spark FE kanıtı değildir.
- Tam core: `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests
  -q --tb=short --disable-warnings -o addopts= -p no:cacheprovider
  --basetemp .cache/sm04-core`
  → **9945 passed, 107 skipped, 1077 warnings**, 168.18 saniye.
  Ruff/format, tam repo ty ve
  `mkdocs build --strict` geçti. Rehberdeki yeni codec örneği base ortamda çalıştı.
- Sınır: ayrı codec API'sidir; FeatureEngineer/model worker için otomatik paketleme
  yok. Byte limiti toplam process memory limiti değildir. Native SimpleImputer
  SM-05, StandardScaler SM-06; Spark runtime/Databricks bu görevde çalıştırılmadı.

## SM-05 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `81216e16`; bu kayıt SM-05 çalışma ağacına aittir.
- `imputation/_spark_simple.py`: mean/constant için native fit/apply. Mean,
  missing counts ve otomatik seçim istatistikleri aynı aggregate sorgusunda;
  driver'a tek istatistik satırı döner. Constant da mevcut artifact missing-count
  sözleşmesi için aggregate kullanır. Boş kolon seçimi doğrudan no-op.
- Apply native expressions kullanır; veri action'ı, Python UDF veya pandas
  fallback yok. FeatureEngineer'ın key/target doğrulama action'ları devam eder.
  Integer constant hassasiyeti korunur; mean double promotion yapabilir.
  All-null mean uydurulmaz; numeric/string/bool sabitlerin sınırları rehberde.
- Registry mean/constant için fit/apply capability ve codec v1 bildirir.
  Runner node default'larını preflight öncesi normalize eder; açık `columns=[]`
  korunur. `state_max_bytes` fit artifact'ında ve transform action'larından önce
  uygulanır. Direct apply codec yapısal doğrulamasını wire limiti dayatmadan kullanır.
- pandas/Polars/Spark arasında 18 fit/apply kombinasyonu JSON round-trip ile
  doğrulandı. Yeni test dosyasında 36 vaka: train-only öğrenme, null/NaN,
  all-null, kolon seçimi, dtype, key/target, lazy plan, state bütçesi ve precision.
- İlk TDD koşusu **19 failed, 8 passed**; native uygulama sonrası **26 passed**,
  bir fixture DDL quoting hatası düzeltildi. Bağımsız review üç yakın-binary
  değerin yanlış elenmesini ve imported state'in `x`/`X` çakışmasını buldu.
  Gerçek runtime tekrarları ve regresyonlarla düzeltildi; review recheck temiz.
- Gerçek Spark komutu: `JAVA_HOME=.cache/spark-jdk/jdk-17.0.20.1+1`,
  `SKYULF_REQUIRE_SPARK=1`, `.venv-spark/Scripts/python.exe -m pytest
  skyulf-core/tests/spark skyulf-core/tests/unit/test_execution_capabilities.py
  -q --tb=short -o addopts= -p no:cacheprovider
  --basetemp .cache/sm05-spark-full`
  → **157 passed, 2 warnings**, 112.97 saniye. PySpark 4.0.3 / Java 17;
  Windows JVM child-process cleanup başarılı. Warning'ler mevcut Split alias'ına ait.
- Tam core: `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests
  -q --tb=short --disable-warnings -o addopts= -p no:cacheprovider
  --basetemp .cache/sm05-core-full`
  → **9945 passed, 143 skipped, 1077 warnings**, 185.45 saniye.
  Base ortamda PySpark yok; 36 yeni runtime testi burada skip, üstte gerçek Spark'ta geçti.
- Ruff/format, tam repo ty ve `mkdocs build --strict` geçti. Spark rehberindeki
  session kurulumundan sonraki yedi Python snippet'i aynı gerçek session'da
  çalıştırıldı: `.venv-spark/Scripts/python.exe -m pytest
  .cache/sm05_docs/test_spark_examples.py -q --tb=short -o addopts=
  -p no:cacheprovider --basetemp .cache/sm05-doc-examples`
  → **1 passed**, 27.12 saniye; mevcut pandas fillna FutureWarning.
- Sınır: median/mode, StandardScaler native apply, distributed model inference,
  MLflow ve Databricks/Connect entegrasyon kanıtı henüz yok. Sonraki görev SM-06.
