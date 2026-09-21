# Spark ve MLflow — Open Queue

Güncelleme: 2026-09-21. **SM-00–SM-08 tamamlandı; sıradaki SM-09. Hedef: 0.9.0.**
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
| SM-06 | Native StandardScaler | SM-05 | DONE | Population variance, dört flag ve numeric parity; aşağıda kanıt |
| SM-07 | Uçtan uca FE kapısı | SM-06 | DONE | Üç engine çapraz fit/apply; kayıt/yükleme; aşağıda kanıt |
| SM-08 | Ortak inference bundle | SM-07 | DONE | Raw/features ayrımı; model metadata round-trip; aşağıda kanıt |
| SM-09 | Native FE + worker model | SM-08 | READY | Spark tahmini local ile key bazında aynı |
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

## SM-06 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `992d13b2`; bu kayıt SM-06 çalışma ağacına aittir.
- `scaling/_spark_standard.py`: native fit/apply, population variance (`ddof=0`),
  dört with_mean/with_std kombinasyonu, portable codec v1 capability. Mean/std
  seçeneklerine bağlı None alanları mevcut artifact sözlüğüyle uyumlu.
- Fit en fazla iki distributed aggregate sorgusu çalıştırır; her biri tek
  O(kolon) istatistik satırı döndürür. İlk sorgu min/max referansları, infinity
  ve satır/otomatik seçim kontrolleri; ikinci sorgu referanstan farkların
  mean/var_pop/count değerleri. İki flag kapalıysa ikinci sorgu gerekmez.
  Sıfırı kapsayan aralık referans 0; diğerlerinde sıfıra yakın uç değer kullanılır.
- Null/NaN fit'te dışlanır. All-missing istatistikler NaN; tek gözlem/sabit
  varyans 0 ve scale 1. Near-constant scale için sklearn'in float64 hata sınırı
  uygulanır. Seçili infinity/overflow reddedilir; Spark double precision sınırı
  ve sayısal tolerance rehberde açık. Decimal/string/nested scaling unsupported.
- Apply native ifadelerle lazy; Python UDF/action/pandas dönüşümü yok. Kolon
  adları literal, input kolon sırası korunur; keyed runner key/target'ı doğrular.
  Doğrudan apply eksik kolonları local scaler gibi atlar; pipeline mevcut giriş
  kolon sözleşmesini korur. Integer/bool seçili kolonlar enabled apply'da double olur.
- `_spark_numeric.py`: imputer ile ortak kolon çözümleme, null/NaN, otomatik
  seçim istatistikleri ve case-sensitive isim kontrolü. Mevcut imputer davranışı
  regresyon testleriyle korundu; local pandas/Polars uygulaması değiştirilmedi.
- 57 scaler testi: 36 çapraz engine/flag kombinasyonu, held-out veri, column order,
  null/NaN, sabit/tek/boş veri, dtype, key/target, lazy plan ve bounded collection.
  Normal fixture rtol=1e-10/atol=1e-12; 1e12 offset ve küçük spread fixture'ında
  temsil aralığına bağlı rtol=0.002, repartition 1/2/7.
- TDD başlangıcı **33 failed, 16 passed**; ilk scaler+imputer koşusu **85 passed**.
  Bağımsız review doğrudan avg'nin büyük sabit kolonda 0 yerine 0.0968
  üretmesini, ardından koşulsuz min-reference'ın mixed-sign küçük katkıyı
  kaybetmesini buldu. Her bulgu iki red regresyonla yeniden üretildi; son tam
  Spark koşusunda düzeltmeler geçti. Review son recheck'te önemli bulgu kalmadı.
- Tam Spark: `JAVA_HOME=.cache/spark-jdk/jdk-17.0.20.1+1`,
  `SKYULF_REQUIRE_SPARK=1`, `.venv-spark/Scripts/python.exe -m pytest
  skyulf-core/tests/spark skyulf-core/tests/unit/test_execution_capabilities.py
  -q --tb=short -o addopts= -p no:cacheprovider
  --basetemp .cache/sm06-spark-full`
  → **214 passed, 2 warnings**, 197.25 saniye. PySpark 4.0.3 / Java 17;
  Windows JVM child-process cleanup başarılı. Warning'ler mevcut Split alias'ına ait.
- Tam core: `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests
  -q --tb=short --disable-warnings -o addopts= -p no:cacheprovider
  --basetemp .cache/sm06-core-full`
  → **9945 passed, 198 skipped, 1077 warnings**, 186.87 saniye.
  Bu koşu 55 yeni scaler runtime testini PySpark olmayan base ortamda skip etti;
  sonradan eklenen iki mixed-sign testi dahil tüm 57 vaka üstte gerçek Spark'ta geçti.
- Ruff/format, tam repo ty ve `mkdocs build --strict` geçti. Rehberdeki session
  kurulumundan sonraki sekiz Python snippet'i aynı gerçek session'da çalıştı:
  `.venv-spark/Scripts/python.exe -m pytest .cache/sm06_docs/test_spark_examples.py
  -q --tb=short -o addopts= -p no:cacheprovider --basetemp .cache/sm06-doc-examples`
  → **1 passed**, 32.00 saniye; mevcut pandas fillna FutureWarning.
- Sınır: FeatureEngineer export_state/from_state, tüm zincirin persistence
  kapısı ve örnek script SM-07'de. Distributed model inference, MLflow ve
  Databricks/Connect entegrasyon doğrulaması henüz yok.

## SM-07 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `802f60d3`; bu kayıt SM-07 çalışma ağacına aittir.
- Public `FeatureEngineer.export_state() -> bytes` ve `from_state(...)`:
  sıralı step isimleri/config'leri ve mevcut node codec envelope'ları tek
  versioned JSON paketinde. İlk destek mean/constant imputer ve StandardScaler.
  Öğrenilmiş kolon seçimi explicit hale gelir; auto-selection yeniden çalışmaz.
  Session/keys/target runtime'dan yeniden alınır; core dosya I/O yapmaz.
- `core/portable_pipeline.py` toplam wire bütçesini parse öncesinde kontrol eder;
  step sırası, config ve nested state checksum kapsamındadır. Bilinmeyen
  node/version, custom applier, bozuk veya eksik fit state'i açıkça reddedilir.
  `_feature_state.py` yalnız bilinen applier'ları kurar; Spark key/target ile
  öğrenilmiş feature çakışmasını execution başlamadan reddeder.
- Node ve pipeline wire JSON compact UTF-8; semantic checksum'ın mevcut ASCII
  canonical biçimi değişmedi. Unicode kolonlarda re-encoding kaynaklı yanlış
  bütçe aşımı giderildi; escaped surrogate isimler eski codec ile uyumlu.
- Başarılı tuning'in adopted FE state'i export edilebilir. Başarısız local/model
  refit export'u kapatır; boş pipeline da buna dahildir. Spark'ın başarısız refit'i
  önceki başarılı state'i koruma sözleşmesi değişmedi. Legacy pickle ve pipeline
  fingerprint hesapları mevcut regresyonlarla korundu.
- `tests/spark/test_feature_pipeline.py`: gerçek üç fit × üç apply engine,
  dosyada JSON round-trip, repartition 1/2/7, ters girdi ve tekrar transform.
  Key bazlı sonuçlar aynı; train-only learned state değişmez. 10.000 satırlı
  imputer→scaler testinde unbounded collect ve local conversion yasak; yalnız
  bounded aggregate/validation satırları driver'a döner. Fixture dosya içindedir.
- Bağımsız review Unicode byte limiti ve tuning fitted-state aktarımını buldu.
  Gerçek Spark Unicode testi düzeltmeden önce failed, sonra passed. Tuning/model
  × boş/scaler dört regresyonu önce failed, sonra passed; exception sonrası boş
  pipeline'ın yanlış export edilmesi de düzeltildi. Son review recheck temiz.
- Dar local state/codec/lifecycle koşusu: **101 passed, 28 warnings**, 1.75 saniye.
  Tam core: `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests
  -q --tb=short --disable-warnings -o addopts= -p no:cacheprovider
  --basetemp .cache/sm07-base-full`
  → **9969 passed, 223 skipped, 1077 warnings**, 192.35 saniye. Base ortamda
  PySpark yok; runtime skip'leri gerçek Spark lane'inden ayrı değerlendirilir.
- `skyulf-core/examples/spark_feature_engineering.py --state-path
  .cache/sm07-example-features.json` gerçek PySpark 4.0.3/Java 17 ile çalıştı:
  **1949 byte**, üç keyed feature beklenen ±sqrt(1.5)/0; dosya save/load ve
  JVM kapanışı başarılı. Rehber kullanım, frozen columns ve runtime sınırlarını açıklar.
- Tam Spark: `JAVA_HOME=.cache/spark-jdk/jdk-17.0.20.1+1`,
  `SKYULF_REQUIRE_SPARK=1`, `.venv-spark/Scripts/python.exe -m pytest
  skyulf-core/tests/spark skyulf-core/tests/unit/test_execution_capabilities.py
  skyulf-core/tests/unit/test_feature_state.py -q --tb=short -o addopts=
  -p no:cacheprovider --basetemp .cache/sm07-spark-full`
  → **261 passed, 2 warnings**, 345.28 saniye. PySpark 4.0.3 / Java 17;
  Windows JVM child-process cleanup başarılı. Warning'ler mevcut Split alias'ına ait.
- Ruff/format, tam repo ty ve `mkdocs build --strict` geçti. Rehberde session
  kurulumundan sonraki dokuz Python snippet'i aynı gerçek session'da çalıştı:
  `.venv-spark/Scripts/python.exe -m pytest .cache/sm07_docs/test_spark_examples.py
  -q --tb=short -o addopts= -p no:cacheprovider --basetemp .cache/sm07-doc-examples`
  → **1 passed, 1 warning**, 38.00 saniye; mevcut pandas fillna FutureWarning.
- Sınır: G1 feature engineering içindir; model bundle/inference, MLflow,
  Databricks/Connect ve endpoint/template doğrulaması değildir. Sonraki görev SM-08.

## SM-08 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `7099733c`; bu kayıt SM-08 çalışma ağacına aittir.
- `skyulf/inference/` public API: `build_bundle`, `save_bundle`, `load_bundle`,
  `predict_local`, immutable `InferenceBundle`. Metadata ve payload bytes ayrı;
  `raw` input'ta FE bir kez uygulanır, `features` doğrudan model tahminidir.
  pandas/Polars girdisi; çıktı pandas prediction/probability kolonlarıdır.
- `SkyulfPipeline.fit` metadata-only `_inference_schemas` kaydeder: target hariç
  raw input ve gerçek model training feature sırası/dtype'ları. Başarısız refit
  temizler; raw frame tutulmaz. NumPy bridge'in kaybettiği kolon adları tahminle
  doldurulmaz. Schema capture 31 gerçek frame/tuple/split/wrapper/tuning testiyle
  korundu; mevcut pickle ve fingerprint davranışı değişmedi.
- Katı manifest: stage, input/model/output schema, class/probability sırası,
  positive label, tuning ve explicit pipeline threshold provenance, sabit
  dependency listesi ve digest'ler. pandas/Polars Boolean/bool eşleştirmesi açık.
  Eksik/ekstra/tekrarlı/sırası yanlış veya dtype'ı farklı feature'lar reddedilir.
- Kayıt üç sabit isimli dosyadır: manifest.json, features.json, model.pkl.
  Var olan hedefe yazılmaz. FE+manifest state bütçesi ile model wire bütçesi ayrı;
  protokol-5 PickleBuffer byte sayımı ve explicit FE limitinin tüm çağrılara
  aktarımı test edildi. Kaynak pipeline'ın sonraki mutasyonu paketi değiştirmez.
- Semantic digest pickle protokolüne bağlı değildir; ayrıca model/FE wire
  checksum kontrolü yapılır. Pickle yalnız güvenilir üreticiden yüklenir.
  Python major/minor ve skyulf-core/sklearn/NumPy/SciPy sürümleri yüklemeden
  önce doğrulanır. pandas/Polars sürümleri provenance olarak kaydedilir.
- Ruling: local giriş yalnız feature kolonlarını kabul eder; id gibi ekstra
  kolonlar otomatik elenmez. SM-09 runner key'leri model girişinden açıkça ayırır.
  Fixture seed=17, iki farklı ölçekli x/z kolonu ve held-out satırlar içerir;
  test dosyasında tutulur. Bu, plandaki tek x/id örneğine göre kolon sırası
  hatasını da yakalar; SM-09 shared fixture'a ihtiyaç duyduğunda taşınabilir.
- Ruling: mevcut `core/serialization.py` provider'ı ve eski backend okuyucu
  değiştirilmedi. Standalone adapter `SkyulfPipeline.load` → `build_bundle`;
  schema bilgisi olmayan eski pickle yeniden fit gerektirir. Backend dict
  doğrudan reddedilir, adapter SM-18'de. İlk FE desteği SM-07'nin node'ları;
  split node yerine explicit SplitDataset ile training partitions verilebilir.
- Bağımsız review Polars Boolean alias'ını, 8 MiB üstü FE için custom budget
  aktarımını ve büyük estimator protocol-5 PickleBuffer serialization hatasını
  yeniden üretti. Red regresyonlardan sonra düzeltildi. Re-review: gerçek
  144 KB KNN bundle ve 8.39 MB FE / 16 MiB bütçe ile pandas+Polars inference geçti;
  açık önemli bulgu kalmadı. KNN gibi modeller kendi state'lerinde örnek tutabilir;
  schema capture'ın veri tutmaması estimator'ın bu davranışını değiştirmez.
- Dar testler: **76 passed, 2 skipped, 21 warnings**, 4.46 saniye. İki skip
  Spark-local giriş reddi vakasıdır; gerçek Spark lane'inde ayrıca çalışır.
- `skyulf-core/examples/inference_bundle.py --bundle-dir .cache/sm08-example-model`
  çalıştı: kayıt/yükleme sonrası **[40.0, 20.0]**, raw/features parity ve aynı
  semantic digest. Ayrı base-process import kontrolü PySpark/MLflow yüklemedi.
- Yeni `docs/user_guide/inference_bundles.md`, Spark rehberi linki ve nav kaydı.
  Rehberin Python örneği gerçek base ortamında **1 passed**, 2.08 saniye;
  `mkdocs build --strict` başarılı. Ruff/format ve tam repo ty geçti.
- Tam core: `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests
  -q --tb=short --disable-warnings -o addopts= -p no:cacheprovider
  --basetemp .cache/sm08-core-full`
  → **10045 passed, 225 skipped, 1097 warnings**, 193.33 saniye. Base ortamda
  PySpark yok; yeni gerçek Spark input-ret testleri burada iki skip'tir.
- Tam Spark: `JAVA_HOME=.cache/spark-jdk/jdk-17.0.20.1+1`,
  `SKYULF_REQUIRE_SPARK=1`, `.venv-spark/Scripts/python.exe -m pytest
  skyulf-core/tests/spark skyulf-core/tests/unit/test_execution_capabilities.py
  skyulf-core/tests/unit/test_feature_state.py
  skyulf-core/tests/unit/test_pipeline_inference_schema.py -q --tb=short
  -o addopts= -p no:cacheprovider --basetemp .cache/sm08-spark-full`
  → **339 passed, 2 warnings**, 340.64 saniye. PySpark 4.0.3 / Java 17;
  Windows JVM child-process cleanup başarılı. Warning'ler mevcut Split alias'ına ait.
- Sınır: bu görev local paketleme/inference içindir. Spark worker model runner
  SM-09, ikinci Python FE yolu SM-10, dağıtık classification kapısı SM-11'dir.
  MLflow, Databricks/Connect, endpoint ve template doğrulaması henüz yok.
