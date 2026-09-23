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

## SM-09 — 2026-09-21 tamamlanma kaydı

- Başlangıç commit'i `81745a7b`; bu kayıt SM-09 çalışma ağacına aittir.
- `inference/spark.py` ve public `predict_spark`: raw girdiye native FE,
  ardından `mapInPandas` iterator'larında Python regression tahmini. Model
  iterator başına bir kez yüklenir; yalnız feature'lar ve key'ler worker'a gider.
  Ortak model tahmin helper'ı local davranışı tekrar kullanır. Session, frame
  veya bağlantı worker closure'a girmez; pandas dönüşümü worker batch'indedir.
- `tests/spark/test_native_features_inference.py`; SM-08'in fitted pipeline ve
  raw-frame fixture'ları ortak conftest'e taşındı. pandas/Polars eğitim, bundle
  kayıt/yükleme, repartition 1/3, Arrow limit 1/5, eksik değerler, boş veri ve
  partition'lar, birleşik key'ler, `2**53` üstü identity, worker projection,
  bounded driver validation, iterator model yükleme ve chunk sınırı doğrulandı.
- Ruling: native model schema önce yalnız Spark ifadeleri oluşturularak
  incelenir; sonra mevcut FE key/preservation kontrolleri çalışır. Preview veri
  action'ı içermez, öğrenilmiş kuralları tekrar fit etmez. Local integer imputer
  çıktısı ile Spark double promotion uyuşmazlığı artık action'dan önce hata verir.
- Ruling: nullable integer/boolean model feature schema'ları erken reddedilir;
  Arrow null batch dönüşümü hassasiyet kaybetmeden çözülene kadar destek ilan
  edilmez. Non-nullable integer model feature ve büyük integer key gerçek action
  ile geçti. Key'lerin nullable schema metadata'sı ayrı değer kontrolüyle korunur.
  Broader nullable transport SM-11'e kayıtlıdır.
- Ruling: raw model input kolonlarının kendi sırası/dtype'ları korunmalı;
  extra kolonlar projection ile elenir. Key'ler integer/string/boolean'dır;
  feature/output isimleriyle çakışamaz. Incoming prediction kolonu da reddedilir.
  `FrameSpec.target` boş olmalıdır. Raw regression dışındaki stage/task/mode ve
  streaming açık ret verir. `python_batch_rows` model çağrısını sınırlar;
  Arrow transport konfigürasyonuna veya kullanıcının Spark session'ına yazılmaz.
- TDD red: eksik public API; streaming, native dtype ve nullable transport
  girdilerinin doğrulamadan önce Spark action'a ulaşması ayrı ayrı yeniden
  üretildi. Dar final test: `JAVA_HOME=.cache/spark-jdk/jdk-17.0.20.1+1`,
  `SKYULF_REQUIRE_SPARK=1`, `.venv-spark/Scripts/python.exe -m pytest
  skyulf-core/tests/spark/test_native_features_inference.py
  skyulf-core/tests/spark/test_inference_bundle.py -q -o addopts=
  -p no:cacheprovider --basetemp .cache/sm09-runner-final`
  → **101 passed**, 110.42 saniye; owned JVM temiz kapandı.
- Bağımsız review iki schema/Arrow bulgusunu doğruladı; düzeltmelerin ve red
  regresyonların yeniden incelemesinde açık yüksek/orta bulgu kalmadı.
- Eski backend artifact akışı ayrıca doğrulandı: `.venv/Scripts/python.exe -m
  pytest tests/integration/test_bundle_preprocessing_integrity.py
  tests/integration/test_full_inference_pipeline.py -q --tb=short -o addopts=
  -p no:cacheprovider --basetemp .cache/sm09-legacy`
  → **22 passed, 30 warnings**, 7.80 saniye. Backend kaynakları değiştirilmedi;
  full FE/model joblib reader korunuyor, yeni bundle adapter'ı SM-18'de.
- Tam core: `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest skyulf-core/tests
  -q --tb=short --disable-warnings -o addopts= -p no:cacheprovider
  --basetemp .cache/sm09-core-full`
  → **10050 passed, 274 skipped, 1097 warnings**, 194.81 saniye. Base ortamda
  PySpark yok; gerçek Spark testleri ayrı lane'de doğrulanır.
- Ruff/format ve tam repo ty başarılı. Bundle/Spark rehberleri mevcut artifact
  davranışını, yeni runner'ı, dtype ve sürüm sınırlarını açıklıyor.
  `mkdocs build --strict` başarılı.
- Tam Spark: `JAVA_HOME=.cache/spark-jdk/jdk-17.0.20.1+1`,
  `SKYULF_REQUIRE_SPARK=1`, `.venv-spark/Scripts/python.exe -m pytest
  skyulf-core/tests/spark skyulf-core/tests/unit/test_execution_capabilities.py
  skyulf-core/tests/unit/test_feature_state.py
  skyulf-core/tests/unit/test_pipeline_inference_schema.py -q --tb=short
  -o addopts= -p no:cacheprovider --basetemp .cache/sm09-spark-full`
  → **393 passed, 2 warnings**, 418.45 saniye. PySpark 4.0.3 / Java 17;
  JVM temiz kapandı. Warning'ler mevcut Split alias'ına ait.
- Spark runtime: Python 3.12.10, skyulf-core 0.9.0, pandas 2.3.2,
  Polars 1.44.2, sklearn 1.9.1, NumPy 1.26.4, SciPy 1.17.1, Arrow 24.0.0.
- Rehberin iki Python bloğu `.venv-spark/Scripts/python.exe
  .cache/sm09_docs_example.py` ile aynı ortamda çalıştırıldı: local kayıt/yükleme
  ve key'li Spark sonucu **[40.0, 20.0]**, başarılı assert'ler ve temiz JVM kapanışı.
- Sınır: bu G2a / native FE + regression teslimatıdır. SM-10 Python FE worker
  modu, SM-11 classification/worker wheel/ölçek kapısıdır. Databricks/Connect,
  MLflow/UC, endpoint ve template doğrulaması yoktur. Sıradaki görev SM-10.

## SM-11 — 2026-09-22 final validation record

- Starting point: `a930f8a5`; implementation commits: `04dbe303` and
  corrective `7cdd622c`. The earlier DONE label was reopened during review and
  is now restored only after the corrective evidence below.
- `predict_spark` now accepts raw regression and classification bundles in both
  `native_features` and `python_pipeline` modes. The output schema maps
  manifest label dtypes to Spark types and keeps probability columns in the
  estimator class order. Saved tuning and explicit pipeline threshold rules
  reuse the local `_predict_features` contract.
- Added `tests/spark/test_prediction_contract.py` for string binary labels,
  integer multiclass labels, probability ordering, saved thresholds, two
  partition layouts and two Arrow batch limits. Added
  `tests/spark/test_worker_isolation.py` to verify one model load per iterator
  and bounded model calls. The former classification rejection regression test
  now asserts the supported output contract.
- Added `skyulf-core/examples/spark_batch_inference.py`, which runs the same
  classification bundle through both Spark modes. The example sets the current
  interpreter as the Python worker on Windows when no worker override exists.
- Corrective focused verification:
  `.venv-spark/Scripts/python.exe -m pytest -o addopts='' --basetemp
  .tmp-sm11-regression skyulf-core/tests/spark/test_native_features_inference.py
  skyulf-core/tests/spark/test_python_pipeline_inference.py
  skyulf-core/tests/spark/test_prediction_contract.py
  skyulf-core/tests/spark/test_worker_isolation.py
  skyulf-core/tests/spark/test_inference_bundle.py -q --disable-warnings
  --maxfail=1` → **63 passed**. JVM cleanup prints the
  existing Windows `ERROR: Access denied` after a successful exit.
- Full Spark gate after the corrective transport guard:
  `.venv-spark/Scripts/python.exe -m pytest -o addopts='' --basetemp
  .tmp-sm11-spark-final skyulf-core/tests/spark
  skyulf-core/tests/unit/test_execution_capabilities.py
  skyulf-core/tests/unit/test_feature_state.py
  skyulf-core/tests/unit/test_pipeline_inference_schema.py -q --disable-warnings
  --maxfail=1` → **418 passed, 2 warnings**, 525.50 seconds. The warnings are
  existing; Windows prints the cleanup access-denied line after exit 0.
- Both example commands completed with exit code 0 and printed matching
  `prediction`, `probability_0` and `probability_1` results. Ruff check and
  format passed for all changed Python files.
- Nullable integer/boolean model-feature transport remains rejected because
  Arrow can widen values with nulls. Worker wheel isolation, scale/RSS
  measurements, MLflow, Databricks, endpoints and templates remain later work.
- Corrective preflight rejects nullable integral/boolean raw inputs in
  `python_pipeline` before key validation. Independent estimator class/probability
  assertions and a non-default threshold decision assertion are covered. A
  wheel subprocess smoke imported `skyulf.inference.spark` from the built 0.9.0
  wheel. Local scale measurements recorded 10k/2 partitions at 3.929s and
  50k/8 partitions at 8.089s with driver peak RSS 246.2/246.8 MB and child
  process peak RSS 887.2/922.7 MB. These results are evidence, not a production
  performance guarantee.
- Next task: SM-12 optional MLflow tracking, with tracking disabled remaining a
  valid local/Spark workflow.

## SM-15I - automatic new-row scoring, DONE

The user requires scheduled jobs to discover new source rows automatically.
The first run must score the existing bounded snapshot; later runs must score
only inserts since the last successful target commit, including late-arriving
rows in an earlier calendar period. The current `replaceWhere` writer cannot
publish only new rows in an overlapping period without deleting older
predictions. Implement a separate append-safe publisher and derive the source
watermark from the committed output receipt. See the
[SM-15I plan](12-sm15i-incremental-scoring-plan.md). This is a prerequisite
for SM-20a. The runner passed local real-Delta gates and the isolated two-job Databricks
rehearsal, including a source with no date column; see the
[SM-15I validation report](13-sm15i-live-validation-report.md). A separate
[real NYC taxi rehearsal](15-sm15i-real-nyctaxi-live-report.md) trained a Skyulf
pipeline with MLflow metrics and registered UC model version 1, then passed
200 initial + 100 later real-row predictions with an unchanged prior batch and
no-op replay. SM-20a remains the next task.

## SM-15L - 2026-09-23 monthly local UC Delta publication, DONE

- Added `run_local_batch` with truthful `local_pipeline` publication mode.
  It loads one pinned UC source month under row/byte limits, scores through
  fitted pandas or Polars FE/model code, converts only final keyed results to
  an explicitly typed Spark DataFrame, and reuses guarded Delta period writes.
  The target and admission control table are provisioned separately.
- Focused SDK/reader tests: **32 passed, 1 optional Spark skip**; batch
  contracts: **25 passed**. Real local Spark/Delta: **6 passed**, including
  two months, exact replay, stale version, accidental empty period, admission
  denial, invalid target schema and stale model/source pins without mutation. Ruff, Ty and strict
  MkDocs passed for the changed scope.
- Two isolated Databricks serverless jobs completed **SUCCESS**. January
  published 80 pandas predictions to target version 1. February appended 80
  source rows, published 80 Polars predictions to target version 2, preserved
  all January rows/metadata, replayed without a new commit, and rejected a
  stale logical request. Both periods matched direct local gold by key.
  See the [SM-15L live report](11-sm15l-live-validation-report.md) for run IDs,
  model digests, test resources and scope limits. The date/version choices
  were scripted for this proof; automatic new-row scoring is SM-15I.

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
  transport-budget option for wider or larger workloads. SM-15I then added
  automatic new-row scoring; next is SM-20a.

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
