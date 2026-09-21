# Skyulf Core: Spark, MLflow ve Databricks incelemesi

Tarih: 2026-09-21. İncelenen HEAD: `f713003e6eccf7372fcf7676e8f38a7cb3b06e3d`.
Son kontrolde HEAD `3587521b469b3d957c3066c86b65048349aef117` oldu;
bu iki commit arasında `skyulf-core` ve `backend` değişikliği bulunmuyor.
Core paket sürümü: `0.8.24`. Bu belge bir inceleme ve tasarım önerisidir;
Spark/MLflow implementasyonunun tamamlandığını göstermez.

## İnceleme kapsamı ve kanıt sınırı

`skyulf-core/skyulf` altındaki 197 Python dosyasının tamamı yapısal olarak
taranmıştır: 40.772 satır; importlar, sınıflar, fonksiyonlar, node kayıtları ve
yerel materialization çağrıları çıkarılmıştır. Ana yürütme yolları ve node
ailelerinin fit/apply/state davranışları kaynak koddan ayrıca izlenmiştir.
269 test dosyası envantere alınmış ve bütün core pytest paketi çalıştırılmıştır.
Runtime registry bu ortamda 100 kayıt içerir: 62 transformer ve 38 model kaydı.
Bu sayılar alias kayıtlarını da içerir; 100 bağımsız algoritma anlamına gelmez.

Ek olarak backend graph yürütme, artifact üretme ve deployment sınırları ile
frontend registry/config dönüştürme girişleri incelenmiştir. Backend/frontend'in
tam ürün denetimi veya bütün testleri bu çalışmanın kapsamında değildir.

Tam dosya ve node dökümü: [İnceleme envanteri](2026-09-21-spark-core-inventory.md).
Yapısal tarama her branch'in doğruluğunun kanıtı değildir. Bu çalışma bir Spark
uyumluluk/mimari incelemesidir; satır satır bağımsız doğruluk denetimi değildir.

Mevcut ortamda `pyspark`, `mlflow` ve `databricks` paketleri bulunmuyor.
Databricks workspace, Unity Catalog, Spark executor veya serving endpoint üzerinde
çalıştırma yapılmadı. Bu sınırlar yerel testlerin geçmesiyle ortadan kalkmaz.
Uygulama kodu, bağımlılıklar ve mevcut kullanıcı değişiklikleri değiştirilmedi.

## Sonuç

Core'un Calculator/Applier ayrımı, fitted state yaklaşımı ve engine dispatcher'ı
korunmaya değer. Ancak Spark desteği bir enum değeri ve birkaç node dalı eklemekten
fazlasını gerektiriyor. Veri kimliği, lazy execution, şema, istatistik semantiği,
artifact biçimi ve destek matrisi önce tanımlanmalı.

Önerilen sıra:

1. Engine/capability ve artifact sözleşmesi.
2. Spark temel yürütme yolu ve sınırlı bir native node grubu.
3. Kaydet/yükle ve engine'ler arası fitted-state uyumluluğu.
4. Opsiyonel MLflow tracking/model paketi ve Unity Catalog adaptörü.
5. Dönemsel batch inference, veri adaptörü ve güvenilir tablo yazma.
6. Online serving/SQL erişimi; ayrı sözleşmeyle streaming.
7. Canvas uyarlaması gerekiyorsa bu çalışan katmanlara bağlama.
8. En son template ve DAB üretimi.

MLflow ile ilgili bağımsız işler sözleşme belirlendikten sonra Spark node
çalışmalarıyla aynı dönemde geliştirilebilir. Bütün node'ların Spark'a taşınması,
ilk batch inference teslimatının önkoşulu olmamalı.

## Mevcut çalışma yolları

### Standalone core

`SkyulfPipeline` yapısal config doğrulaması yapar, registry'den model node'unu
çözer ve `FeatureEngineer` ile preprocessing zincirini oluşturur. `fit` önce
leakage kontrolünü çalıştırır. Normal eğitim ile tuning akışlarının preprocessing
yönetimi farklıdır: tuning fold-local fit ve son refit'in state'ini kullanır.

`FeatureEngineer` sıralı adımlar çalıştırır. `StatefulTransformer`, train üzerinde
Calculator.fit çalıştırıp artifact'i held-out split'lere uygular. TargetEncoder ve
WOE gibi adımlar için `fit_transform_train` ayrı eğitim temsili üretir.

Modeling sklearn tabanlıdır. `SklearnBridge` pandas/Polars verisini yerel NumPy
temsiline çevirir. CV, tuning ve değerlendirme yolları da yerel diziler kullanır.
`predict`, fitted preprocessing'i uygular ve satır sayısını korumayı denetler.

Kaynaklar: [pipeline](../../../skyulf-core/skyulf/pipeline/_pipeline.py),
[preprocessing](../../../skyulf-core/skyulf/preprocessing/pipeline.py),
[stateful transformer](../../../skyulf-core/skyulf/preprocessing/base.py),
[model wrapper](../../../skyulf-core/skyulf/modeling/sklearn_wrapper.py).

### Backend ve Canvas

Backend `PipelineEngine`, ayrı bir graph/topological yürütücüdür. Ara DataFrame'leri
artifact store'a kaydeder; branch merge ve fold preprocessing yolları vardır.
Son inference artifact'i çoğunlukla `model`, `feature_engineer`, kolonlar,
dtype'lar, engine ve job bilgisini içeren bir dict'tir.

Backend deployment, bu dict'i ve eski biçimleri okuyabilir; kendi target temizliği,
kolon kontrolü, label decoding ve threshold seçimini yapar. Standalone pipeline
ise bütün nesneyi pickle ile kaydeder. Bu iki biçimi aynı artifact sanmak hatalıdır.
Databricks model paketi ilk aşamada standalone core'dan üretilebilir; mevcut backend
artifact'leri için daha sonra açık bir adaptör gerekir.

Spark DataFrame/session/plan nesneleri mevcut joblib ara-artifact alışkanlığına
doğrudan konulmamalı. Dağıtık ara verinin yaşam döngüsü cache/checkpoint veya
sürümü belirli tablo referansı olarak tasarlanmalı.

Kaynaklar: [backend engine](../../../backend/ml_pipeline/_execution/engine/__init__.py),
[bundle üretimi](../../../backend/ml_pipeline/_execution/engine/_feature_eng.py),
[deployment](../../../backend/ml_pipeline/deployment/service.py).

## Var olan arayüzler ve eksik bağlantılar

| Alan | Kaynakta görülen durum | Gerekli karar |
| --- | --- | --- |
| EngineRegistry | pandas/Polars kayıtlı; Spark yalnızca geleceğe yönelik module eşlemesi | Spark için gerçek adapter; tanınan ama kurulu olmayan engine'e açık hata |
| Engine seçimi | Güncel fallback Polars, ContextVar ile kapsamlanmış | Input engine ve çalışma platformunu ayrı tut |
| Dispatcher | Engine-keyed mapping var; giriş hazırlama pandas/Polars'a özel | Spark hazırlama yolu, mixed X/y kontrolü, capability preflight |
| DataFrame protokolü | `shape`, `len`, assignment, copy ve local dönüşümler bekliyor | Metadata ile action/materialization işlemlerini ayır |
| ComputeBackend | `execute(func, ...)` seam'i; pipeline tarafından otomatik çağrılmıyor | Bunu node semantiği sağlamayan bir Spark çözümü olarak sunma |
| DataCatalog | load/save/exists; save seçenekleri genel kwargs | Ayrı Databricks adaptöründe açık kaynak/yazma sözleşmesi |
| ModelRegistry | InMemoryModelRegistry nesne referanslarını sürümlüyor | Kalıcı artifact URI, uzak registry version/alias ve yükleme ayrı ele alınmalı |
| Serializer | ContextVar provider var; pipeline save/load bunu kullanmıyor | Mevcut pickle formatını koruyarak yeni sürümlü paket biçimi |
| Metadata | learns_from_data/is_splitter var | fit/apply engine, satır değişimi, bağlam, model bağımlılığı ve serving uygunluğu |
| Schema | pandas/Polars çıkarımı ve sade dtype isimleri | Spark StructType, decimal, timestamp, nullability ve nested tip politikası |
| Config | Outer shape kontrolü, node params serbest | Yeni deployment/execution config'i açıkça doğrulanmalı; eski config bozulmamalı |

AGENTS metnindeki pandas-only ve filesystem-free açıklamalarının mevcut kodla
uyuşmadığı görüldü. Core pandas/Polars içeriyor ve pipeline save/load dosya kullanıyor.
Core README'si pandas fallback diyor; canlı registry ve ilgili testler Polars
fallback'i doğruluyor. Tasarım uygulamadan önce bu doküman farkları düzeltilmeli.

## Spark için temel sözleşmeler

- Spark frame'i dağıtık ilişki olarak ele al. pandas index'i veya sıralı X/y dizisi
  varsayma. Target ve entity/row key aynı ilişkide taşınmalı; join kardinalitesi
  denetlenmeli. Spark'taki fiziksel sıra model çıktı eşleştirmesinin anahtarı olamaz.
- `len`, `shape`, boşluk kontrolü ve her node öncesi/sonrası satır sayımı action
  tetikleyebilir. `get_data_stats`, prediction doğrulaması ve profiling buna göre
  düzenlenmeli. Transform planlama zamanı, gerçek execution zamanı ve Python
  tracemalloc ölçümü cluster çalışma süresi/belleği olarak raporlanmamalı.
- Otomatik `toPandas`, `collect` veya `to_numpy` fallback'i olmamalı. Yerel eğitim
  gerekiyorsa ayrı materialization adımı ve satır/kolon/bellek bütçesi olmalı.
- Fit küçük bir istatistik vektörü döndürebilir; bu tüm satırları toplamakla aynı
  değildir. Büyük vocab/group mapping için sınırsız dict yerine boyut sınırı ve
  gerekirse harici state/tablo referansı gerekir.
- Spark session runner tarafından sağlanmalı; fitted artifact içine girmemeli.
- Spark Classic ve Spark Connect aynı destek varsayılmamalı. DataFrame/SQL API'si
  öncelikli olmalı; RDD, SparkContext ve JVM erişimine dayalı çekirdek tasarlanmasın.
- Kaynak snapshot, engine/runtime sürümü ve session timezone run metadata'sında
  bulunmalı. ANSI cast davranışı ve geçersiz değer politikası açık seçilmeli.

Databricks Connect'te RDD/SparkContext ve dağıtık ML training kısıtları bulunduğu
resmi dokümanda belirtiliyor. Yerel Connect istemcisinde çalışmak, aynı kodun her
cluster/serving ortamında desteklendiği anlamına gelmez.
[Connect sınırlamaları](https://docs.databricks.com/aws/en/dev-tools/databricks-connect/python/limitations)

## Node ailelerine göre taşıma kararı

Tablodaki Spark davranışları öneridir; mevcut destek iddiası değildir.

| Aile | Mevcut ayrıntı | Öneri |
| --- | --- | --- |
| Casting | Bazı category/datetime yolları veriyi inceliyor/dönüştürüyor | Basit cast ile learned categories ayrılmalı; decimal/UTC/invalid cast testleri |
| Cleaning | Alias, regex, text, invalid/value replacement engine dalları var | SQL ifadeleri; regex/string/boolean/null eşdeğerliği; özel fonksiyonlar açık capability |
| Drop/missing | Explicit kolon seçimi ile missing-threshold fit farklı | İlk native dal; row dropping için eğitim/inference farkını koru |
| Deduplicate | Keep policy ve target survivor eşleşmesi var | `first/last` için deterministik sıra; inference skip davranışını koru |
| SimpleImputer | Kolon bazlı fill_values taşınabilir | İlk learned native node; mean/median/mode ve all-null semantiği |
| Scalers | Fit sklearn'e gider; apply çoğunlukla sayısal state kullanır | Spark aggregation ile aynı mean/scale state; native apply |
| Outliers | IQR/ZScore/ManualBounds filtreler; Winsorize clip yapar | Filtreyi online model içinde gizleme; clip ile row filtering ayrı capability |
| Binning | Custom/equal-width/quantile/kmeans yolları farklı | Custom/equal-width önce; quantile doğruluğu ve kmeans ayrı destek |
| Dummy/label/ordinal/one-hot | Bazıları mapping, bazıları sklearn encoder nesnesi saklar | Kategori canonicalization, sırayla feature isimleri, unknown/null/drop davranışı korunmalı |
| HashEncoder | BLAKE2b ile sabit bucket ataması | Spark hash fonksiyonuyla sessizce değiştirme; aynı algoritma veya açık yeni semantik |
| TargetEncoder/WOE | Eğitim temsili cross-fit; inference full-train state kullanır | Fold identity/OOF sözleşmesi olmadan native fit desteklenmesin |
| FeatureGeneration | Arithmetic/ratio/date ve learned group_agg aynı node'da | Operation bazında capability; group mapping ile geçmiş pencere feature'ını ayır |
| FeatureInteraction/Polynomial | Bazı fit yolları NumPy/pandas; sklearn feature isimleri | Kontrollü kombinasyon/kolon limiti; aynı expansion sırası |
| FeatureSelection | Fit yerel sklearn olabilir; apply yalnızca kolon düşürebilir | Spark apply desteğini Spark fit desteğinden bağımsız sun |
| DateFeatures | Yeni artifact'lerde UTC politikası | Spark timezone/day-of-week/hafta sınırlarını aynı sözleşmeye uydur |
| Lag/Rolling | Satır sayılı pencere, grup/sıra, opsiyonel row drop | Window specification ve tie-breaker; geçmiş veri/context zorunlu |
| KNN/Iterative imputation | Fitted Python nesnesi | İlk native kapsam dışında; açık worker-Python veya bounded local yol |
| EllipticEnvelope | Kolon bazlı sklearn nesneleri | Native eşdeğerlik iddiası yok; filtre davranışı ayrı planlanmalı |
| Resampling | imblearn; Polars yolu pandas'a geçiyor | Train-only; distributed sampling ile SMOTE aynı işlem sayılmamalı |
| Count/TF-IDF/Hashing/Tokenizer | sklearn tokenizer/vocab; dense frame çıktısı | Vocab/IDF/tokenization ve sparse/dense politika; Spark sınıflarıyla kör değişim yok |
| SentenceEmbedder | Model ismiyle yükleme, process cache, harici model bağımlılığı | Model revision ve dosyalarını paketle; worker/GPU/batch bütçesi |
| GeoDistance/H3 | Yerel NumPy/H3 yolları | Distance ifadesi ayrı; H3 library/runtime parity ayrı |
| DatasetProfile/DataSnapshot | İçerik okuyan fit ve passthrough apply | Bounded sample/aggregate; büyük veri snapshot'ı serialize etme |

Önemli semantik örnek: sklearn StandardScaler `ddof=0` kullanırken Spark ML
StandardScaler düzeltilmiş örnek standart sapması kullanır. Aynı sınıf adı aynı
sonuç demek değildir. Skyulf semantiğini koruyan Spark aggregate/ifadeleri gerekir.
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
[Spark](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StandardScaler.html)

Quantile/median yaklaşık hesaplanacaksa hata toleransı artifact/config'de açık
olmalı; mevcut kesin davranışın sessiz değişimi sayılmamalı. Model eşdeğerliği,
aynı fitted state'i uygulamak ve farklı engine'lerde yeniden fit etmek için ayrı
test edilmelidir; floating-point sonuçlarda uygun tolerans gerekir.

## Artifact ve inference: iki farklı taşınabilirlik

1. **Python runtime taşınabilirliği:** mevcut fitted pipeline MLflow Python model
   paketi içinde çalışır. Spark worker batch'leri pandas olarak alabilir; gerekirse
   wrapper Polars'a dönüştürür. Bu, Spark-native preprocessing değildir.
2. **State taşınabilirliği:** öğrenilmiş median/mean/category mapping aynı state ile
   Spark-native Applier tarafından uygulanır. Her node ve artifact sürümü için
   destek ayrı doğrulanır. Encoder/model nesneleri kendiliğinden Spark'a çevrilmez.

Model paketi aşağıdakileri içermeli:

- Format sürümü, Skyulf/Python/dependency sürümleri ve semantic fingerprint.
- Model blob/flavor, node türleri, fitted state ve gereken custom code.
- Ham giriş şeması, model feature şeması, kolon sırası, çıktı şeması.
- Target label mapping, probability class order, threshold politikası.
- Feature engineering sürümü ve external feature/table state referansları.
- Row-local, context gerektiren ve online uygun adımların açık bilgisi.
- Eğitim veri referansı ve sürümü; credentials/session objeleri içermez.

Mevcut pickle okuyucuları korunabilir; yeni paket, migration/adaptör olmadan
eski artifact'lerin tamamını taşınabilir ilan etmemeli. `export_model_card` ve
`fingerprint` tekrar kullanılabilir; fingerprint bir dependency lock veya veri
snapshot kimliğinin yerine geçmez.

MLflow `spark_udf` pandas batch arayüzü sunar; driver'a tüm tabloyu toplamak gerekmez.
Worker'da model bir kere yüklenmeli/reuse edilmeli ve saf predict çalışmalı;
UDF içinde registry'ye kayıt, tracking run açma veya tabloya yan etkili yazma olmamalı.
[MLflow pyfunc](https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html)

## Deneyler

Mevcut kodla, yeni ürün kodu yazmadan iki küçük çalışma yapıldı:

1. Polars üzerinde `SimpleImputer -> StandardScaler -> OneHotEncoder -> linear_regression`
   fit edildi; pickle round-trip sonrası aynı girdi pandas ve Polars ile skorlandı.
   En büyük tahmin farkı `7.105427357601002e-15`. İki parçalı Polars tahmininin
   tam batch tahmininden farkı `0.0`. Bu yalnızca bu zincirin yerel kanıtıdır;
   Spark/MLflow uyumluluğunun veya bütün node'ların kanıtı değildir.
2. `[1,2,3,4,5,6]` için window=3 rolling mean tam veride
   `[1,1.5,2,3,4,5]`; iki üçlü parçaya bölündüğünde `[1,1.5,2,4,4.5,5]` çıktı.
   Satır sayısı/sırası korunurken bile batch sınırı anlamı değiştirebilir.

Bu nedenle sadece `preserve_rows=True` kontrolü distributed UDF güvenliği için
yeterli değildir. Lag/rolling'in gerekli history/context'i ayrı hazırlanmalıdır.

## Eğitim, tuning, evaluation ve monitoring

- Mevcut model ailesi sklearn temelli; XGBoost/LightGBM wrapper'ları da yerel
  estimator arayüzündedir. Spark engine seçimi modelleri distributed yapmaz.
- İlk aşamada Spark FE + açık bounded materialization + mevcut trainer desteklenebilir.
  Distributed training ayrı backend/model desteğidir; bütün algoritmaların Spark
  karşılığı varmış gibi generic `engine=spark` altında gizlenmemeli.
- CV/tuning NumPy, sklearn splitter/clone, positional slicing ve joblib/Optuna
  yürütmesine dayanıyor. `n_jobs` executor sayısı değildir. Fold refit ve target
  encoder OOF davranışı distributed portta korunmalı; cached full-data FE leakage yaratabilir.
- Split membership seed kadar row/entity/time kimliğine de bağlı olmalı. Yeniden
  partition edilince aynı split isteniyorsa sabit key/fold assignment kaydedilmeli.
- Evaluation sklearn/SciPy array tabanlı. Distributed confusion matrix/sum/count
  gibi yeterli istatistikler ayrı yol olabilir; AUC/quantile ve global sıralama
  isteyen metriklerin yaklaşık/kesin anlamı açıklanmalı. Ham prediction listesini
  bütün dataset için run metadata'sına koymamak gerekir.
- Threshold tuning validation verisi üzerinde; class sırası ve threshold seçimi
  endpoint/batch arasında aynı olmalı. Tuning metadata'sı ile aktif serving kararını ayır.
- SHAP yerel model ve bounded sample gerektiriyor; online predict'in zorunlu parçası olmamalı.
- EDAAnalyzer Polars'a özel; lazy view kullanması Spark dağıtık çalışma değildir.
  Numeric/categorical/date/text/geo, PCA/outlier/causal/rules/temporal modülleri
  farklı sample ve yerel algoritma yolları kullanıyor.
- DriftCalculator Polars/SciPy temelli. Büyük tablolar için Spark'ta histogram/count
  üretip küçük özetleri değerlendirmek uygun bir tasarım adayıdır; mevcut KS veya
  Wasserstein algoritmalarını histogram yaklaşımıyla aynı metrik diye değiştirme.
- Expectation kontrolleri pandas/Polars'a özel. Spark kolon/null/unique/range
  kontrolleri aggregate ile çalışmalı; EDA/visualizer modülleri tabloyu toplamasın.

## Aylık batch, online, SQL ve streaming

Platform, feature engine, training backend, tracking, registry, tetikleyici ve
çıktı birbirinden bağımsız boyutlardır. Laptop'ta MLflow/UC kullanılabilir;
Databricks job içinde küçük veri pandas/Polars ile işlenebilir. DAB bir dağıtım
tanımıdır, dataframe engine'i değildir.

### Aylık batch sözleşmesi

Run girdileri `period_start`, `period_end`, `as_of`, model version ve kaynak
snapshot'tır. Varsayılan öneri tarih aralığını `[start, end)` olarak tanımlamak ve
timezone'u açık tutmaktır. Skorlanacak dönem ile feature üretmek için gereken
history aralığı aynı şey değildir.

Model alias'ı run başında somut sürüme çözülür. Veriyi okuma, feature hazırlama,
schema/quality kontrolü, skor üretme, çıktı kontrolü ve publish ayrı adımlardır.
Çıktı entity/row key, scoring period, prediction, gerekiyorsa probability,
model version, feature version, run ID ve hesaplama zamanını taşır.

Append/history ile dönem replace/merge ayrı seçeneklerdir. Merge anahtarı ve
tekrar çalıştırma politikası açık olmalı. Stage/validate/publish yaklaşımı ve Delta
işlem sınırı tasarlanmalı; checkpoint tek başına exactly-once iddiası sağlamaz.
Backfill'de tarihsel model mi güncel model mi kullanılacağı açık parametre olmalı.
Late-arriving data, aynı dönem için eşzamanlı iki run ve yarıda kalan yayın test edilmeli.

Scheduler core dışında kalır. Databricks Jobs aylık/zaman tabanlı, tablo/file
olaylı veya manuel tetikleme sağlar.
[Jobs](https://docs.databricks.com/gcp/en/jobs/triggers)

### Online ve ai_query

Online endpoint yalnızca paketlenmiş model ve istekte sağlanan/lookup ile alınan
feature'larla çalışabilmeli. Spark window/history işlemlerini küçük HTTP request'in
içinde var sayamayız. Önceden hesaplanmış feature tablosu/online lookup gerekebilir.

`ai_query`, custom model senaryosunda serving endpoint'ini SQL'den çağırır.
Registry URI'sine doğrudan local predict değildir. Aynı endpoint HTTP ve SQL
istemcilerine hizmet verebilir; bir batch job da ai_query kullanabilir. Request
şeması, sonuç tipi, hata politikası, retry ve kapasite endpoint adaptöründe tanımlanmalı.
[ai_query](https://docs.databricks.com/gcp/en/sql/language-manual/functions/ai_query)

### Streaming

Bağımsız kayıt skorlama ile stateful zaman özellikleri ayrı kapsamdır. Micro-batch
model sürüm politikası, checkpoint, watermark, geç gelen veri, deduplication ve
tekrar işleme sözleşmesi olmadan batch runner'a sadece streaming bayrağı eklenmemeli.

## Paketleme ve custom code

Spark/MLflow opsiyonel olmalı; temel import bunları veya Databricks SDK'yı zorunlu
kılmamalı. Local OSS Spark, Databricks job runtime ve Connect development için
ayrı dependency profilleri gerekir. Databricks Connect ve OSS PySpark aynı
ortamda kurulu olduğunda çakışabilir.
[Connect kurulum kısıtı](https://docs.databricks.com/aws/en/dev-tools/databricks-connect/python/troubleshooting)

Core halen pandas, Polars, PyArrow, sklearn ve diğer yerel kütüphanelere bağımlı.
Spark extra eklemek bunları kaldırmaz. İlk teslimatta geniş dependency ayrıştırması
yerine hedef DBR/Python ile doğrulanmış constraints kullanılmalı. Daha sonra
gerekiyorsa serving paketi hafifletilebilir. Model serving wheel'i sürümlenmeli.

Custom feature fonksiyonları config'de bir callable/module referansı ve capability
bilgisiyle yer alabilir. Registry importları her worker'da erişilebilir paket içinde
olmalı; yalnızca notebook'ta tanımlanmış fonksiyona veya driver belleğine dayanılmamalı.
Fonksiyon engine, giriş/çıkış şeması, row key ve context sözleşmesini uygulamalı.
İlk aşamada arbitrary kodu bütün engine'lere otomatik çevirme sözü verilmemeli.

## Teslimat dilimleri ve kabul ölçütleri

| Dilim | Somut çıktı | Kabul ölçütü |
| --- | --- | --- |
| A | Capability, veri/kimlik, şema ve artifact v1 kararı | Her node/operation fit ve apply için destek/ret sebebi; mevcut API uyumu |
| B | Spark adapter + sınırlı native FE | SimpleImputer, StandardScaler, açık kolon/drop ve basit dönüşümler gerçek Spark'ta çalışır; gizli collect yok |
| C | State portability | Local fit -> Spark apply ve Spark fit -> local apply; null/category/time/decimal edge case; save/load |
| D | MLflow ve UC | Tracking kapalıyken core çalışır; model package kayıt/yükleme; signature/class/threshold parity |
| E | Aylık batch runner + Delta sink | Gerçek Databricks run; model pinning; dönem, backfill ve idempotent yayın kanıtı |
| F | Bağlamsal FE ve kalan node grupları | Window/group, OOF ve büyük state kuralları ayrı testlerle; desteklenmeyenler açık |
| G | Online/SQL ve gerekiyorsa streaming | Python batch ile eşdeğer tahmin; bağımlılıklar bağımsız ortamda yüklenir; context eksikliği reddedilir |
| H | Canvas ve template | UI yalnızca desteklenen yolları sunar; DAB çalışan runner'ları paketler |

İlk uygulama işi bütün node'ları bir seferde değiştirmek olmamalı. A dilimini
netleştirip B/C için bir küçük uçtan uca akış kurmak, geri kalan node portlarına
güvenilir örnek ve test sözleşmesi sağlar. Bu belge uygulama onayı veya tamamlanmış
implementation plan değildir; önerilen geliştirme sırasıdır.

## Doğrulama

- Seçili engine/artifact/pipeline/leakage sözleşmeleri: **260 passed**, 88 warnings.
- Bütün core: **9815 passed, 68 skipped, 1 failed**, 1075 warnings; **216.48 saniye**.
- Başarısız test:
  `tests/integration/test_wrapped_polars_frames.py::test_sentence_embedder_accepts_wrapped_polars_frame`.
  Dar tekrarda Hugging Face model metadata isteği `[WinError 10013]` ile
  engellendi; ardından `Cannot send a request, as the client has been closed`
  hatası geldi. Aynı test `HF_HUB_OFFLINE=1` ile önbellekteki modeli kullanarak
  **1 passed (5.15 saniye)** sonucu verdi. Bu dar doğrulama, tam suite sonucunu
  tümüyle başarılı olarak değiştirmez.
- İki yerel davranış deneyi: engine değişimi/round-trip ve rolling batch sınırı.
- Spark/MLflow/Databricks integration testi yapılmadı; paketler bu ortamda yok.
- Uygulama kodu değişmedi; lint/typecheck/build sonucu iddia edilmiyor.

Tam core komutu:

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests -q --tb=short -o addopts= -p no:cacheprovider --basetemp .tmp-spark-review-full
```
