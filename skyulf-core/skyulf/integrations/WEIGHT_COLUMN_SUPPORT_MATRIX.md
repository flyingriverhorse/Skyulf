# Ağırlıklı eğitim: kullanım ve destek

Kullanım rehberi ve destek tablolarının ana dokümantasyondaki karşılığı:
[Weighted Training & Support](../../../docs/user_guide/weighted_training.md).
Bu dosya geliştirme sırasındaki ayrıntılı doğrulama kayıtlarını da korur.

2026-10-02, `091` çalışma ağacı; commit edilmedi.
Python 3.12.10 / sklearn 1.8.0 / imbalanced-learn 0.14.1.

## Üç ayarın farkı

- `weight_column`: verideki ağırlık kolonunun adı. Bundle bunu feature'lardan ayırır.
- `sample_weight`: her eğitim satırının ağırlığı; Core bunu modelin `fit` metoduna verir.
- `class_weight`: sınıflandırma ayarı; model `params` bölümünde kalır. Native olmayan
  sınıf ağırlığını Core o fit'in etiketlerinden hesaplar. İki ağırlık bir kez çarpılır.

Desteklenen modellerde class_weight ve sample_weight tek başına veya birlikte
kullanılabilir. Voting `weights` model tahminlerinin katsayısıdır; KNN
`weights="distance"` komşu uzaklığına ilişkindir. Bunlar satır ağırlığı değildir.

## Modeller

34 katalog seçeneği gerçek Core fit + bağımsız model referansıyla denendi:
**32 geçti, 2 KNN beklenen şekilde reddedildi.** Bu doğrudan eğitim denemesidir;
her veri/parametre kombinasyonu için garanti değildir.

| Tür | Desteklenen node'lar |
| --- | --- |
| Classification | logistic_regression, adaboost_classifier, bernoulli_nb, decision_tree_classifier, extra_trees_classifier, gaussian_nb, gradient_boosting_classifier, hist_gradient_boosting_classifier, lgbm_classifier, multinomial_nb, random_forest_classifier, sgd_classifier, svc, xgboost_classifier |
| Regression | linear_regression, adaboost_regressor, decision_tree_regressor, elasticnet_regression, extra_trees_regressor, gradient_boosting_regressor, hist_gradient_boosting_regressor, lasso_regression, lgbm_regressor, random_forest_regressor, ridge_regression, svr, xgboost_regressor |
| Koşullu ensemble | voting_classifier/regressor, stacking_classifier/regressor, calibrated_classifier |
| Desteksiz son model | k_neighbors_classifier, k_neighbors_regressor: fit'te sample_weight yok |

Ensemble blanket engeli kaldırıldı; ağırlığı alan bütün alt/final modeller
rekürsif kontrol edilir. KNN içeren kombinasyon hata verir. Sadece `**kwargs`
kabul etmek destek sayılmaz. Ortak kontrol:
[`capabilities.py`](../modeling/capabilities.py).
XGBoost/LightGBM ilgili optional paketleri gerektirir.

## Preprocessing

| İşlem | Durum ve ağırlık davranışı |
| --- | --- |
| Simple/KNN/Iterative/Group imputation | Destekli; doldurulan satır ağırlığını korur; imputer hesabı ağırlıksızdır |
| İzinli scaling, encoding, feature selection/üretimi, cleaning | Destekli; satır ağırlığı modele taşınır |
| DropMissingRows / Deduplicate | Destekli; silinen satırın ağırlığı da silinir |
| IQR / ZScore / Winsorize / EllipticEnvelope / ManualBounds | Destekli; node'un gerçek satır seçimi ağırlığa uygulanır |
| LagFeatures / RollingAggregate | Destekli; sıralama ve satır düşürme X/y/ağırlığa birlikte uygulanır |
| ColumnFunction / FittedFunction | Satır koruma sözleşmesiyle destekli |
| RowFilterFunction | Filtre maskesi etiket ve ağırlığa da uygulanır |
| Özel calculator/applier | Açık ve doğrulanan `apply_with_row_mapping` sözleşmesi gerekir |
| KNNImputer → ağırlıklı Ridge vb. | Destekli; KNN burada son eğitim modeli değildir |

Kayıt listesi ve sözleşme: [`_weight_policy.py`](../preprocessing/_weight_policy.py).
DataFrame indeksi veya yalnızca aynı çıktı uzunluğu hizalama kanıtı sayılmaz.
"Window" isimli her özel dönüşüm otomatik desteklenmez; yerleşik zaman desteği
LagFeatures/RollingAggregate içindir. Inference'ın satır sayısı/sırası kuralları
ayrıca geçerlidir. Kolon birleştiren dallarda farklı satır seçimleri ve keyfi
özel eşleme desteklenmez; güvenli olmayan merge açık hata verir.

Özel hook `(transformed_data, positions)` döndürür: çıktıdaki her satır için
girdideki sıfır tabanlı konum. Core etiket/ağırlığı bu konumlardan alır.
Hatalı boyut, boolean/ondalıklı veya sınır dışı konumlar reddedilir.
Özel eğitim temsilini farklı üreten `fit_transform_train` otomatik kabul edilmez.

### Tam built-in kayıt listesi

Mevcut **67 kayıt adı** (alias'lar dahil) ağırlık sözleşmesine kabul edilir.
Bu, normal node gereksinimlerini kaldırmaz: geçerli kolonlar, modelin kabul ettiği
feature türleri ve optional bağımlılıklar yine gereklidir.

| Grup | Kayıt adları |
| --- | --- |
| Scaling | StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler |
| Imputation | SimpleImputer, KNNImputer, IterativeImputer, GroupImputer |
| Encoding | OneHotEncoder, OrdinalEncoder, LabelEncoder, DummyEncoder, HashEncoder, TargetEncoder, WOEEncoder |
| Vectorization | count_vectorizer, tfidf_vectorizer, hashing_vectorizer, sentence_embedder, tokenizer |
| Feature selection | feature_selection, VarianceThreshold, UnivariateSelection, ModelBasedSelection, CorrelationThreshold |
| Feature generation | FeatureGeneration, FeatureGenerationNode, FeatureMath, PolynomialFeatures, PolynomialFeaturesNode, FeatureInteraction |
| Binning | CustomBinning, GeneralBinning, KBinsDiscretizer |
| Dönüşüm | Casting, SimpleTransformation, GeneralTransformation, PowerTransformer |
| Geo | GeoDistance, H3Index |
| Zaman | DateFeatures, LagFeatures, RollingAggregate |
| Eksik/satır temizliği | DropMissingRows, DropMissingColumns, MissingIndicator, Deduplicate |
| Fonksiyon | ColumnFunction, FittedFunction, RowFilterFunction |
| Sampling | Oversampling, Undersampling |
| İnceleme | DataSnapshot, DatasetProfile: artifact üretir, asıl veriyi korur |
| Bölme | TrainTestSplitter, Split, feature_target_split |
| Outlier | ClipValues, IQR, ZScore, Winsorize, EllipticEnvelope, ManualBounds |
| Değer temizliği | TextCleaning, AliasReplacement, ValueReplacement, InvalidValueReplacement |

Son 18 kayıt için 38 yeni envanter testi eklendi. H3 gerçek `h3==4.5.0`
kütüphanesiyle sınandı; bu optional paket yerel test ortamına `uv pip` ile eklendi.
Sentence embedder'ın gerçek node fit/apply ve satır taşıması kontrollü encoder
ile test edildi; indirilen pretrained model inference'ı bu kontrolün kapsamı değildir.
Yeni alias desteği yalnızca aynı denetlenmiş calculator/applier sınıf çiftine
verilir; aynı isme custom sınıf kaydetmek desteği devralmaz.

## Sampling

| Yöntem | Ağırlık kuralı |
| --- | --- |
| random_over | Tekrarlanan kaynak satırın ağırlığı tekrar edilir |
| random_under_sampling, nearmiss, tomek_links, edited_nearest_neighbours | Gerçek sample_indices_ konumlarının ağırlığı alınır |
| smote, adasyn, borderline_smote, svm_smote, kmeans_smote | Mevcut ağırlıklar korunur; sentetik satırlara açık politika uygulanır |
| smote_tomek | Önce sentetik ağırlık atanır, sonra Tomek seçimi uygulanır |

Kullanıcının seçtiği kullanım:

İki ayrı ayar gerekir: `src/modeling/single_model.py` içinde
`WEIGHT_COLUMN = "importance"` gerçek satırların ağırlık kolonunu seçer.
`synthetic_weight` ise `src/features/preprocessing.py` içindeki
`build_preprocessing()` listesinin SMOTE adımının `params` bölümüne yazılır;
modelin `params` bölümüne yazılmaz ve kolon seçilince otomatik etkinleşmez.

```python
{"name": "balance", "transformer": "Oversampling", "params": {
    "method": "smote", "synthetic_weight": "class_mean", "random_state": 42
}}
```

`class_mean`, **o eğitim fold'undaki** aynı sınıfın kaynak satırlarının aritmetik
ortalama ağırlığını verir. Bu Skyulf'un açık politikasıdır; imbalanced-learn'ün
native sample-weight özelliği değildir. `uniform` alternatifi sentetik satıra 1
verir. Politika belirtilmeden sentetik sampling + ağırlık kullanımı hata verir.
Ağırlık mesafe hesabına feature olarak eklenmez. Sampling yalnızca eğitimde
uygulanır. Class-weight hesabı resampling sonrası gerçek fit etiketlerini kullanır.

| Kullanım | `synthetic_weight` gerekli mi? |
| --- | --- |
| Sample weight olmadan sampling | Hayır |
| Sample weight + random_over | Hayır; kaynak satırın ağırlığı kopyalanır |
| Sample weight + undersampling | Hayır; kalan satırların ağırlıkları seçilir |
| Sample weight + sentetik yöntemler | Evet; eksikse açık hata verilir |

[SMOTE API](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html).

## Bundle

Wizard önce sample-weight kullanılıp kullanılmayacağını sorar; varsayılan kapalı.
Açılırsa kolon adı ve uyumlu modeller gösterilir. Multi-target'ta her dal ayrı seçer.

| Dosya | Düzenlenecek yer |
| --- | --- |
| single_model.py | WEIGHT_COLUMN = None veya "importance" |
| model_competition.py | Bütün adaylar için ortak WEIGHT_COLUMN |
| multi_model.py | Her dalın workflow["weight_column"] ayarı |

Ayrı weights.py üretilmez/okunmaz. Model kaynağı ve seçilen ayar request'e
dondurulur; eğitim/scoring mutable dosyayı tekrar okumaz. Eski frozen request
evidence alanlarının isimleri uyumluluk için korunur.
Class_weight aynı modelin params bölümündedir; tuning'de base_model.params.
Kapalı sabit ayar varsayılan search-space tarafından yeniden açılmaz.

Ağırlık kolonu feature, target, kayıt anahtarı, group veya zaman kolonu olamaz.
Kolon belirtilip bulunamazsa hata verir. Değerler sayısal, sonlu, negatif olmayan
ve her gerçek fit'te pozitif toplamlı olmalıdır. Boolean/null/string, negatif/
sonsuz değerler, yanlış uzunluk ve sıfır toplam reddedilir.

## Ayrı kalan işler

- **Weighted imputation istatistikleri yok.** Normal imputation + ağırlıklı model çalışır.
- **Weighted metrikler yok.** CV/validation/test ve threshold hedefi ağırlıksızdır.
  f1_weighted sınıf support ortalamasıdır; kullanıcı satır ağırlığı değildir.
- **Canvas weight-column seçicisi yok.** Bu Bundle/Core işi yeni frontend seçicisi eklemez.
- **Keyfi özel dönüşüm otomatik desteklenmez.** Açık satır eşleme sözleşmesi gerekir.

## Doğrulama

**Güncel temiz tam Core koşusu — 2026-10-03:** 13.848 geçti, 662 atlandı,
0 başarısız / 0 hata; süreç çıkış kodu 0. Süre 28 dakika 39 saniye.
Satır ve dallanmayı birlikte ölçen toplam coverage **%90,01**: değişmeyen
**%90 CI eşiği geçti**. Eşiğin üzerindeki pay küçüktür.

Koşu yeni coverage dosyası, test geçici dizini ve cache ile tek süreçte yapıldı.
817 kaynak/test/yapılandırma dosyasının önce/sonra hash'leri aynı kaldı.
Bu doğrulama sırasında production kodu veya testler değiştirilmedi.
Ruff, format, tüm CI Ty kapsamı, CCN ve Bundle şema güncelliği de geçti.

662 atlama optional Spark/Delta ortamları, açıkça etkinleştirilen CLI kontrolleri
ve benchmark'ları içerir; bunlar geçmiş sayılmaz. Sonuç yerel Windows ortamında
CI Core kapsamına aittir; tüm depo CI işleri veya bütün optional ortamlar
çalıştırılmış değildir.
Tarihsel `core_weight_regression_result.json` kaydı örneklerle birlikte kaldırıldı.
Doğrulama kapsamı [ağırlıklı eğitim kılavuzunda](../../../docs/user_guide/weighted_training.md)
özetlenir; JSON kaydı artık dokümantasyonla dağıtılmıyor.
Log/XML: `.superpowers/sdd/WEIGHT_COLUMN_SIMPLE_V1_PLAN/core-clean-20261003/`.

Önceki %89,02 sonucu tarihsel kayıttır. O koşuda bulunan boosting `get_params()`
hatası ve import sorunları düzeltildikten sonra önce ilgili dosyalar 291 geçti /
3 atlandı; şimdi tam suite de yukarıdaki temiz koşuyla doğrulandı.

- Model katalog denemesi: 32 geçti / 2 beklenen KNN reddi.
- Sampling yeni + mevcut testler: **179 geçti**. Yeni 36 kontrol 11 yöntemin
  X/y çıktısını imbalanced-learn ile karşılaştırır; 58 gerçek pipeline kontrolü
  doğrudan, beş tuning yöntemi ve nested CV'de fit ağırlığını denetler.
- Ensemble ağırlık testleri: **200 geçti**, bunların içinde 108 routing senaryosu
  var. Ayrıca 86 ensemble/calibration ve 144 tuning regresyon testi geçti.
- Son ağırlık kapsamı birleşik kontrolü: **679 geçti, 3 atlandı**. Üç optional
  CLI testi ayrıca gerçek CLI ile geçti. API class-weight testleri: **24 geçti**;
  frontend Class Weight / payload testleri: **31 geçti**.
- Gerçek Databricks CLI generation/model-space kontrolleri: **150 geçti**;
  üç layout artifact kontrolü ayrıca **3 geçti**. Schema freshness doğrulandı.
- Satır eşleme / CV / imputation / envanter kapsamı: **218 geçti**; bu sayılar birleşik
  testlerle örtüşür, toplam test sayısı olarak toplanmamalıdır.
- Ruff, format (1.250 Python dosyası), tüm CI Ty kapsamı ve backend/Core CCN ≤ 10 geçti.
- Son wheel ayrı ortamda import, gerçek fit/predict ve save/load kontrollerini geçti.
- Frontend tam Vitest: **2.922 geçti**. Lint, complexity, TypeScript/build ve
  size-check geçti; frontend source bu genişletmede değiştirilmedi.
- Önceden bulunan iki private-import sınır ihlali public ortak helper adlarıyla
  düzeltildi; 101 ilgili test geçti. Kontrol devre dışı bırakılmadı.

### Canlı kabul

2026-10-03 ek kanıtı: aynı single-model SMOTE akışı ağırlıklı ve ağırlıksız
çalıştırıldı. İkisinde de 192 kaynak + 92 sentetik eğitim satırı, 48 holdout
tahmini doğrulandı. Gerçek fit vektörleri ve UC'den yüklenen modeller bağımsız
referansla eşleşti; MLflow CSV'leri indirilip ayrıca kontrol edildi.
[İki koşunun ayrıntılı kanıtı](../../../docs/user_guide/sampling_weight_acceptance.md).

`skyulf` profiliyle [final Databricks koşusu](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/596372253512145/run/990629522256246)
ve task `606948115207627` başarılı. Test şeması:
`workspace.skyulf_weight_extended_20261002_626f525c`.
Wheel SHA256: `c7233a63d8dfaf6f033aa79ee5b3410dcc0eb24c8067527c2a462fcd8d9fa1f2`.

- SMOTE class_mean + voting/stacking/calibrated classifier: her biri 168 eğitim
  satırı ve 120 tahmin; bağımsız sklearn referansı ve save/load eşleşti.
- Bundle single-model kaynak dosyasından iki eğitim: class_weight balanced +
  row_weight, ardından class_weight None + alternate_weight. Her biri 192 eğitim,
  48 scoring satırı; iki UC model version ve toplam 96 Delta tahmini doğrulandı.
- Dondurulan model dosyası sonradan bozulmasına rağmen yakalanmış ayarlar kullanıldı;
  feature/label satırları değişmedi, scoring ağırlık kolonu istemedi, alias'lar boş kaldı.
- Canlı kayıt/promotion senaryosu single-model içindir; competition/multi-target
  üretim ve ayar sözleşmeleri yerel testlerle doğrulandı. Bütün layout'lar için
  uçtan uca cloud lifecycle çalıştı iddiası yoktur.

Kalıcı testler:
[`ensemble`](../../tests/unit/test_ensemble_sample_weight_routing.py),
[`satır eşleme`](../../tests/integration/test_weighted_row_mapping.py),
[`imputation`](../../tests/integration/test_weighted_imputation_contract.py),
[`preprocessing envanteri`](../../tests/integration/test_weighted_preprocessing_inventory.py),
[`sampling`](../../tests/unit/test_weighted_resampling.py),
[`sampling + eğitim`](../../tests/integration/test_weighted_resampling_training.py).
Yerel ayrıntılı loglar `.superpowers/sdd/WEIGHT_COLUMN_SIMPLE_V1_PLAN/` altındadır.
