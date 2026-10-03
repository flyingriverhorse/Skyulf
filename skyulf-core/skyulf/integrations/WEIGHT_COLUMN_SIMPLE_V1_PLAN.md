# Weight Column — Sade V1 Uygulama Planı

> **Durum:** Uygulama ve canlı Databricks kabulü tamamlandı. Genel Core CI kapısı tam yeşil değil; açık kontrol sonuçları bölüm 6'da. 2026-10-02, `091`. Commit/push yapılmadı.
> **Uygulayan agent:** `executing-plans` ile adımları sırayla uygula;
> yalnız çalıştırılmış kontrollerin sonuçlarını tamamlandı olarak işaretle.

**Amaç:** Mevcut model eğitim yoluna kullanıcı satır ağırlığını eklemek;
hem `class_weight` hem `weight_column` ayarını Bundle'dan düzenlenebilir tutmak.

**Mimari:** Ortak `sample_weight` doğrulama/iletim yolu kullanılır. Model başına
ayrı ağırlık sistemi kurulmaz; native sınıf ağırlığı davranışı korunur.
Split/CV indeksleri ağırlığa da uygulanır. Genel satır kökeni altyapısı ve
sentetik örnek ağırlıklandırması ilk teslimatın dışında kalır.

**Teknolojiler:** Python, mevcut pandas/Polars Core yolları, sklearn,
XGBoost/LightGBM, Databricks Bundle ve MLflow.

**Tasarım kaynağı:** Bu konuşmada kabul edilen sade yaklaşım. Önceki
[plan](WEIGHT_COLUMN_PLAN.md) ve
[geniş V1 planı](WEIGHT_COLUMN_V1_IMPLEMENTATION_PLAN.md) tarihsel referanstır.
Sade V1 için bu dosya önceliklidir: geniş plandaki SMOTE/RowLineage işleri
bu sürümün tamamlanma şartı değildir.

## 1. Terimler ve bugünkü kontrol noktaları

`class_weight`, sınıf etiketine göre önem verir; `sample_weight`, her satıra
ayrı önem verir. `weight_column`, bu satır ağırlığının kaynak kolonu seçimidir.
Satır ağırlığı sınıf ağırlığına dönüştürülmez; regresyonda da kullanılabilir.

| Yer | Bugünkü durum | Sade V1 |
|---|---|---|
| Core model config | `params.class_weight` mevcut; destek modele bağlı | Koru; kullanıcı `sample_weight` girdisini ekle |
| Frontend | RF/XGBoost/LightGBM classifier tanımlarında `Class Weight`: None/Balanced | Mevcut kontrolü koru ve gerçek payload geçişini doğrula |
| Bundle | Model Python dosyalarının `params` alanı düzenlenebilir | `class_weight` burada kalır; açıklamalı örnek ekle |
| Bundle satır ağırlığı | Henüz yok | Yeni `src/modeling/weights.py` |
| Scoring | Kayıtlı modeli kullanır | Ağırlık kolonu istemez; mutable ayar dosyasını okumaz |

Kanıt yolları: `modeling/_class_weights.py`, `modeling/sklearn_wrapper.py`,
`modeling/hyperparameters/_tree.py`,
`backend/ml_pipeline/_internal/_routers/meta.py`,
`frontend/ml-canvas/src/modules/nodes/modeling/trainingSettings/ParameterSections.tsx`.
Frontend bilgisi kod incelemesine dayanır; tarayıcıda uçtan uca doğrulanmadı.
2026-10-02'de `test_hyperparameters_class_weight.py` ve
`test_modeling_sklearn_wrapper.py`: **22 passed**. Yeni özellik kanıtı değildir.

## 2. Kullanıcı Bundle'da neyi değiştirecek?

### Sınıf ağırlığı: mevcut model dosyası

| Layout | Düzenlenen dosya ve alan |
|---|---|
| Single | `src/modeling/single_model.py`: `MODELING` |
| Competition | `src/modeling/model_competition.py`: adayın `modeling` alanı |
| Multi-target/model-set | `src/modeling/multi_model.py`: branch'in `workflow.pipeline.modeling` alanı |

Doğrudan modelde `params`; tuner varsa `base_model.params` düzenlenir:

```python
# Mevcut model parametrelerine eklenir; diğer parametreler korunur.
"params": {"class_weight": "balanced"}
# Kapatmak için gerçek Python None kullanılır.
"params": {"class_weight": None}
```

`class_weight` için ikinci bir ayar kaynağı oluşturulmaz. Tuning search_space
aynı alanı arıyorsa sabit değerle çakışma mevcut kuralla reddedilir; kullanıcı
sabit ayar istediğinde o arama eksenini kaldırır. Regresyona class_weight eklenmez.
Core'un desteklediği özel sınıf-ağırlık sözlükleri korunur; Bundle hook'ları
JSON anahtarlarını string ister. Sayısal etiketli sözlükler sessiz string
dönüşümüyle desteklenmiş sayılmaz; V1 Bundle örnekleri None/Balanced kullanır.

### Satır ağırlığı: yeni dosya

Üç layout için de oluşturulacak `src/modeling/weights.py`:

```python
DEFAULT_WEIGHTS = {"weight_column": None}
MODEL_WEIGHTS = {}
```

Kullanıcı kaynak tablosuna gerçek sayıları koyar; dosyada kolonu seçer:

```python
DEFAULT_WEIGHTS = {"weight_column": "training_weight"}
MODEL_WEIGHTS = {
    "revenue": {"weight_column": "revenue_weight"},
    "risk": {"weight_column": None},
}
```

- `MODEL_WEIGHTS` yalnız multi-target branch adlarını kabul eder; single ve
  competition'da boş olmalıdır. Competition adayları ortak satır ağırlığını
  kullanır; kendi model dosyalarında ayrı class_weight seçebilirler.
- Branch override varsayılanı ezer; açık None o branch için kapatır.
- Bilinmeyen branch/ayar, boş kolon adı veya rol çakışması açık hatadır.
- Bütün tanımlı weight kolonları bütün branch feature listelerinden dışlanır;
  kullanıcı açıkça feature seçmişse sessiz değişiklik yerine hata verilir.
- Yeni wizard sorusu gerekmez. Eski projede dosya yoksa ağırlıksız devam edilir.
- Her iki ayar da sonraki manuel/otomatik eğitime uygulanır. Kaydedilmiş model
  değişmez; deploy edilmiş job için düzenlenen proje kaynakları senkronize edilir.

## 3. Ortak eğitim sözleşmesi ve kapsam

```text
Kaynak snapshot + ağırlık kolonu
  -> mevcut güvenli pre-split hazırlık / branch etiket filtresi
  -> aynı satırları train/holdout'a böl
  -> weight kolonunu ayır: X, y, w
  -> satır koruyan preprocessing
  -> model.fit(X, y, sample_weight=w)
```

- Public Core: `SkyulfPipeline.fit(data, target_column, *, sample_weight=None)`.
  Vektör ham girdi sırasındadır; Core'a verilen feature tablosu weight kolonunu
  içermez. Bundle kolon ayrımını Core çağrısından önce yapar.
- `SplitDataset` için ayrı `train_sample_weight=None` alanı eklenir ve kopyalanır.
  Bu girdiyle ayrıca public sample_weight vermek belirsiz girdi hatasıdır.
  Test/validation ağırlığı V1'de taşınmaz; bu bölümlerin metrikleri ağırlıksızdır.
- Raw split ve random/temporal/group/nested CV, X/y için kullanılan gerçek
  indeks/permutation üzerinden w'yu da taşır. DataFrame index'ine güvenilmez.
- Split sonrası filtre, deduplicate, reorder, lag/rolling ve resampling weighted
  V1'de reddedilir. Yalnız mevcut sözleşmesi satır sayısı/sırasını koruyan
  built-in dönüşümler kabul edilir; bilinmeyen custom adımlar reddedilir.
  Pre-split filtre yalnız w aynı kaynak satırındayken yapılabilir; w'ya yazamaz.
- Ağırlık sonlu, sayısal, tek boyutlu, uzunluğu doğru ve negatif olmayan olmalı;
  bool/null reddedilir. Her gerçek fit altkümesinin toplamı pozitif olmalıdır.
  Otomatik normalizasyon yapılmaz. Birleştirilmiş dış ağırlık tekrar doğrulanır.
- Native class_weight destekleyen modele yalnız kullanıcı w gönderilir.
  Native destek yoksa mevcut helper, o fit'in y'sinden sınıf ağırlığı üretip
  kullanıcı w ile bir kez çarpar. Native `balanced` hesabı yeniden yazılmaz;
  modellerin weighted-frequency davranışının aynı olduğu iddia edilmez.
- İmzada `**kwargs` bulunması destek kanıtı değildir. Desteklemeyen model veya
  ağırlığı bütün alt fit'lere ilettiği kanıtlanmamış ensemble/calibration birleşimi
  eğitim başlamadan açık hata verir; sessiz ağırlıksız fallback yapılmaz.
- Model seçim skoru, holdout, threshold ve early stopping validation ağırlıksız
  kalır. Scaler/imputer istatistiklerinin de ağırlıklı olduğu iddia edilmez.
- SMOTE ayrı takip işidir: gerçek ebeveyn/lambda üzerinden sentetik ağırlık
  politikası ve sürüm uyumluluğu kanıtlanmadan açılmaz. Weighted V1'de bütün
  sampler'lar reddedilir; ağırlıksız mevcut sampler davranışı korunur.

## 4. Uygulama adımları

Her adımda davranış testi önce başarısız görülür; minimal değişiklikten sonra
aynı test ve etkilenen mevcut testler çalıştırılır. Aşağıdaki yollar repo köküne
göredir; kısa Core yolları `skyulf-core/skyulf/` altındadır.

### S1 — Ortak ağırlık helper'ı ve doğrudan model fit

Dosyalar: yeni `modeling/_sample_weights.py`; mevcut `modeling/_class_weights.py`,
`base.py`, `sklearn_wrapper.py`, `classification.py`, `regression.py`, `ensemble.py`.
Yeni test: `skyulf-core/tests/unit/test_sample_weights.py`.

Arayüz: `validate_sample_weight(values, expected_rows)` -> float vektör veya None;
`sample_weight_for_fit(model, class_weight, y, sample_weight=None)` -> fit vektörü.
Calculator fit metodları geriye uyumlu keyword `sample_weight=None` kabul eder.

- [x] Geçersiz değer/boyut, kullanıcı girdisinin değişmemesi ve None testlerini yaz.
- [x] Doğrudan weighted LinearRegression sonucunu sklearn referansıyla karşılaştır.
- [x] Native class_weight çift uygulanmasın; non-native yalnız bir kez birleşsin.
- [x] Desteksiz model ve birleşim sonrası sıfır/sonsuz ağırlık hatalarını doğrula.

S1 kanıtı: 55 yeni ağırlık testi; ilgili mevcut testlerle 310 passed.
Ruff/format/CCN ve entegrasyon sonundaki tam CI Ty kapsamı geçti. Sonlu Decimal
girdiler destekleniyor; kullanıcı satır ağırlığı V1'de classification/regression
için açık. Yerel test, bağımsız inceleme, canlı kabul ve geniş Core suite
sonuçları bölüm 6'da kaydedildi.

### S2 — Pipeline ve bütün CV/tuning fit yolları

Dosyalar: `data/dataset.py`, `pipeline/_pipeline.py`, `preprocessing/base.py`,
`preprocessing/split.py`, `preprocessing/pipeline.py`, `preprocessing/fold_adapter.py`,
`modeling/cross_validation.py`; `modeling/_tuning/` altında `engine.py`,
`cv_policy.py`, `nested.py`, `grid_random.py`, `fold_pipeline.py`, `refit.py`,
`nested_threshold.py`, `strategies/runner.py`, `strategies/halving.py`,
`strategies/optuna_search.py`, `strategies/optuna_folds.py`.
Kod incelemesinde ayrıca `modeling/_policy_cv.py` ve `strategies/optuna.py`
yollarının da ağırlık iletimine dahil edilmesi gerektiği doğrulandı.
Yeni test: `skyulf-core/tests/integration/test_simple_weighted_training.py`.

Arayüz: S1 sample_weight girdisi; raw fit vektörü veya SplitDataset train alanı.
Fold adapter'larında w değiştirilmez, çünkü satır değiştiren adımlar reddedilir.

- [x] Satır kimliği kaydeden estimator ile split/CV/final refit w eşleşmesini kanıtla.
- [x] Temporal sıralama, group/nested CV, halving altkümesi ve Optuna'yı ayrı sına.
- [x] Aynı uzunlukta yanlış sıralanmış w'yu yakalayan test ve sıfır toplam fold ekle.
- [x] Raw input ve SplitDataset yollarını, kopyalamayı ve pandas/Polars eşitliğini sına.
- [x] Filtre/reorder/SMOTE/custom adım reddini ve ağırlıksız eski davranışı doğrula.

### S3 — Bundle ayarları, üç layout ve yeniden eğitim

Dosyalar: `integrations/databricks/` altında yeni `weight_config.py`; mevcut
`project.py`, `_project_files.py`, `workflow_config.py`, `local_workflow.py`,
`local_retraining.py`, `local_pre_split.py`, `local_batch.py`, `local_cv.py`,
`local_search.py`, `competition_project.py`, `competition_training.py`,
`local_competition.py`, `local_branches.py`, `model_set_project.py`.
Multi-target yükleyicisi `branch_notebook.py`, competition fold eğitimleri
`competition_evaluation.py` ve arama sonrası CV `local_search_results.py`
üzerinden de geçtiği için bu yollar da kapsamdadır.
Yeni test: `skyulf-core/tests/integrations/test_simple_bundle_weights.py`.

Arayüz: `resolve_weight_config(defaults, overrides, branch_name)` ->
`{"weight_column": str | None}`. Çözüm trusted loader üzerinden bir kez yapılır.

- [x] Eksik dosya, varsayılan, override, açık None ve yanlış ayar testlerini yaz.
- [x] Single/competition/multi-target için hem class_weight hem weight_column değişsin.
- [x] Kaynak projection, pre-split koruma, branch etiket maskesi ve weight ayrımını sına.
- [x] Otomatik retraining aynı eğitim yolunu kullansın; yalnız w değişmesi mevcut
  yeni-etiketli-veri kontrolünü geçirmesin. Manuel train değişen w'yu kullansın.
- [x] Yeni pasif alanlar eski `LocalTrainingSpec.dataset_id` hash'ini değiştirmesin.

### S4 — Template, artifact ve scoring

Dosyalar: template `src/modeling/weights.py` (yeni), mevcut `single_model.py.tmpl`,
`model_competition.py.tmpl`, `multi_model.py.tmpl`, `README.md.tmpl`;
Core `inference/local_pipeline.py`, Databricks `training_parameters.py`,
`local_training_evidence.py`, `retraining_data.py`, `monitoring_reference.py`.
Testler: yeni `skyulf-core/tests/integrations/test_simple_weight_artifacts.py`;
mevcut `test_databricks_bundle_generation.py`, `test_competition_template.py`,
`test_databricks_branch_template.py` (`skyulf-core/tests/integrations/` altında).

- [x] Üç layout'ta weights.py oluşsun; class_weight düzenleme yolu yorumlarla gösterilsin.
- [x] Çözülmüş class_weight/weight_column, kaynak sürümü ve w özeti artifact'e yazılsın.
  Weight kaynağının digest'i ve eğitim satırlarıyla eşlenmiş ağırlık digest'i kaydedilsin.
- [x] Eğitim başladıktan sonra ayar dosyasındaki değişiklik o fit/replay'i etkilemesin.
- [x] Save/load sonrası weight kolonsuz scoring ve eski artifacts uyumluluğunu sına.
- [x] Ağırlık feature/schema/SHAP/drift girdisine sızmasın; console, parametre ve
  metrik loglarına ham ağırlık yazma. Mevcut tam eğitim snapshot artifact'inde
  ağırlık kolonu, diğer kaynak kolonlarıyla aynı erişim kuralları altında korunur;
  bu, aynı eğitimin sonradan tekrar üretilebilmesi içindir.

### S5 — Mevcut class_weight kontrolü ve kabul

Dosyalar: `modeling/hyperparameters/_tree.py`, `_registry.py`;
frontend `TrainingSettings.test.tsx`, `trainingSettings/ParameterSections.tsx`,
`core/utils/pipelineConversion/training.test.ts`; backend hyperparameter meta router.
Üretim frontend değişikliği yalnız mevcut kontrolün doğrulaması gerçek hata bulursa yapılır.

- [x] Core'daki destekli modellerin None/Balanced seçeneklerini API üzerinden doğrula.
- [x] Frontend Customize -> Class Weight -> Balanced -> None geçişini payload'a
  kadar sına. HTML select stringleri gerçek None yerine geçip eğitim bozmasın.
- [x] Bundle model dosyasında yapılan class_weight değişikliği gerçek fit'e ulaşsın.
- [x] Yeni weight_column frontend ekranı ekleme; bu özellik Core/Bundle kapsamındadır.
- [x] Üç layout x random/temporal için küçük yerel eğitim/scoring kabulünü çalıştır.
  Ayrı Databricks test kaynaklarında temsilî weighted eğitim + retraining + scoring
  doğrulanmadan cloud desteğini doğrulanmış olarak sunma.

## 5. Doğrulama ve kapsam sınırı

İlk test komutu (S1 test dosyası oluşturulduktan sonra):

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests/unit/test_sample_weights.py -q -o addopts=
```

Entegrasyon sonunda `.github/workflows/skyulf-core-tests.yml` ve `pr_check.yml`
komutlarıyla Core suite/collection, Ruff, tam Ty kapsamı, CCN <= 10 ve template
schema kontrolü çalıştırılır. Frontend değişirse etkilenen testler, lint,
complexity:check, build, size-check ve generated assets zorunludur.
Bağımlılık değişirse `uv`, root manifestler, Core setup.py ve uv.lock birlikte
ele alınır. Yeni Python kodu modern anotasyonlar ve `from __future__ import annotations`
kullanır; yeni testlerin docstring ve gerçek assertion'ı olur.

Otomatik recency üretimi, ağırlıklı metrik UI'si, genel RowLineage, weighted
SMOTE, Spark-native eğitim ve retraining isteği ile sonraki job başlangıcı
arasında yeni bir snapshot kilitleme protokolü bu sade teslimata dahil değildir.
Mevcut fresh-data/cooldown/approval kuralları korunur; yeni garanti iddia edilmez.

Commit/push yalnız ayrıca istenirse yapılır. Tamamlanma durumu, işaretli
adımlar ve aşağıdaki çalıştırılmış doğrulama sonuçlarıyla birlikte okunmalıdır.

## 6. Uygulama ve kabul kanıtı — 2026-10-02

- S1: 55 ağırlık testi ve ilgili 310 test geçti.
- S2: split/pipeline tarafında 128 test; CV/tuning tarafında 57 yeni test ve
  ilgili 549 test geçti. Grid/random/halving/Optuna, nested ve threshold refit
  çağrılarında gerçek satır/ağırlık eşleşmesi sınandı.
- S3: 45 loader testi; 31 yeni Bundle aktarım testi ve ilgili 161/133/152 testlik
  gruplar geçti. Gruplar örtüşür; bu sayılar benzersiz toplam değildir.
- S4: 7 artifact/replay testi, üç gerçek CLI layout üretimi ve ilgili 49 test
  geçti. Gerçek Parquet snapshot, kayıtlı model yükleme, SHAP girdisi ve
  monitoring replay yolu kontrol edildi.
- S5: 2.922 frontend testi, 24 API testi, ilgili bir Playwright testi; lint,
  complexity, TypeScript, build ve size-check geçti. Generated assets yenilendi.
- Tam Ruff, CI Ty kapsamı, Python format ve backend/Core CCN ≤ 10 geçti.
  Template schema/model spaces güncel. Wheel, checkout dışında ayrı ortamda
  import/eğitim/predict/save-load doğrulamasını geçti.
- Son birleşik özellik/API/regresyon grubu: **427 passed, 3 skipped**.
  Atlanan üç CLI testi ayrı gerçek CLI çalıştırmasında geçti.
- Geniş Core ilk çalıştırması: **13.399 passed, 657 skipped, 16 failed**,
  28 dakika 12 saniye; branch coverage **%89,63**, zorunlu alt sınır **%90**.
  Sonradan eklenen testlerle toplama kontrolü 14.084 test buldu. Tam geniş
  paket son düzeltmelerden sonra yeniden çalıştırılmadı; ilgili gruplar tekrarlandı.
- İlk 16 hatanın 15'i tekrarda geçti: sandbox ağ engeline takılan mevcut
  embedding testi ağ erişimiyle geçti; eski template test yardımcısı düzeltildi
  ve dosyanın 64 testi geçti; eski dataset-id referansı güncellendi ve ilgili
  97 test geçti. Production dataset-id değeri HEAD ile birebir aynı kaldı.
- **Açık genel kontrol:** `test_internal_api_boundary.py` içindeki bir test,
  `monitoring_model_set.py` → `_enrollment_config` ve `retraining_requests.py`
  → `_merge_with_retry` import'larını reddediyor. Test ve iki production dosyası
  HEAD ile aynı; bu ağırlık değişikliğinden önce de bulunan sınır ihlali bu
  kapsamda değiştirilmedi. Coverage kapısı da geçmedi. Genel CI sonucu bu
  nedenle başarılı olarak sunulmuyor; coverage eşiği düşürülmedi.

### Canlı Databricks

Profil: `skyulf`. [Başarılı kabul run'ı](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/560937412778564/run/400910598707257).
Task run: `235857202607356`. Ayrı test şeması:
`workspace.skyulf_weight_v1_20261002_3307ad5e`.

| Eğitim | class_weight | weight_column | Train | Scoring |
|---|---|---|---:|---:|
| Model v1 | balanced | row_weight | 192 | 48 |
| Manuel yeniden eğitim v2 | None | alternate_weight | 192 | 48 |

İki kayıtlı modelin katsayıları ve tahminleri sklearn referansıyla eşleşti.
Kaydedilen UC artifact'leri ağırlık kolonu olmadan scoring yaptı; predictions
Delta tablosunda toplam 96 satır doğrulandı. Ağırlık kaynak/dizi digest'leri
farklı, feature/label satırları aynı kaldı. Eğitimden önce mutable weights.py
bozulmasına rağmen yakalanmış ayarlar kullanıldı. Model alias'ları boş kaldı.
Bu canlı kabul single-model ve manuel yeniden eğitimi kapsar; üç layout ve
otomatik retraining yönlendirmesi yerel testlerle doğrulandı.

İlk kabul denemesinde test experiment adı workspace klasörüyle çakıştı;
notebook'taki test adı düzeltildi. Üretim kodu bu nedenle değiştirilmedi.
Test kaynakları inceleme için korundu; mevcut uygulama kaynaklarına dokunulmadı.

### Uygulamada netleştirilen kararlar

1. Plandaki kısa dosya listesine gerçek `_policy_cv`, Optuna, branch loader,
   competition evaluation ve post-selection CV yolları eklendi. Aksi halde
   bazı fit'lerde ağırlık kaybolurdu.
2. Ham ağırlıklar mevcut kontrollü source/train Parquet snapshot'ında kalır;
   console/parametre/metrik özetlerine yazılmaz. Bu karar yanlışsa aynı veri
   artifact'inde ek ağırlık değerleri saklanmış olur; replay için gereklidir.
3. Sonlu Decimal kaynak değerleri kabul edilir. Ek bir sayısal scalar türünü
   destekler; otomatik normalizasyon veya sessiz string dönüşümü yapmaz.
4. TuningCalculator ham ağırlık vektörünü saklamaz. Yalnız değerlendirme için
   üretilen özel post-fit dataset ağırlıksızdır; gerçek fit'ler hizalı vektörü
   alır. Gelecekte bu özel dataset eğitimde kullanılacaksa açık aktarım gerekir.
5. Eski template test yardımcısı mevcut rolling-days ifadelerine uyarlandı;
   production template davranışı değişmedi. Gerçek üç-layout CLI üretimi ve
   bağımsız inceleme, test düzeltmesinin hata gizlemesi riskine karşı kontrol oldu.
6. İlgisiz mevcut private import sınır ihlali kapsam dışı bırakıldı. Bedeli:
   genel Core kapısı açık kalır; bu durum feature kabulünden ayrı raporlandı.

## 7. Kullanıcı incelemesi sonrası destek denetimi ve takip işleri

Güncel kombinasyonlar, KNN model / KNN imputation ayrımı, gerçek ensemble
routing kontrolleri ve ayrı kalan işler
[destek matrisinde](WEIGHT_COLUMN_SUPPORT_MATRIX.md) tutulur.

İlk V1 denetiminde ensemble desteği ve model dosyalarındaki ağırlık ayarları
takip işiydi; bölüm 8 bu işleri uygular. Imputation kontrolleri normal
imputation + ağırlıklı model eğitimini doğrular. Ağırlıklı imputation
istatistikleri ve ağırlıklı değerlendirme metrikleri ayrı davranışlardır.

## 8. Genişletme çalışması — 2026-10-02

Kullanıcının onayıyla mevcut `091` dalında yürütülür. Mevcut staged değişiklikler
korunur; commit ve push yapılmaz.

- [x] Ensemble: gerçek alt modellerin ve stacking final modelinin ağırlık
  desteğini kontrol ederek voting, stacking ve calibration yollarını aç.
  Doğrudan eğitim, CV, beş tuning yöntemi ve nested/refit yollarını doğrula.
- [x] Ortak model yetenek kontrolü: Bundle seçimi ve runtime aynı sözleşmeyi
  kullansın; KNN gibi desteklemeyen eğitim modelleri açık hata versin.
- [x] Bundle: isteğe bağlı ağırlık sorusunu model seçiminden önce sor;
  ayarları single/competition model dosyasına ve multi-model workflow'una taşı.
  Üretilen `weights.py` dosyasını kaldır; donmuş ayarlarla replay'i koru.
- [x] Preprocessing: satır silme, sıralama ve çoğaltmada açık konumsal eşleme
  kullan; X/y/ağırlık birlikte değişsin. Özel fonksiyonlar açık ve doğrulanan
  bir satır eşleme sözleşmesiyle çalışsın.
- [x] Resampling: mevcut satırları seçen/tekrarlayan yöntemlerde gerçek sampler
  indekslerini kullan. Sentetik satırlara verilecek ağırlığı açık bir ayarla
  tanımla; ağırlığı hiçbir zaman mesafe hesabına giren feature olarak ekleme.
- [x] Birleşik doğrulama çalıştırıldı: etkilenen testler, tüm Core suite, Ruff, CI Ty scope,
  CCN kapısı, paket ve mümkün olan gerçek Bundle kabul yolları. Başarısız veya
  çalıştırılmayan kontrolleri destek matrisinde ayrı göster.

Güncel doğrulama (2026-10-03): son kodla temiz tam Core koşusu **13.848 geçti,
662 atlandı, 0 başarısız/hata**, çıkış kodu 0; toplam coverage **%90,01** ile
değişmeyen %90 kapısı geçti. 817 dosyanın koşu öncesi/sonrası hash'leri aynı.
Bu sonuç önceki %89,02 coverage ve yalnızca başarısız dosyaların yeniden
çalıştırılmasıyla sınırlı kaydın yerini alır. Ruff, format, CI Ty, CCN ve schema
freshness de geçti; optional atlamalar ve yerel Windows kapsamı destek matrisinde
açıklanır. Bu, tüm depo CI işleri için başarı iddiası değildir.
Önceki bölüm 6'daki private-import ihlalleri bu genişletmede düzeltildi.
Son wheel ile canlı Databricks kabulü ve ayrı ortam paket kontrolü de geçti.

İstatistikleri ağırlıklı öğrenen imputation ile ağırlıklı değerlendirme
metrikleri, eğitim ağırlığını taşıma sözleşmesinden ayrıdır; bu genişletme
eğitim davranışını düzeltirken mevcut metrik anlamını sessizce değiştirmez.

Kullanıcı sentetik örnekler için açık `synthetic_weight="class_mean"`
politikasını seçti. Güncel destek matrisi önceki V1 sınırlarının yerini alır;
eski bölüm 6/7 kayıtları o tarihte doğrulanan kapsamı anlatır.

## Kaynaklar

- [sklearn sample_weight ve sınıf ağırlığı](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [CV fit parametrelerinin dilimlenmesi](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [AutoGluon kolon seçimi ve ayrı değerlendirme ayarı](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)
- [SMOTE public sözleşmesi](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
