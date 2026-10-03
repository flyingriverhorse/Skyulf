# Weight Column V1 Implementation Plan

> **For agentic workers:** `executing-plans` skill'ini kullanarak görevleri
> sırayla uygulayın. Tamamlanan adımları kanıtlarıyla işaretleyin. Bu planın
> yazılması, aşağıdaki davranışların uygulanmış veya test edilmiş olduğu
> anlamına gelmez.

**Goal:** Bundle/Core eğitim akışlarında satır ağırlığını single, competition
ve model-set için preprocessing, CV, tuning ve drift sonrası retraining boyunca
doğru satırla birlikte taşımak.

**Architecture:** Ağırlık, feature tablosundan ayrı satır metadata'sıdır.
Filtreler, sıralamalar ve desteklenen sampler'lar bu metadata'yı açık satır
eşlemeleriyle taşır. Model fit aşamasında `sample_weight` kullanılır; inference
sözleşmesine ağırlık kolonu eklenmez.

**Tech Stack:** Python, mevcut Skyulf Core/Databricks entegrasyonu,
scikit-learn, imbalanced-learn, MLflow, pytest, Databricks Bundles.

**Spec:** [WEIGHT_COLUMN_PLAN.md](WEIGHT_COLUMN_PLAN.md) ve aşağıdaki kapsam
güncellemeleri. Bu dosya eski tasarıma eşlik eden uygulama sırasıdır.

**Baseline:** 2026-10-02, branch `091`, commit `47f778bf`.

## Global Constraints

- V1 kapsamı Core ve Bundle'dır. Backend/Canvas üzerinden ağırlık ayarı eklenmez.
- Bundle oluştururken yeni soru sorulmaz. Düzenlenebilir varsayılan dosya gelir.
- Eski projeler ve `weight_column=None` kullanan projeler mevcut davranışı korur.
- Ağırlık kolonu feature, model imzası veya zorunlu scoring girdisi olamaz.
- Ağırlıklar sonlu sayısal değerlerdir; bool, null ve negatif değer reddedilir.
  Her gerçek fit girdisinde toplam ağırlık pozitif olmalıdır.
- Sessizce ağırlıksız eğitime dönülmez. Desteklenmeyen model/node açık hata verir.
- Eğitim ve doğrulama ayrımı korunur; resampling yalnızca eğitim tarafındadır.
- Metrikler mevcut Skyulf metric katmanından alınır; paralel hesaplama yazılmaz.
- Yeni Python koduna `from __future__ import annotations` eklenmez.
- Testlerde docstring ve davranışı doğrulayan assertion bulunur. CCN sınırı 10'dur.
- Bağımlılık gerekirse `uv` kullanılır; manifestler ve lock dosyası birlikte
  güncellenir. Yerelde kurulu paket CI desteği sayılmaz.
- Commit ve push ayrıca istendiğinde yapılır. Geçici cloud kanıtları commit'e girmez.

## 1. Eski plandan değişen kararlar

| Eski öneri | Bu uygulama planındaki karar |
|---|---|
| Drift retraining gelene kadar ertele | Mevcut retraining akışına bağlanacak |
| Bundle wizard'da weight sorusu | `src/modeling/weights.py` varsayılan gelir |
| V1'de satır sayısı değişirse reddet | Açık satır eşlemesi olan filtre/resampling desteklenir |
| SMOTE sonraki sürümde | Standart SMOTE, aşağıdaki ağırlık politikasıyla V1'de |
| Ağırlığı korunan X kolonu olarak taşı | X dışında metadata olarak taşı |
| Genel olarak sıfır ağırlık = satırı silmek | Bu eşitlik tüm preprocessing zinciri için iddia edilmez |

Otomatik zamana bağlı ağırlık üretimi, ağırlıklı değerlendirme metrikleri ve
Canvas ayar ekranı sonraki kapsamdır. Bunlar V1 tamamlandı diye sunulmayacak.

## 2. Ağırlık başka nerede kullanılır?

| Aşama | V1 davranışı |
|---|---|
| Eğitim tablosunu/penceresini seçme | Mevcut veri seçimi sürer; ağırlık veri seçmez |
| Random veya temporal split | Mevcut bölme sürer; w aynı satırlarla bölünür/sıralanır |
| Imputer, scaler ve feature üretimi | Öğrenilen istatistikler ağırlıksız; w feature değildir |
| Filtre, deduplicate, sıralama | Kalan/taşınan satırların ağırlığı birlikte taşınır |
| Oversampling/undersampling/SMOTE | Bölüm 4'teki açık eşleme uygulanır |
| Doğrudan model fit | `sample_weight` kullanılır |
| Her CV/tuning training fold'u | Yalnız o fold'un eğitim ağırlıkları kullanılır |
| Seçilen parametrelerle final refit | Gerçek final eğitim satırlarının ağırlıkları kullanılır |
| Nested threshold eğitimi | İç model fit'leri ağırlıklı; threshold seçimi ağırlıksız |
| Ensemble | Gerekli tüm fit'lere doğru yönlendirme kanıtlanırsa desteklenir |
| Early stopping validation metriği | V1'de ağırlıksız; eğitim ağırlığı validation'a sızmaz |
| Holdout, tuning skoru, champion/challenger | Mevcut ağırlıksız metrik ve terfi kuralları korunur |
| Scoring | Ağırlık kolonu gerekmez |
| Feature/prediction drift ve veri kalitesi | Ağırlıksız ölçülür; weight kolonu drift feature'ı değildir |
| Monitoring labeled performance | Gerçek sonucu bulunan kayıtlar üzerinden ağırlıksız |
| MLflow ve eğitim raporu | Politika, kolon, ağırlık özeti ve veri kimliği kaydedilir |

`class_weight` ile satır ağırlığı farklıdır. Native `class_weight` alan modelde
sınıf ağırlığını model uygular; dışarıdan tekrar çarpmayız. Native destek yoksa
mevcut sınıf ağırlığı yardımcı fonksiyonu kullanıcı ağırlığıyla bir kez
birleştirilir. `balanced` hesabı her fit'in gerçekten gördüğü etiketlere dayanır.

Sıfır ağırlıklı satır scaler/imputer istatistiklerini yine etkileyebilir.
Bu nedenle "satırı silmekle aynı" testi yalnızca uygun doğrudan estimator
örneğinde yapılır. Ağırlıkların ölçeği regularization etkisini değiştirebilir;
V1 otomatik normalize etmez ve tüm modellere aynı matematiksel eşitliği atfetmez.

## 3. Kullanıcının düzenleyeceği dosya

Yeni oluşturulan projede `src/modeling/weights.py` bulunacak:

```python
DEFAULT_WEIGHTS = {
    "weight_column": None,
    "smote_weight_policy": "interpolate",
}

MODEL_WEIGHTS = {}
```

Örneğin eğitim tablosunda `training_weight` kolonu varsa kullanıcı
`DEFAULT_WEIGHTS["weight_column"]` değerini `"training_weight"` yapar.
Single ve competition adayları ortak varsayılanı kullanır. Model-set için:

```python
DEFAULT_WEIGHTS = {
    "weight_column": None,
    "smote_weight_policy": "interpolate",
}

MODEL_WEIGHTS = {
    "revenue": {"weight_column": "revenue_weight"},
    "risk": {"weight_column": "risk_weight"},
}
```

- Anahtarlar model-set branch adlarıdır; bilinmeyen branch/ayar adı hata verir.
- Branch ayarı varsayılanı ezer; açık `None` o branch için ağırlığı kapatır.
- Gerçek `1`, `2`, `4` gibi ağırlık değerleri eğitim tablosundaki kolondan gelir.
  Bu dosya hangi kolonun kullanılacağını seçer; kendiliğinden değer üretmez.
- `smote_weight_policy` SMOTE'u açmaz. SMOTE preprocessing içinde seçilir.
- V1'de desteklenen SMOTE politikası `interpolate` olur; başka değer açık hatadır.
- Projede dosya yoksa geriye uyumlu olarak ağırlıksız devam edilir.
- Ayarlar trusted proje loader'ıyla okunur. Eğitim için çözülmüş JSON kopyası
  ve kaynak digest'i artifact'e yazılır; scoring mutable Python dosyasını okumaz.
- Yeni ayar sonraki eğitimi etkiler; mevcut kayıtlı modelin davranışını değiştirmez.
- Bütün branch'lerde tanımlı ağırlık kolonları feature seçiminin dışında tutulur.
  Böylece risk ağırlığı revenue modeline yanlışlıkla feature olmaz.

## 4. Preprocessing ve SMOTE sözleşmesi

Her satır değiştiren işlem, çıktısının hemen önceki girdide hangi satırlardan
geldiğini bildirir. Önerilen iç DTO `RowLineage(left, right, fraction)`:

```text
w_out[i] = (1 - fraction[i]) * w_in[left[i]]
           + fraction[i] * w_in[right[i]]
```

Filtre/sıralama/tekrarlama için `left == right` ve `fraction == 0` yeterlidir.
Birden fazla işlem varsa eşleme her adımda uygulanır. DataFrame index'i veya
çıktı uzunluğu tek başına satır kimliğinin kanıtı değildir.

| İşlem | Ağırlık politikası |
|---|---|
| Satır koruyan dönüşüm | Mevcut satır sırası ve w korunur |
| Filtre, deduplicate, outlier drop | Gerçek mask/indeks üzerinden kalan w alınır |
| Sıralama, temporal hazırlık | X, y ve w aynı permutation ile taşınır |
| Random undersampling | Sampler'ın gerçek `sample_indices_` çıktısı kullanılır |
| Random oversampling | Kopyalanan ebeveynin ağırlığı kopyalanır |
| Standart SMOTE | Gerçek ebeveyn/komşu ve aynı lambda ile w interpolate edilir |
| ADASYN, Borderline/SVM/KMeans SMOTE, SMOTETomek | Weighted V1'de açıkça reddedilir |
| Satır eşlemesi sağlamayan özel dönüşüm | Weighted modda açıkça reddedilir |

SMOTE örneği: ebeveyn ağırlıkları 1 ve 3, sentetik feature için lambda 0,25
ise yeni ağırlık 1,5 olur. Bu bir Skyulf politikasıdır; tüm iş/frekans ağırlığı
anlamları için zorunlu veya tek doğru tanım olduğu iddia edilmez.

Ağırlık kolonu SMOTE komşuluk mesafesine girmez. Ebeveynler sentetik feature
değerlerinden tahmin edilmez. Standart SMOTE adapter'ı gerçek üretim sırasında
ebeveyn indeksleri ve lambda'yı yakalar; ek RNG çekimi yapmaz.

Mevcut imbalanced-learn 0.14.1 uygulamasında bu bilgiler private üretim
metotlarındadır. Adapter tek modülde tutulmalı; desteklenen sürümler ve callback
sözleşmesi test edilmelidir. Gerekirse destek aralığı dependency manifestleri ve
lock ile açıkça sınırlandırılır. Geniş mevcut dependency aralığındaki her sürümün
uyumlu olduğu varsayılmaz. Uyumsuzluk sessiz yanlış eşlemeye dönüşemez.

Resampling her CV training fold'unun içinde yapılır. Validation/holdout/scoring
üzerinde yapılmaz. Mevcut merged-branch kısıtları sırf ağırlık ekleniyor diye
kaldırılmaz. Unweighted sampler davranışı korunur.

## 5. Drift ve retraining bağlantısı

```text
Scoring -> monitoring/drift -> mevcut retrain koşulu
  -> cooldown / çalışan job / tekrar isteği kontrolleri
  -> kullanılabilir yeni etiketli eğitim verisi kontrolü
  -> mevcut train job: seçilen veri + weights.py politikasının sabit kopyası
  -> weighted fit -> mevcut ağırlıksız değerlendirme ve model_decision
```

Drift ağırlık üretmez, etiket oluşturmaz ve otomatik champion kararı vermez.
Yeni model mevcut aday/değerlendirme akışına girer. Competition seçimi ve
model-set terfi kuralları aynı kalır; birden fazla branch drift gösterirse
tek set eğitimi için aynı parent job tekrar tekrar tetiklenmez.

Örnek: eski etiketli kayıtlarda `training_weight=1`, yeni etiketli kayıtlarda
`training_weight=3` olsun. Eğitim verisini hazırlayan süreç bu kolonu doldurur.
Random split seçildiyse yeni kayıtların yalnız training tarafına düşenleri fit'e
girer. Temporal seçildiyse yalnız training penceresinde kalanlar fit'e girer.
Testte kalan kayda yüksek ağırlık vermek onu training tarafına taşımaz.

`event_time` tablodan seçilen tarih kolonudur; ağırlık kolonu ayrı bir roldür.
V1 otomatik half-life/recency formülü üretmez. Böyle bir kolon dış veri hazırlama
sürecinde üretilebilir; ileride açık bir recency politikası eklenebilir.

İki ayrı kimlik korunacak:

1. **Yeni eğitim verisi kanıtı:** mevcut feature/target ve uygunluk kurallarıyla
   yeni etiketli kayıt bulunması. Random için eski train+holdout popülasyonu,
   temporal için mevcut pencere kuralları korunur.
2. **Gerçek fit girdisi kimliği:** seçilen veri sürümü, satırlarla eşlenmiş
   ağırlıklar ve çözülmüş ağırlık politikası. Artifact ve istek kimliğine yazılır.

Yalnız ağırlık değiştirmek ilk kontrolü geçirmez. Böyle bir ayarın denenmesi
manuel train ile mümkündür; otomatik retraining için yeni veri şartı korunur.
İlk kez ağırlık eklenen eski modele, gerçekten yeni veri varsa geçiş yapılabilir;
yeni weight metadata alanı tek başına uyumsuz feature şeması sayılmaz.

Drift referansı gerçek eğitim popülasyonudur: SMOTE sentetik örnekleriyle veya
ağırlık kadar çoğaltılmış örneklerle değiştirilmez. Ağırlık kolonu drift feature
sayısına girmez. Development modunda mevcut monitoring/retraining kapatma
davranışı korunur.

## 6. Uygulama sırası

Aşağıdaki yollar repo köküne göredir. `Yeni` belirtilen dosyalar henüz yoktur.
Her görevde önce belirtilen davranış testi yazılır ve eksik davranış nedeniyle
başarısız olduğu görülür; ardından minimal uygulama ve aynı testin başarılı
sonucu kaydedilir. Sonraki göreve geçmeden ilgili mevcut testler de çalıştırılır.

### P01 — Ağırlık doğrulaması ve class_weight birleşimi

**Dosyalar:** Yeni `skyulf-core/skyulf/modeling/_sample_weights.py`;
mevcut `skyulf-core/skyulf/modeling/_class_weights.py`.
**Yeni test:** `skyulf-core/tests/unit/test_sample_weights.py`.

**Sözleşme:** `validate_sample_weight(values, expected_rows)` doğrulanmış
tek boyutlu float vektör döndürür; `None` ağırlıksız anlamına gelir.
Mevcut class-weight helper'ına opsiyonel kullanıcı vektörü eklenir; native
class-weight kullanan modele yalnız kullanıcı vektörü gönderilir.

- [ ] Null/bool/NaN/inf/negatif, yanlış boyut ve sıfır toplam testlerini yaz.
- [ ] Native/non-native sınıf ağırlığı birleşimini gerçek küçük estimator ile test et.
- [ ] Doğrulayıcıyı ve tek kez çarpma davranışını uygula; modeli ve hatalı
  weight kolonunu hata mesajında belirt.
- [ ] Kullanıcı vektörünün değişmediğini, normalize edilmediğini ve `None`
  yolunun eski davranışı koruduğunu doğrula.

### P02 — Satır metadata'sı ve preprocessing eşlemesi

**Dosyalar:** Yeni `skyulf-core/skyulf/preprocessing/_row_metadata.py`;
mevcut `skyulf-core/skyulf/data/dataset.py`,
`skyulf-core/skyulf/preprocessing/pipeline.py`,
`skyulf-core/skyulf/preprocessing/function_steps.py`,
`skyulf-core/skyulf/preprocessing/fold_adapter.py`.
**Yeni test:** `skyulf-core/tests/unit/test_weighted_row_metadata.py`.

**Sözleşme:** `RowLineage` girişe göre tamsayı indeksler ve sonlu [0,1]
fraction taşır; `apply_row_lineage(weights, lineage)` yeni vektör döndürür.
Mevcut `(X, y)` kullanıcı sözleşmesini üç elemanlı tuple'a çevirmeden,
SplitDataset'in train/test/validation bölümlerine ayrı metadata ekle.

- [ ] Aynı feature değerlerine ve tekrarlı index'e sahip satırlarda filtre,
  reorder ve kopyalama testlerini yaz; uzunluk eşitliğiyle geçen yanlış hizayı yakala.
- [ ] SplitDataset kopyalama/bölme boyunca metadata'yı koru; ağırlığı X'ten ayır.
- [ ] Yerleşik filtre/sıralama düğümlerinin gerçek seçim maskelerini taşı.
  Outlier clip/drop ve lag drop/sort modlarını ayrı doğrula.
- [ ] Bilinmeyen row-changing custom node ve güvensiz merged branch için
  açık hata ver; desteklenen column function yolunun bozulmadığını doğrula.

### P03 — Weighted resampling ve standart SMOTE

**Dosyalar:** Yeni `skyulf-core/skyulf/preprocessing/_weighted_resampling.py`;
mevcut `skyulf-core/skyulf/preprocessing/resampling.py`.
**Yeni test:** `skyulf-core/tests/unit/test_weighted_resampling.py`.

**Sözleşme:** Weighted sampler sonucu X/y yanında P02 `RowLineage` taşır.
Random sampler'larda `sample_indices_`; standart SMOTE'da gerçek üretim
callback'inden alınan ebeveyn/komşu/lambda kullanılır.

- [ ] Random over/under için seçilen gerçek indekslerden beklenen w'yu doğrula.
- [ ] Fixed seed ile adapter X/y çıktısının stock SMOTE ile aynı olduğunu test et.
- [ ] Çok sınıflı ve tekrarlı feature örneklerinde sentetik w formülünü,
  özgün satırların korunmasını ve arka arkaya iki işlemde eşlemeyi doğrula.
- [ ] Desteklenmeyen weighted sampler ve uyumsuz callback/sürüm açık hata versin.
- [ ] Desteklenen sürüm sözleşmesini belgeleyip test et; gerekiyorsa dependency
  aralığını manifest/CI/lock birlikte değiştirerek sınırla.
- [ ] Mevcut resampling testlerini çalıştır; validation ve inference'a
  sampler uygulanmadığını doğrula.

### P04 — Core fit, CV, tuning ve refit

**Dosyalar:** `skyulf-core/skyulf/pipeline/_pipeline.py`,
`skyulf-core/skyulf/modeling/base.py`, `modeling/sklearn_wrapper.py`,
`modeling/classification.py`, `modeling/ensemble.py`, `modeling/cross_validation.py`
ve aynı `skyulf-core/skyulf/` kökü altında:
`modeling/_tuning/engine.py`, `modeling/_tuning/grid_random.py`,
`modeling/_tuning/fold_pipeline.py`, `modeling/_tuning/refit.py`,
`modeling/_tuning/nested_threshold.py`, `modeling/_tuning/strategies/runner.py`,
`modeling/_tuning/strategies/optuna_folds.py`, `modeling/_tuning/strategies/halving.py`.
**Yeni test:** `skyulf-core/tests/integration/test_weighted_training_paths.py`.

**Sözleşme:** Public fit'e geriye uyumlu opsiyonel `sample_weight` girdisi ekle.
Ham tablo girdisinde bu vektör aynı tablonun satırlarına aittir. SplitDataset
girdisinde bölüm başına P02 metadata'sı kullanılır; bu durumda ayrıca tek bir
`sample_weight` vektörü vermek belirsiz girdi hatasıdır. Bütün veri vektörünü
bir training fold'una doğrudan geçirmek reddedilir. P02 metadata'sı
split, sort, fold preprocessing ve P03 sampler boyunca taşınır.

- [ ] Direct estimator ve pipeline tahminlerini aynı ağırlıklı sklearn fit ile karşılaştır.
- [ ] Her fold'a ulaşan satır kimliği/w çiftini kaydeden test estimator'ı ile
  random ve temporal CV'yi doğrula; sıralama sonrası alignment'ı özellikle test et.
- [ ] Grid/random/halving/Optuna, final refit ve nested threshold fit yollarında
  ağırlığın gerçekten estimator'a ulaştığını test et; halving'in kaynak/satır
  altkümesine aynı ağırlık altkümesinin gittiğini ayrıca doğrula.
- [ ] Sadece bir fold'da toplam ağırlık sıfırsa açık hata ver; global pozitif
  toplamın bu hatayı gizlemediğini doğrula.
- [ ] Ensemble alt/final estimator'ları için routing'i doğrula; desteklenmeyen
  kombinasyonu fit başlamadan reddet. `**kwargs` bulunmasını tek başına destek sayma.
- [ ] CV seçim skoru, holdout, threshold ve early-stop validation hesabının
  ağırlıksız kaldığını bağımsız beklenen değerlerle doğrula.

### P05 — Proje ayar dosyası ve kolon rolleri

**Dosyalar:** Yeni
`skyulf-core/skyulf/integrations/databricks/weight_config.py`;
mevcut aynı dizindeki `project.py`, `_project_files.py`, `workflow_config.py`,
`local_workflow.py`, `local_retraining.py`, `competition_project.py`,
`model_set_project.py`, `local_branches.py`.
**Yeni test:** `skyulf-core/tests/integrations/test_databricks_weight_config.py`.

**Sözleşme:** `resolve_weight_config(defaults, overrides, branch_name)`
çözülmüş JSON uyumlu kolon/politika verir. Mevcut trusted proje kaynak
yükleme, boyut ve path containment kuralları kullanılır.

- [ ] Dosyasız eski proje, varsayılan, branch override ve açık None testlerini yaz.
- [ ] Bilinmeyen anahtar/branch/politika ile target/key/event/group/feature
  kolon çakışmalarını reddet.
- [ ] Training spec'e optional weight rolünü ekle; source projection'a dahil et,
  inference input_columns'dan ve bütün branch feature listelerinden çıkar.
- [ ] Çözülmüş ayarı eğitim planına ve config digest'ine dahil et; dosya
  değişikliğinin eski artifact replay'ini değiştirmediğini doğrula.

### P06 — Single, competition ve model-set veri akışı

**Dosyalar:** `skyulf-core/skyulf/integrations/databricks/` altındaki
`local_batch.py`, `local_retraining.py`, `local_cv.py`, `local_search.py`,
`local_competition.py`, `competition_training.py`, `local_branches.py`.
**Yeni test:** `skyulf-core/tests/integrations/test_databricks_weighted_training.py`.

- [ ] Single için source projection -> pre-split filtre -> random/temporal
  split -> fold preprocessing -> final fit zincirini gerçek küçük modelle test et.
- [ ] Competition adaylarına aynı eğitim popülasyonu ve doğru w ulaştığını,
  kazananın mevcut ağırlıksız seçim metriğiyle seçildiğini doğrula.
- [ ] Model-set branch'lerinde ayrı target eksiklik maskesi, farklı ağırlık
  kolonu ve ağırlıksız branch kombinasyonunu test et.
- [ ] Ana veri sürümü/snapshot pin'ini koru; weight kolonunu ayrıca başka
  source sürümünden okuyarak satır eşleşmesini bozma.
- [ ] Desteklenen yerel veri yollarının eşdeğer sonucunu doğrula; yeni bir
  Spark-native weighted eğitim desteği varmış gibi sunma.

### P07 — Artifact, MLflow ve scoring sözleşmesi

**Dosyalar:** `skyulf-core/skyulf/integrations/databricks/` altındaki
`local_batch.py`, `local_retraining.py`, `local_training_evidence.py`,
`training_parameters.py`, `training_nodes.py`, `project.py`, `model_set_project.py`;
`skyulf-core/skyulf/inference/local_pipeline.py`.
**Yeni test:** `skyulf-core/tests/integrations/test_weighted_training_artifacts.py`.

- [ ] Eğitim artifact'ine `weighting.json` ekle: çözülmüş politika, kaynak
  digest'i, kaynak sürümü, fit girdisi digest'i ve ağırlık özeti.
- [ ] Kaynak kullanıcı ağırlığı ve sampler sonrası fit ağırlığı için count,
  min/max/mean/sum/zero_count özetlerini ayrı kaydet; ham satırları loglama.
- [ ] MLflow'a weight_column/policy ve özetleri mevcut logging katmanından yaz.
  Charts alt run'ını parametrelerle doldurma.
- [ ] Save/load sonrası ağırlık kolonu bulunmayan scoring tablosuyla tahmin
  eşitliğini ve model imzasından weight kolonunun dışlandığını test et.
- [ ] Weight artifact'i olmayan eski modellerin yüklenmesini ve açıklama
  çıktılarında ağırlığın feature olarak görünmemesini doğrula.

### P08 — Drift referansı ve retraining kanıtı

**Dosyalar:** `skyulf-core/skyulf/integrations/databricks/` altındaki
`retraining_data.py`, `retraining_task.py`, `monitoring_reference.py`.
**Testler:** aynı repo test kökünde
`skyulf-core/tests/integrations/test_retraining_data.py`,
`test_retraining_task.py`, `test_monitoring_model_set.py`;
**yeni** `skyulf-core/tests/integrations/test_weighted_retraining.py`.

- [ ] Aynı X/y, yalnız değişmiş w: yeni etiketli veri sayısı artmasın ve otomatik
  retraining tetiklenmesin; manuel eğitim yeni w'yu kullansın.
- [ ] Yeni etiketli veri + weights ayarı: guard geçsin; seçilen satırlarla
  ağırlıkların birlikte digest'i kaydedilsin. Ağırlıksız eski baseline geçişini test et.
- [ ] İstek oluşturma ve gerçek fit arasında source/config değişirse mevcut
  doğrulama yoluyla durdur; farklı girdiye ait kanıtı yeniden kullanma.
- [ ] Aynı observation/data/policy tekrarında tek istek; cooldown ve çalışan
  job kontrolleri; birden fazla drift branch'inde tek parent model-set train doğrula.
- [ ] Model fit ağırlığı değişse de aynı gözlem/reference için drift metriğinin
  değişmediğini ve reference'ın SMOTE sentetik satırları içermediğini test et.
- [ ] Mevcut champion/approval kuralları ve development modu guard'larını koru.

### P09 — Template ve kullanım örneği

**Yeni dosya:**
`skyulf-core/templates/databricks/template/{{.project_name}}/src/modeling/weights.py`.
**Testler:** `skyulf-core/tests/integrations/test_databricks_bundle_template.py`,
`test_databricks_bundle_generation.py`, `test_competition_template.py`,
`test_databricks_branch_template.py`.

- [ ] Üç layout'un ürettiği projede aynı düzenlenebilir varsayılan dosyayı doğrula.
- [ ] Yeni wizard sorusu/required Bundle değişkeni oluşmadığını test et.
- [ ] Dosyadaki kısa yorumlarda kolon seçimi, None varsayılanı ve branch override
  örneğini açıkla; gerçek ağırlık değerlerinin tablodan geldiğini belirt.
- [ ] Ağırlıksız varsayılan proje ve ağırlıklı örneği üretip load/preview yollarını doğrula.
- [ ] Bu planın ilerleme tablosuna desteklenen model/sampler kombinasyonlarını
  ve doğrulama sonuçlarını yaz; desteklenmeyenleri tamamlandı sayma.

### P10 — Yerel ve Databricks kabul kanıtı

Yerel kabul matrisi `single / competition / model-set` x `random / temporal`
altı senaryoyu kapsar. Her senaryoda başlangıç eğitimi, driftli scoring,
yeni etiketli eğitim verisi, retrain koşulu ve son model kararı izlenir.
SMOTE sınıflandırma branch'inde; regresyonda doğrudan ağırlıklı fit sınanır.

- [ ] Kontrollü küçük veri üzerinde w=1 baseline ve eşit olmayan w sonucunu
  bağımsız estimator ile karşılaştır; yalnız metriğin iyileşmesini kanıt sayma.
- [ ] Desteklenen sampler zincirini her fold içinde çalıştır; holdout'a
  sentetik satır geçmediğini kayıt kimlikleriyle kanıtla.
- [ ] Databricks'te ayrılmış test şemalarında altı senaryoyu çalıştır; mevcut
  kullanıcı modellerini/schemalarını silme veya değiştirme.
- [ ] Drift metric adı, değeri, eşiği ve true/false task sonucu kaydedilsin.
  Sadece RMSE değişimine bakarak feature drift oluştu denmesin.
- [ ] Drift observation zamanı -> yeni etiketli kayıtların source Delta commit'i
  -> train snapshot sürümü -> fit satır kimlikleri ve ağırlık digest'i sırasını
  kaydet. Sadece job'un SUCCESS olması yeni verinin kullanıldığını kanıtlamaz.
- [ ] Temporal testte yeni kayıtların gerçekten training penceresine girdiğini,
  random testte train'e düşen yeni kayıtların w değerlerini doğrula.
- [ ] Aynı girdili ikinci scoring kontrolü yeni retrain başlatmasın; weights-only
  değişiklik otomatik fresh-data guard'ını geçmesin.
- [ ] Train run ID, branch/candidate model sürümleri, mevcut ve yeni metrikler,
  terfi kararı ve champion alias'ı kaydedilsin. İyileşme zorla oluşturulmasın.
- [ ] Development testinde monitoring/retraining atlanmasını; normal modda
  ilgili görevlerin çalışmasını ve weights olmadan scoring'i doğrula.

## 7. Kontrol komutları ve kapanış ölçütü

Komutlar repo kökünde mevcut `.venv` ile çalıştırılır. Yeni test dosyası
oluşturulmadan onun komutunun geçmesi beklenmez. P01 örneği:

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests/unit/test_sample_weights.py -q -o addopts=
```

Her görev kendi yeni/değişen test dosyalarıyla aynı şekilde doğrulanır.
Entegrasyon tamamlandığında CI kapsamı korunur:

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests --collect-only -q -o addopts=
.venv/Scripts/python.exe -m pytest skyulf-core/tests -q --disable-warnings --cov=skyulf-core/skyulf --cov-branch --cov-report=term-missing --cov-report=xml:coverage-core.xml --cov-fail-under=90
.venv/Scripts/ruff.exe check .
.venv/Scripts/ty.exe check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py
.venv/Scripts/lizard.exe skyulf-core/skyulf --CCN 10 -w
git diff --check
```

CI'nin güncel test/environment seçenekleri `.github/workflows/skyulf-core-tests.yml`
ve `pr_check.yml` ile karşılaştırılır. Opsiyonel bağımlılık eksikliği yüzünden
atlanan bir weighted test başarılı kabul edilmez. Backend sözleşmesi etkilenirse
ilgili backend testleri de çalıştırılır. Frontend değişikliği bu planın kapsamında
değildir. Commit istendiğinde staged diff incelenir ve repo pre-commit hook'ları
çalıştırılır; çalışmayan/geçmeyen kontroller açıkça raporlanır.

## 8. İlerleme ve sonraki adım

- [x] Mevcut weight planı, fit/fold/resampling ve drift-retraining yolları incelendi.
- [x] Bundle sorusu olmadan ayar dosyası ve standart SMOTE kapsamı plana yazıldı.
- [ ] P01–P03: doğrulama, satır eşlemesi, sampler kanıtları.
- [ ] P04–P06: bütün eğitim yolları ve üç layout.
- [ ] P07–P09: artifact, retraining ve üretilen proje.
- [ ] P10: altı senaryo için gerçek Databricks kanıtı.

**İlk uygulama adımı P01'dir.** Bir adımın kanıtı olmadan sonraki adımın
tamamlandığı işaretlenmez. Template üzerinden özellik kullanılabilir hale
getirilmeden önce bütün zorunlu eğitim yolları ağırlığı doğru taşımalıdır.
