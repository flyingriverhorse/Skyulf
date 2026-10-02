# `weight_column` planı: satır ağırlığıyla eğitim

> Durum: **ertelendi**. Drift'e bağlı retraining işi bu repoya geldiğinde önce
> **Güvenli v1** (sadece Bundle/core) yapılacak. **Tam versiyon** yalnızca
> gerçek bir canvas ihtiyacı çıkarsa ele alınacak.
> Satır numaraları 2026-10-02, branch `091` (commit `e6bb3de2`) içindir.

## 1. Ne yapıyor, neden lazım?

Model eğitilirken her satırın hataya katkısı normalde eşittir. `weight_column`
ile her satır kendi ağırlığını taşır: ağırlığı 3 olan satırın hatası 3 kat
sayılır (`model.fit(X, y, sample_weight=w)`).

| | `class_weight` (bugün var) | `weight_column` (bu plan) |
|---|---|---|
| Ağırlığı kim belirler | Sınıf frekansı (sadece etiket) | Kullanıcı, satır satır |
| Tipik kullanım | Dengesiz sınıflar | İş değeri, güncellik, sayım |
| Regresyonda | Yok | Var |

Gerçek örnekler:

- **dyac** (`docs/artifacts/reconstructed_training.py:104-108`): her hedefin
  ayrı bant ağırlığı var (`{mean_age, mean_salary, mean_AuM, hazard_A_pct}_band_weight`
  ∈ {1, 2, 4}). Ortadaki ~%60 → 1, sonraki ~%20 → 2, uçlardaki ~%20 → 4. Amaç
  regresyonun ortalamaya çekme eğilimini kırmak. Not: değerlendirme
  ağırlıksız yapılmış ve fayda sınırlı (yaş modelinde R² ≈ 0,13).
- **Drift sonrası retraining**: eski ve yeni veri birlikte eğitilirken yeni
  satırlara daha çok ağırlık vermek. Bu planın asıl gerekçesi budur.
- **Sayım**: bir satır N aynı kaydı temsil ediyorsa ağırlık N olur.

## 2. Değişmez kurallar (her iki versiyon)

Testler bu kuralları doğrudan sınamalı:

1. **Ağırlık hiçbir zaman feature değildir.** Model girdisine, `feature_names_in_`'e
   ve preprocessing node'larına girmez.
2. **Scoring'de gerekmez.** Kayıtlı manifestteki `input_columns`'a
   (`inference/local_pipeline.py:39`, `:95`) yazılmaz. Scoring tablosunda
   ağırlık kolonu olmaması hata değildir.
3. **Satır hizası korunur.** Her fit'te `len(w) == len(y)` olur ve `w[i]`,
   `y[i]`'nin satırına aittir. Fold, filtre ve refit sonrasında da bu geçerli.
4. **Geçerli değerler**: sonlu, `>= 0`, null yok, toplam > 0. Aksi halde
   açık bir hata mesajı verilir. `bool` kabul edilmez.
5. **`class_weight` ile birleşir**: ikisi birden varsa `w_final = w_user * w_class`.
   `w_class` her fit'in kendi etiketlerinden hesaplanır (bugünkü
   `sample_weight_for_fit` mantığı).
6. **`sample_weight` almayan model** açık hatayla reddedilir. Bugün
   `class_weight` için kullanılan mesaj kalıbı (`_class_weights.py:32-42`)
   aynen kullanılır.
7. **Metrikler v1'de ağırlıksızdır.** MLflow'a `weight_column` adı ve ağırlık
   özeti (min/max/ortalama/sıfır sayısı) yazılır. Ağırlıklı metrikler tam
   versiyonda bir seçenek olarak gelir.
8. **Branch başına** ayarlanabilir; dyac'ta her hedefin ayrı ağırlığı var.

## 3. Bugünkü kod haritası (dokunulacak yerler)

### 3.1 Config ve veri hazırlığı (Bundle)

| Dosya:satır | Ne var | Ne değişecek |
|---|---|---|
| `integrations/databricks/workflow_config.py:24` `WORKFLOW_FIELDS` | İzinli config anahtarları | `weight_column` eklenir |
| `workflow_config.py:96` `_columns()` | Rol kolonlarının (target/event/group) birbirinden farklı olması | Ağırlık kolonu target, event, key, group ve `input_columns` ile çakışamaz |
| `local_workflow.py:226` `training_spec()` | Config → `LocalTrainingSpec` | `weight_column` spec'e taşınır |
| `local_retraining.py:76` `LocalTrainingSpec` | Eğitim sözleşmesi | `weight_column: str \| None = None` |
| `local_retraining.py:195` `_validate_columns()` | Kolon rolleri | Ağırlık kolonu rol çakışma kontrolü |
| `local_retraining.py:257` `source_columns` | Snapshot'a okunan kolonlar | Ağırlık kolonu okunur ama feature listesine girmez |
| `local_retraining.py` `split_labeled_snapshot()` (~665-708) | Pre-split + train/holdout ayrımı | Ağırlık kolonu satırla birlikte bölünür; pre-split adımları ağırlık kolonuna **yazamaz** |
| `local_pre_split.py` `_FIXED_RULES` / `fixed_columns()` | Pre-split'in yazdığı kolonlar | Yazılan kolonlar arasında ağırlık kolonu varsa hata |
| `local_batch.py:74` `fit_local_workflow()` | `pipeline.fit(data, target_column=...)` | Ağırlık vektörü fit'e geçirilir |
| `local_cv.py:158` `evaluate_training_cv()` | Bundle CV | Fold ağırlıkları |
| `local_search.py:154` `_prepare_selected_model()` | Bundle search | Arama ağırlıkla yapılır |
| `local_competition.py`, `competition_training.py`, `local_branches.py` | Çoklu model ve branch | Branch'in `weight_column`'ı kullanılır |
| `templates/databricks/schema/*.json` → `databricks_template_schema.json` | Guided setup | Ağırlık sorusu (opsiyonel), `build_schema.py` ile yeniden üretilir |

### 3.2 Pipeline ve modelleme (core)

| Dosya:satır | Ne var | Ne değişecek |
|---|---|---|
| `modeling/base.py:32` `extract_xy()` | X'ten sadece target'ı ayırır | Değişmez (ağırlık buraya gelmeden ayrılır) |
| `pipeline/_pipeline.py:344` `SkyulfPipeline.fit()` | `fit(data, target_column, *, on_leakage)` | `sample_weight=None` keyword'ü |
| `pipeline/_pipeline.py:440` `_fit_input_metadata()` | Input şeması = ham kolonlar − target | Ağırlık kolonu da şemadan çıkarılır (kural 2) |
| `pipeline/_pipeline.py:447` `_fit_model()` | Model fit | Ağırlık estimator'a iletilir |
| `modeling/base.py:404` `fit_predict()`, `:452` | Calculator fit | `sample_weight` parametresi |
| `modeling/sklearn_wrapper.py:84-105` | `class_weight` → `sample_weight` | `w_user * w_class` birleşimi |
| `modeling/_class_weights.py:32` `sample_weight_for_fit()` | Sınıf ağırlığı | `combine_sample_weights(model, user_w, class_weight, y)` helper'ı |
| `modeling/_tuning/grid_random.py:111-174` (fit `:162`) | Fold döngüsü | `w[train_idx]` dilimlenir |
| `modeling/_tuning/fold_pipeline.py:175-177` | Fold-aware step | `fit(X, y, sample_weight=...)` kabul eder |
| `modeling/_tuning/strategies/runner.py:91-93` | `searcher.fit(X_arr, y_arr)` (halving/optuna sklearn searcher) | `searcher.fit(X_arr, y_arr, sample_weight=w)`; sklearn fit param'larını fold'a göre dilimler |
| `modeling/_tuning/strategies/optuna_folds.py:78`, `:184-186` | Optuna fold hazırlığı | `_PreparedFold.sample_weight` = `w_user[train] * w_class` |
| `modeling/_tuning/refit.py:86-100` | Final refit | Tüm train ağırlığı |
| `modeling/_tuning/nested_threshold.py:67` | Eşik ayarı için iç fit | `step.fit(X, y, sample_weight=...)` |
| `modeling/cross_validation.py:76`, `:235` `_slice_fold_data`, `:277` `_run_cv_fold` | CV değerlendirme | Ağırlık fold ile dilimlenir |
| `modeling/ensemble.py:142`, `:215` | Voting/Stacking | sklearn `fit(sample_weight)` destekliyor; desteklemeyen alt modelde hata |
| `modeling/classification.py:245-252` | `sample_weight` geçiren fit override | Değişmez, sadece test edilir |
| Boosting `eval_set` (early stopping) | Doğrulama seti | v1: ağırlıksız (dyac da öyle); belgelenir |

### 3.3 Preprocessing (sadece tam versiyon)

| Dosya:satır | Ne var | Sorun |
|---|---|---|
| `preprocessing/pipeline.py:75` `_RESAMPLING_TYPES`, `:82` `_ROW_DROPPING_TYPES` | Satır sayısını değiştiren adımlar | Ağırlık satırlardan ayrı taşınırsa hiza bozulur |
| `preprocessing/fold_adapter.py:29` `ROW_COUNT_CHANGING_STEP_TYPES` | IQR, ZScore, Winsorize, EllipticEnvelope, ManualBounds + yukarıdakiler | v1 bunları reddetmek için bu listeyi kullanır |
| `preprocessing/_helpers.py` `resolve_columns*` (`:152`, `:179`) ve `auto_detect_*` | Kolon seçilmezse otomatik algılama | Ağırlık X'te kalırsa ölçeklenir veya encode edilir |

### 3.4 Backend ve frontend (sadece tam versiyon)

| Dosya:satır | Ne var |
|---|---|
| `backend/ml_pipeline/_execution/engine/_node_runners.py:718` `_run_training`, `:751` | Canvas eğitim node'u `target_column`'ı params'tan okur |
| `backend/ml_pipeline/_execution/schemas.py:61` | `target_column` alanı |
| `backend/ml_pipeline/_execution/_schema_validator.py` `_STRING_REF_KEYS` | Kolon referans doğrulaması (`weight_column` eklenecek) |
| `frontend/ml-canvas/src/modules/nodes/modeling/TrainingSettings.tsx`, `trainingSettings/ModelConfigSection.tsx`, `useTrainingSettings.ts` | Eğitim node'u ayarları (target seçimi burada) |
| `frontend/.../EnsembleSettings.tsx`, `ensembleSettings/*` | Ensemble eğitim ayarları |
| `frontend/.../core/utils/pipelineConversion/*` | Canvas → backend params |

## 4. Güvenli v1 (sadece Bundle/core)

### 4.1 Kapsam

- Bundle/local SDK eğitimi: tek model, branch'ler, competition, Bundle CV,
  search (grid, random, halving, optuna) ve retraining.
- Canvas ve backend **yok**. Backend'in `weight_column`'ı bilmemesi sorun
  değil, çünkü bu anahtar sadece Bundle config'inde yaşar.

### 4.2 Tasarım: ağırlık preprocessing'e hiç girmez

Ağırlık, train frame'den **pipeline'a girmeden önce** bir vektör olarak
ayrılır ve pozisyonel olarak taşınır. Preprocessing ağırlığı hiç görmez;
böylece auto-detect yapan node'ların ağırlığı ölçekleme riski de ortadan kalkar.

```text
snapshot ─ pre_split (ağırlığa yazamaz) ─ train/holdout ayrımı
   train_frame ──► pop(weight_column) ──► w (vektör)
        │                                  │
        ▼                                  ▼
 SkyulfPipeline.fit(train_frame, target, sample_weight=w)
        │  preprocessing (satır sayısı DEĞİŞMEZ → w hizalı kalır)
        ▼
   model.fit(X, y, sample_weight = w * w_class)
   tuning/CV: w[train_idx] fold ile birlikte dilimlenir
```

Bu tasarım **tek bir varsayıma** dayanır: split sonrası preprocessing satır
sayısını değiştirmez. v1 bu varsayımı doğrulamada zorunlu kılar.

### 4.3 Doğrulama (config aşamasında, eğitimden önce)

Aşağıdakiler eğitim başlamadan, açık mesajla reddedilir:

| Durum | Mesaj fikri |
|---|---|
| Split sonrası preprocessing'de satır değiştiren adım (`step_changes_row_count`, `ROW_COUNT_CHANGING_STEP_TYPES`, `RowFilterFunction`, resampling) | "weight_column with <step> is not supported yet: the step removes or adds rows. Move the filter to pre_split_steps." |
| Ağırlık kolonu target/event/key/group/`input_columns` ile aynı | rol çakışması |
| Pre-split adımı ağırlık kolonuna yazıyor | "pre_split_steps must not modify weight_column" |
| Değerler: null, negatif, sonsuz, bool, toplam 0 | geçersiz ağırlık |
| Model `sample_weight` almıyor | kural 6 |
| Clustering / unsupervised | ağırlık desteklenmiyor |

Pre-split filtreleri (`filter_step`, `DropMissingRows`, `ManualBounds`
pre-split'te) **serbest**: ağırlık o aşamada henüz normal bir kolon olduğu için
satırla birlikte gider.

### 4.4 Değişecek kod (v1)

1. `workflow_config.py`: `WORKFLOW_FIELDS` + `_columns()` rol kontrolü. Branch
   config'inde de aynı anahtar (`local_branches.py`).
2. `LocalTrainingSpec.weight_column` + `_validate_columns` + `source_columns`.
3. `local_pre_split.py`: yazılan kolon ağırlık kolonu olamaz.
4. Yeni küçük modül: `skyulf/modeling/_sample_weights.py`
   - `validate_weights(w) -> np.ndarray` (kural 4)
   - `pop_weight(frame, column) -> (frame_without, w)` (pandas ve polars)
   - `combine(model, user_w, class_weight, y) -> np.ndarray | None` (kural 5, 6)
5. `SkyulfPipeline.fit(..., sample_weight=None)`: input şemasından ağırlığı
   çıkarır; `fit_predict` ile calculator'a iletir.
6. Fit noktaları: `sklearn_wrapper.py`, `grid_random.py`, `fold_pipeline.py`,
   `runner.py`, `optuna_folds.py`, `refit.py`, `nested_threshold.py`,
   `cross_validation.py`, `ensemble.py`. Her yerde `sample_weight_for_fit`
   çağrısı `combine(...)` ile değişir; fold dilimleme `w[idx]`.
7. Bundle akışı: `local_batch.fit_local_workflow`, `local_cv`, `local_search`,
   `local_competition`/`competition_training`, `local_retraining`.
8. MLflow: parametre olarak `weight_column` ve `training_parameters.json`'a
   ağırlık özeti.
9. Template: `README.md.tmpl` + guided schema'da opsiyonel soru; recipe'lerde
   yorumlu örnek.

### 4.5 Testler (önce yazılacak)

En güçlü testler **eşdeğerlik** testleridir; hizalama hatasını sessiz
kalmadan yakalarlar:

| Test | Neyi kanıtlar |
|---|---|
| **Sıfır ağırlık ≡ satırı silmek**: belirli satırlara `w=0` verip eğit, aynı satırları silip ağırlıksız eğit; deterministik modelde (LinearRegression, LogisticRegression) katsayılar eşit | Ağırlık doğru satıra gidiyor |
| **Tamsayı ağırlık ≡ satırı çoğaltmak**: `w=2` ile eğitmek = satırı iki kez koymak (LinearRegression) | Ağırlık gerçekten uygulanıyor |
| Satırları karıştırıp aynı ağırlıklarla eğit → aynı model | Pozisyon hatası yok |
| Ağırlık kolonu feature değil: `feature_names_in_` / manifest `input_columns` içinde yok | Kural 1, 2 |
| Scoring tablosunda ağırlık kolonu yokken scoring çalışır | Kural 2 |
| `class_weight="balanced"` + `weight_column` = elle hesaplanan `w * w_class` | Kural 5 |
| grid / random / halving / optuna: her fold'da `len(w) == len(y)` ve fold'un ağırlıkları doğru satırlardan geliyor (spy estimator `fit` argümanlarını kaydeder) | Fold dilimleme |
| Refit ve nested threshold ağırlığı alır | Unutulan fit yolu yok |
| pandas ve polars aynı sonucu verir | Motor eşitliği |
| Split sonrası IQR/Oversampling + ağırlık → açık hata | v1 sınırı |
| Pre-split filtre + ağırlık → çalışır, hizalı | Pre-split serbest |
| Geçersiz değerler (null, negatif, inf, bool, hepsi 0) → hata | Kural 4 |
| `sample_weight` almayan model → hata | Kural 6 |
| Branch'lerde farklı ağırlık kolonları (dyac'taki gibi 4 hedef) | Kural 8 |
| Retraining: eklenen yeni veride ağırlık kolonu var ve doğru birleşiyor | Retraining |

### 4.6 Kabul kriterleri

- Bütün testler pandas ve polars'ta geçiyor.
- Ağırlık verilmediğinde davranış **bit bit aynı**: mevcut test suite'i ve
  kayıtlı snapshot'lar değişmiyor.
- ruff, ty, Lizard (CCN ≤ 10) temiz; `build_schema.py --check` geçiyor.
- dyac eğitim verisiyle bir hedef (`mean_salary`) Bundle'da
  `weight_column=mean_salary_band_weight` ile eğitilir ve
  `reconstructed_training.py` ile kıyaslanır (xgboost sürüm farkı payı dahil).

## 5. Tam versiyon (core + backend + frontend)

v1'in üzerine şunlar eklenir.

### 5.1 Split sonrası satır değiştiren adımlarla çalışma

Ağırlık, v1'deki gibi dışarıdan vektör olarak taşınmaz. X içinde
**korunan bir kolon** olarak taşınır ve model fit'inin hemen öncesinde ayrılır.

- `FeatureEngineer` bir `protected_columns` bilgisi alır (ağırlık kolonu).
  Bu kolon tüm preprocessing adımlarında:
  - auto-detect sonuçlarından çıkarılır (`resolve_columns*`, `auto_detect_*`);
  - açıkça seçilirse hata verir;
  - satır filtrelerinde (IQR, ZScore, ManualBounds, DropMissingRows,
    Deduplicate, EllipticEnvelope, RowFilterFunction) satırla birlikte gider.
- Kolonu silen veya yeniden adlandıran adımlar (DropMissingColumns,
  feature selection, OneHot) korunan kolona dokunamaz. Her node için tek
  tek test edilmeli; registry contract testine
  "korunan kolon değişmeden çıkar" maddesi eklenir.
- **Resampling** için ayrıca karar gerekir:
  - Oversampling satırı kopyalıyorsa ağırlık da kopyalanır.
  - SMOTE gibi sentetik satırlarda ağırlık ya 1 olur ya da komşulardan
    interpolasyonla hesaplanır. **Karar gerekiyor**; öneri: sentetik satır = 1
    ve uyarı.
  - Undersampling: ağırlık satırla gider.
- Modelleme tarafında ağırlık `extract_xy` sonrası X'ten çıkarılır
  (`modeling/base.py:32`, `_pipeline.py:447`). Tuning fold'ları X'i
  dilimlediği için ağırlık kendiliğinden hizalı kalır.

Risk: ağırlık bir node'dan sızarsa (ör. yeni eklenen bir node korunan kolonu
bilmezse) feature olur. Önlem olarak `model.fit` öncesinde X'te ağırlık kolonu
kalmadığını doğrulayan bir assert eklenir ve contract testi her yeni node'u
otomatik kapsar.

### 5.2 Backend

- Training ve ensemble node params'ına `weight_column` (`_node_runners.py:718`).
- `_schema_validator.py`: `weight_column` için `_STRING_REF_KEYS` eklenir.
  Bu sayede eksik kolon canvas'ta işaretlenir.
- Job/schema snapshot testleri (`tests/integration/test_pipeline_config_snapshots.py`)
  kasıtlı olarak güncellenir.
- Leakage kontrolü: ağırlık hedeften türetilmişse (dyac'taki bantlar `y`
  kantillerinden) bu feature sızıntısı değildir, çünkü ağırlık sadece eğitimde
  kullanılır. Yine de dokümante edilir.
- Deployment/inference: ağırlık kolonu inference şemasında **yok** (kural 2).

### 5.3 Frontend

- Training ve Ensemble ayarlarında "Sample weight column (optional)" seçici:
  sayısal kolonlar, target hariç.
- Validasyon: target ile aynı olamaz, preprocessing'de açıkça seçilmiş bir
  kolon olamaz (uyarı).
- Converter (`pipelineConversion`): `weight_column` params'a yazılır.
- Yardım metni: `class_weight` ile farkı tek cümleyle.
- Testler: vitest (seçici, validasyon, converter); `npm run lint`, `tsc`,
  `complexity:check`, `build`, `size-check`. Static asset'ler yeniden üretilir.

### 5.4 Metrikler

- Seçenek: `weighted_metrics: false` (varsayılan). `true` olursa
  holdout/CV metrikleri `sample_weight` ile hesaplanır. sklearn metriklerinin
  çoğu `sample_weight` alır; almayanlar listelenir.
- MLflow'da metrik adlarına karışıklık olmasın diye `weighted_` öneki
  kullanılır.

### 5.5 Retraining için güncellik ağırlığı

- Kullanıcının kolon üretmesine gerek kalmadan retraining ağırlığı kendisi
  üretebilir:
  `recency_weight: {half_life_days: 90}` → `w = 0.5 ** (age_days / half_life)`.
  `age_days` değeri `event_column`'dan hesaplanır.
- Kullanıcının `weight_column`'ı da varsa ikisi çarpılır.
- Test: yarı ömür kadar eski satırın ağırlığı tam 0,5 olmalı; gelecekteki
  tarih hata vermeli.

## 6. Takip listesi

**Güvenli v1**

- [ ] `_sample_weights.py` + birim testleri (validate, pop, combine)
- [ ] Config: `WORKFLOW_FIELDS`, `_columns()`, branch config, `LocalTrainingSpec`
- [ ] Pre-split'in ağırlığa yazmasını engelleme
- [ ] v1 reddetme kuralları (split sonrası satır değiştiren adımlar)
- [ ] `SkyulfPipeline.fit(sample_weight=...)` + manifest şemasından çıkarma
- [ ] Fit noktaları: wrapper, grid/random, fold_pipeline, runner (halving/optuna),
      optuna_folds, refit, nested_threshold, cross_validation, ensemble
- [ ] Bundle akışı: batch, CV, search, competition, branches, retraining
- [ ] MLflow parametreleri ve özet
- [ ] Template: guided soru, README, yorumlu örnek; schema yeniden üretimi
- [ ] Eşdeğerlik testleri (sıfır ≡ silme, tamsayı ≡ çoğaltma, karıştırma)
- [ ] dyac `mean_salary` kıyası
- [ ] Docs (`databricks_bundle.md`) + changelog

**Tam versiyon (ek)**

- [ ] `protected_columns` FeatureEngineer + tüm node'larda contract testi
- [ ] Satır filtreleri ve resampling'de ağırlık hizası (+ SMOTE kararı)
- [ ] Fit öncesi "X'te ağırlık kalmadı" assert'i
- [ ] Backend params, schema validator, snapshot güncellemesi
- [ ] Frontend seçici, validasyon, converter, testler, build
- [ ] Ağırlıklı metrik seçeneği
- [ ] Retraining `recency_weight`

## 7. Açık kararlar

1. Metrikler: v1'de ağırlıksız. Tam versiyonda varsayılan ne olsun?
2. SMOTE sentetik satır ağırlığı: 1 mi, interpolasyon mu?
3. Early stopping `eval_set` ağırlıklı mı olsun (v1: hayır)?
4. Ağırlıkların ölçeği: toplamı satır sayısına normalize edilsin mi?
   Öneri: hayır, kullanıcının verdiği aynen kullanılsın. Bazı modellerde
   regularization etkisi değişir; bu dokümante edilir.
5. `recency_weight` v1'e mi girsin? Öneri: retraining işiyle birlikte, v1'in
   hemen ardından.

## 8. Commit planı (uygulanırken)

1. `feat(core): sample weight helpers and SkyulfPipeline.fit(sample_weight)`
2. `feat(core): pass sample weights through tuning, CV and refit`
3. `feat(databricks): weight_column config, validation and training flow`
4. `feat(databricks): template guidance for weight_column`
5. `docs: weight_column`

Her commit kendi testleriyle gelir. Ağırlık verilmediğinde mevcut testlerin
hiçbiri değişmemeli.
