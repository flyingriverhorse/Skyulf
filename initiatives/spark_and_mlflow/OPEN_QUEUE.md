# Spark ve MLflow — Open Queue

Updated: 2026-09-24. **SM-00 through SM-16, SM-15L/15I/24a/25/26 and SM-22a/b/c/28a complete. SM-20a next. Target: 0.9.0.**
Next deliverable: the first working local-engine Bundle. Spark
expansion follows that Bundle. SM-15L
provides explicit-period UC output. SM-15I adds automatic new-row scoring
for the first Bundle; SM-18 and streaming remain parked.

Read [HANDOFF.md](HANDOFF.md), [the integration plan](04-databricks-integration-plan.md)
and [the pre-Bundle lifecycle plan](06-prebundle-model-lifecycle-plan.md) before SM-20a.
The [SM-20 Bundle plan](05-sm20-bundle-plan.md) remains the packaging gate.
Historical completion evidence is preserved in
[OPEN_QUEUE_tamamlanmakaydi.md](OPEN_QUEUE_tamamlanmakaydi.md).
SM-26 added a local MLflow package; SM-15L and SM-15I validated UC writers.
No Bundle has been generated.

Durumlar: READY = başlanabilir; WAIT = önceki görev bekleniyor;
LATER = son aşama; ACTIVE = yürütülüyor; BLOCKED = somut dış engel;
DONE = kanıtla tamamlandı. SM-00 commit: `105a6fe4`.
DEFERRED = user postponed this work; PARKED = do not implement until resumed.

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
| SM-09 | Native FE + worker model | SM-08 | DONE | Spark tahmini local ile key bazında aynı; aşağıda kanıt |
| SM-10 | Worker Python FE + model | SM-09 | DONE | Batch-safe portable FE; window gibi yollar açık ret |
| SM-11 | Classification ve ölçek kapısı | SM-10 | DONE | Class/proba/threshold parity; transport, packaging and scale evidence |
| SM-12 | Opsiyonel MLflow tracking | SM-11 | DONE | Off bağımsız; gerçek run lifecycle ve izolasyon |
| SM-13 | MLflow model packaging | SM-12 | DONE | Temiz ortamda pyfunc yükleme ve parity |
| SM-14 | Registry/Unity Catalog adapter | SM-13 | DONE | Explicit publish; alias/version pinning; local evidence |
| SM-15 | Monthly batch + Delta sink | SM-14 | DONE | 43 local Delta/contract/admission tests; evidence below |
| SM-16 | Gerçek Databricks kapısı | SM-15 | DONE | Selected serverless workflow passed; live evidence below |
| SM-15L | Explicit-period local predictions -> UC Delta | SM-24a | DONE | Score requested period; preserve rows outside it; Spark only for bounded I/O |
| SM-15I | Automatic incremental local scoring | SM-15L | DONE | No date/source-version input; 80+80 synthetic and 200+100 real taxi rows; no-op replay; [core evidence](13-sm15i-live-validation-report.md), [real-data proof](15-sm15i-real-nyctaxi-live-report.md) |
| SM-17 | Complete existing Spark node/model coverage | SM-20a | LATER | Resume only after the first local Bundle; preserve all inventoried gaps |
| SM-18 | Backend/Canvas and legacy artifact bridge | SM-17 | PARKED | Capability UI/API, DAG contracts and existing deployment compatibility |
| SM-26 | Local pandas/Polars pipeline MLflow packaging | SM-16, existing local persistence | DONE | Fitted FE/model/engine, schema and thresholds preserved; MLflow 3.16.1 isolated-process parity; Spark/HTTP scopes rejected |
| SM-25 | Local-first config / SDK preflight | SM-26 | DONE | Immutable local config; pinned artifact load, bounded frame and actionable preflight; no node/model allowlist |
| SM-24a | Local training and bounded batch prediction | SM-25 | DONE | Two UC source tables, five live cross-job models, 62-ID FE audit and bounded monthly reads; [evidence](08-sm24a-live-validation-report.md) |
| SM-22a | Pinned candidate/champion validation | Current evaluation/registry | DONE | Local comparison/report and isolated UC metrics gate passed; [local evidence](16-sm22a-local-validation-report.md), [live evidence](17-sm22a-live-metrics-report.md) |
| SM-22b | Explicit promotion and rollback | SM-22a | DONE | Version-checked alias changes, prior/new-version receipt, conflicts and permission tests; [live evidence](18-sm22b-live-validation-report.md) |
| SM-22c | Challenger and previous-champion aliases | SM-22b | DONE | Validated challenger staging; guarded three-alias promotion/rollback; legacy receipts and partial-write tests; [local and UC evidence](19-sm22c-lifecycle-alias-validation-report.md) |
| SM-28a | Label-aware retraining service | SM-22a, SM-22b, SM-22c, SM-24a | DONE | Pinned training snapshot, temporal holdout, candidate registration/comparison; no automatic promotion; [evidence](20-sm28a-label-aware-retraining-report.md) |
| SM-20a | First local-engine Bundle/template | SM-24a, SM-15I, SM-22b, SM-22c, SM-28a | READY | Package tested train/compare/promote/incremental-score services; validate, deploy and run |
| SM-28b | Optional monthly retraining schedule | SM-20a, SM-28a | WAIT | Wire separate train/compare/explicit-promote/score jobs with label cutoff and pinned versions |
| SM-24b | Optional Databricks Jobs API operations | SM-20a | LATER | Add dynamic submit/status/cancel only if Bundle jobs are insufficient |
| SM-24d | Hard transport budget for wide UC rows | SM-24a | LATER | Add a proven paged/size-limited source adapter when exact transfer-byte enforcement is required; current Spark iterator bounds accepted decoded rows and frame memory only |
| SM-27 | Full-history local rescore and selectable Bundle mode | SM-20a, SM-15L | LATER | Score a pinned, bounded source snapshot into a new prediction generation; validate and activate explicitly; preserve prior generation |
| SM-19 | Optional live HTTP / SQL ai_query / endpoint operations | SM-20a, compatible pyfunc package | LATER | Add only after serving parity; streaming remains parked |
| SM-21 | Optional Databricks feature tables / online lookup | SM-20a; SM-19a for online serving | LATER | Point-in-time lookups and optional online freshness; declare any Spark dependency |
| SM-23 | Optional monitoring and inference observability | SM-20a, relevant batch/serving adapter | LATER | Existing Skyulf metrics + optional Databricks monitoring/inference tables |
| SM-24c | Spark batch workflow adapter | SM-20a, existing Spark sink | LATER | Expose tested Spark runner after first local Bundle |
| SM-20b | Spark Bundle enhancement | SM-24c, selected SM-17 slices | LATER | Add tested Spark engine choice while preserving local variant |

SM-22a/b/c and SM-28a are now prerequisites for the first Bundle. SM-28b only
wires their optional monthly schedule after the Bundle exists. SM-27 remains
post-Bundle and independent. Endpoint, feature lookup and monitoring remain
optional; broad Spark expansion follows the local Bundle.

## Local-first path to the first Bundle

These tasks were identified in the reference review; the latest user direction
moves the model lifecycle services before the first local Bundle. Spark and
other optional integrations follow the Bundle.
Full task scope, proposed code areas, reference sources and acceptance checks are
in [04-databricks-integration-plan.md](04-databricks-integration-plan.md).

1. **SM-26:** fitted pandas/Polars pipeline -> MLflow package -> same local predictions.
2. **SM-25:** config, pinned local artifact loading, limits and preflight.
3. **SM-24a:** local training and bounded local batch prediction in Databricks.
   Its [live validation report](08-sm24a-live-validation-report.md) records two
   source tables, five cross-job models and every preprocessing registration ID.
4. **SM-15L:** explicit-period local predictions reach guarded Delta. The live
   test used scripted period and source-version values; it is not automatic.
5. **SM-15I:** add automatic first-snapshot and later new-row scoring with a
   committed source-version receipt and append-safe output. See the
   [incremental plan](12-sm15i-incremental-scoring-plan.md).
6. **SM-22a/b/c:** compare concrete candidate/champion versions on one labeled
   holdout, stage a validated `@challenger`, then explicitly promote with
   `@previous_champion` and guarded rollback.
7. **SM-28a:** train a candidate from a pinned, label-aware snapshot and
   temporal holdout; compare it without automatic promotion.
8. **SM-20a:** generate, validate, deploy and run the first local-engine Bundle
   using the [two-run rehearsal](05-sm20-bundle-plan.md) without manual dates
   or source versions. Package the already-tested lifecycle services.
9. **After SM-20a:** SM-28b wires optional monthly retraining; SM-27 adds
   explicit `full_rebuild` alongside default `incremental_append`. SM-15L's
   `period_update` remains an explicit backfill.
10. **Later:** optional SM-24b/24d/19/21/23 adapters; Spark SM-24c/SM-17, then SM-20b Bundle enhancement.

SM-26 is the explicit packaging task added after the local-first MLflow discussion.
It reuses the fitted local pipeline rather than requiring every FE node to gain a
portable Spark codec first. Each advertised local pipeline/model variant needs
save/load/prediction evidence. This does not make its artifact Spark-compatible.
Backend joblib dictionary bridging stays in parked SM-18. Local prediction
and UC Delta publication remain separate tasks; SM-15I is now on the first
Bundle's critical path, with an explicit small-data memory limit. Spark can
perform UC table I/O here without becoming the FE or model execution engine.

Pandas/Polars training -> Spark inference is available only for the currently
compatible FE/model bundle contract. Both Spark modes remain capability-gated.
Delaying SM-17 does not enable arbitrary local pipelines or all registered models.
SM-25 extracts the supported-workflow usability subset from later SM-17j; wider
native model/engine coverage waits until the first local Bundle passes.

## Later SM-17 scope and completion contract

The user requested coverage of the existing nodes and models, not closure by
listing the rest as unsupported. Missing combinations stay open unless the user
explicitly defers them. This expansion starts after SM-20a; none of its missing
features is marked DONE. No implicit collection or algorithm swap.

| Task | Status | Deliverable |
| --- | --- | --- |
| SM-17-00 | LATER | Expand the [100-ID inventory](NODE_SUPPORT.md) into per-option fit/apply/state/model/runtime coverage and a registry coverage guard |
| SM-17a | LATER | Column/cast/cleaning/date/math/interaction/polynomial and basic scaler native paths |
| SM-17b | LATER | Remaining imputation, categorical encoders, binning/ranges and robust/quantile state |
| SM-17c | LATER | Feature selection, transforms and outliers; fit/apply distinction and row alignment |
| SM-17d | LATER | Group/window/history and target/WOE OOF semantics; deterministic splits and leakage gates |
| SM-17e | LATER | Text/vectorization/embeddings and geo; sparse/vector contracts and worker dependencies |
| SM-17f | LATER | Split/resampling/inspection; training-only row changes, stable membership and bounded previews |
| SM-17g | LATER | Existing model-family bundle/inference coverage: sklearn variants, XGBoost/LightGBM, ensembles/calibration and clustering |
| SM-17h | LATER | Explicit native Spark model training/transform, artifact kind and MLflow persistence; all existing model IDs retain decisions/tasks |
| SM-17i | LATER | Evaluation/CV/tuning/thresholds and explainability/SHAP contracts; parallel trials versus distributed fit; bounded execution |
| SM-17j | LATER | Easier config/SDK entry point, preflight and independent platform/FE/model/sink/tracking choices; no Canvas or DAB work |
| SM-17k | LATER | Family regression gates and selected Databricks runtime validation; no missing requested coverage silently marked DONE |

Family tasks depend on SM-17-00 and their relevant preceding contracts. Model
work need not wait for every independent FE family, but no task should claim
support before its artifact, inference and runtime requirements pass.

Details, reference-repo comparison and parked endpoint tasks:
[Spark/Databricks gap review](reports/2026-09-22-spark-databricks-gap-review.md).

## Detay planlar

- Completed SM-16 evidence and scope: [PLATFORM_VALIDATION.md](PLATFORM_VALIDATION.md).
- SM-15 and SM-15L provide Spark and explicit-period local UC writers.
  SM-15I is required for automatic new-row output in SM-20a. The first
  template precedes Spark expansion.
- Latest user direction supersedes the prior Spark-first and template-last order:
  deliver a local pandas/Polars Bundle, then expand Spark and add a tested Spark
  option. SM-18 and continuous streaming remain parked.

- SM-00–SM-07: [Core/Spark](01-core-spark-plan.md)
- SM-08–SM-11: [Inference](02-inference-plan.md)
- SM-12–SM-20: [MLflow/batch/son teslimatlar](03-mlflow-batch-delivery-plan.md)
- Current integration lane: [Local-first integration](04-databricks-integration-plan.md)
  and [SM-20 Bundle rehearsal](05-sm20-bundle-plan.md).
- Ortak kurallar: [Mimari](ARCHITECTURE.md), [Doğrulama](VALIDATION.md)

SM-17 requires the full tracked scope above. Some algorithms may need an explicit
alternative backend rather than an equivalent native Spark implementation; that
decision must not hide missing support. SM-19 streaming remains optional and parked.

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
test döngüsüyle tamamla ve kanıtı yaz. Sonraki bağımlılığı aç. İlk local Bundle
kapısından önce Spark veya endpoint işine başlama. SM-00 yalnız test/runtime
hazırlığıydı; güncel destek ve sınırlar tamamlanan görevlerin kayıtlarında belirtilir.
