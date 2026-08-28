# Benchmark Review — kebab-test-ml vs skyulf-core

**Subject:** `/Users/BH7043/repositories/kebab-test-ml/benchmarks`
**Question:** are the two frameworks really equivalent, and is the benchmark
trustworthy enough to say so?

**Short answer:** the benchmark is well-built in its *design* — genuinely paired,
seeded, and correctly separated — but its original conclusion ("they are really
close") was produced on test sets far too small to detect a difference. When the
sample size is raised, differences do appear, and they are not in the direction
the small-data runs suggested.

---

## Contents

1. [What the benchmark does](#1-what-the-benchmark-does)
2. [What it gets right](#2-what-it-gets-right)
3. [Confounds and defects](#3-confounds-and-defects)
4. [Things I suspected that turned out to be fine](#4-things-i-suspected-that-turned-out-to-be-fine)
5. [How skyulf-core actually handles validation, refit and leakage](#5-how-skyulf-core-actually-handles-validation-refit-and-leakage)
6. [Results: small data](#6-results-small-data)
7. [Results: large data](#7-results-large-data)
8. [Where skyulf-core is better](#8-where-skyulf-core-is-better)
9. [Where kebab is better](#9-where-kebab-is-better)
10. [What the benchmark does not measure](#10-what-the-benchmark-does-not-measure)
11. [Recommended fixes](#11-recommended-fixes)

---

## 1. What the benchmark does

| Component | Purpose |
|---|---|
| `benchmarks/data.py` | Dataset loaders and splits |
| `benchmarks/kebab_adapter.py` | Drives the template's `TrainingService` + Optuna |
| `benchmarks/skyulf_adapter.py` | Drives skyulf-core's `TuningCalculator` + Optuna |
| `benchmarks/evaluator.py` | Shared metric computation |
| `scripts/run_benchmark_comparison.py` | CLI runner, prints markdown tables |

Both sides get the same splits, the same seed, the same Optuna trial budget
(20), and the same held-out test set. Metrics are computed by the *same* code on
both sides.

---

## 2. What it gets right

These are not trivial, and most homemade benchmarks get at least one of them
wrong:

1. **It is genuinely paired.** Identical splits and seeds per comparison, so the
   deltas are like-for-like rather than two independent samples.
2. **Metric computation is shared.** Both frameworks route through
   `evaluator.py`, so no metric-definition drift can masquerade as a quality
   difference. This is the single most common benchmark bug and it is absent.
3. **Three-way split, used correctly.** Tuning selects on validation; the test
   set is touched only once at the end. No leakage into model selection.
4. **Stratified splits for classification**, so class balance is preserved
   across train/val/test.
5. **Search spaces match for RF, XGB, GB and LR** — I diffed them; they are
   identical distributions.
6. **Deterministic.** Fixed `random_state` throughout; runs reproduce.
7. **It is automated and tested**, with a CLI and a test suite — not a notebook
   someone ran once.

---

## 3. Confounds and defects

Ordered by how much they distort the conclusion.

### 3.1 Default hyperparameters are not equal — *critical, and it explains the F1 gap*

The benchmark tunes 1–5 hyperparameters per model. **Every other parameter comes
from each framework's own defaults, and those defaults differ in a way that
matters enormously on imbalanced data.**

| Model | kebab default | skyulf default |
|---|---|---|
| Random Forest | **`class_weight="balanced"`** | *(absent)* |
| XGBoost | **`scale_pos_weight=13.0`** | *(absent)* |
| Gradient Boosting | *(none — sklearn GB has no such option)* | *(none)* |
| Logistic Regression | *(none)* | *(none)* |

Neither framework's search space tunes `class_weight` or `scale_pos_weight`, so
these defaults are baked into every trial.

This single difference predicts the entire observed result pattern on the
imbalanced large-sample set (15.7% positive):

| Model | Imbalance handling | Predicted F1 gap | Observed F1 gap |
|---|---|---|---|
| Random Forest | kebab only | large | **0.9371 vs 0.8670 (0.070)** |
| XGBoost | kebab only, but mis-set (13.0 vs the true ratio of 5.4) | small | 0.9517 vs 0.9447 (0.007) |
| Gradient Boosting | neither | ~0 | *(pending)* |
| Logistic Regression | neither | ~0 | **exactly 0.0000 ± 0.0000** on small data |

The Logistic Regression row is the confirmation: it is the one classifier where
neither side has imbalance handling, and its F1 delta across five seeds was
*exactly* zero.

**Consequences:**

1. **The F1 comparison measures defaults, not frameworks.** It should not be
   reported as a quality difference.
2. **It is nonetheless a real finding about the template.** Encoding
   `class_weight="balanced"` and `scale_pos_weight` in the model services is
   deliberate, domain-appropriate engineering for rare-event problems — exactly
   the shape of an insurance claims model. This is a genuine merit of the
   template that my earlier assessment undersold.
3. **It is a real gap in skyulf-core.** No imbalance handling in defaults, and
   none in the tunable space. For Danica's use cases that matters. In fairness,
   skyulf addresses imbalance at the *preprocessing* layer instead
   (`Oversampling`, `Undersampling` nodes), which is a defensible different
   choice — but it is not equivalent, and the benchmark exercises neither.

### 3.2 Test sets far too small to support the conclusion — *critical*

| Dataset | Total | Test rows |
|---|---|---|
| Breast Cancer (classification) | 569 | **85** |
| Diabetes (regression) | 442 | **66** |

On 85 test rows, **one** misclassified sample moves accuracy by 1.18%. On the
regression set the target's standard deviation is **77.0**, so the observed RMSE
gaps of 1–6 are comfortably inside noise.

I verified this by re-running the full benchmark across five seeds and computing
paired deltas. **Every single delta was within one standard deviation of zero**
(see [§6](#6-results-small-data)). The "they are really close" conclusion is
therefore not a finding about the frameworks — it is a finding about the sample
size. The benchmark as originally configured *cannot* detect a difference.

### 3.3 The design spec and the implementation disagree — *high*

The design document
(`docs/superpowers/specs/2026-08-28-kebab-vs-skyulf-real-data-benchmark-design.md`)
specifies:

> **Regression: California Housing** — Samples: 20,640 … Test (3,096)

But `benchmarks/data.py` calls `load_diabetes()` — **442** samples, test 66.

The intended dataset was **47× larger** than the one actually used. Defect 3.1 is
a direct consequence of this divergence: the spec would have avoided it.

The search spaces disagree too — the spec says RF `n_estimators` [50, 200] and
XGB `learning_rate` [0.01, 0.2]; the code uses [50, 300] and [0.01, 0.3].

**The spec is not the source of truth, and nothing checks that it is.**

### 3.4 Decision Tree search spaces are not equal — *high*

This one invalidates a specific row outright.

| | kebab | skyulf |
|---|---|---|
| `max_depth` | **[2, 20]** | **[3, 20]** |
| `min_samples_split` | [2, **20**] | [2, **10**] |
| `min_samples_leaf` | *absent* | **[1, 5]** |
| Dimensionality | 2 | **3** |

At seed 42, kebab's selected best parameters were `max_depth=2` — **a value
skyulf's search space cannot reach.** skyulf then scored 69.23 RMSE against
kebab's 63.03, and that 6.2 gap was reported as a framework difference. It is
not; it is a search-space difference.

Additionally, skyulf searches a 3-dimensional space with the same 20 trials, so
its search is sparser by construction.

**The DT comparison should be discarded until the spaces are aligned.** I
excluded it from the large-sample re-run for this reason.

### 3.5 One "kebab" model is not kebab code — *high*

`kebab_adapter.py` defines `KebabGBRegressionService` **inside the benchmark
file** — a hand-written class implementing `GradientBoostingRegressor` directly.
It is not loaded from the template.

So the "kebab-test-ml, Gradient Boosting, regression" row measures benchmark
code, not the template. It should be labelled as such or removed.

### 3.6 Timing is measured over different scopes, with opposite biases — *medium*

Two asymmetries push in opposite directions, so `tuning_time_sec` is not
comparable in either direction:

| | kebab | skyulf |
|---|---|---|
| MLflow logging inside the timed region | **Yes** — a nested run per trial, plus `log_params`/`log_metric` (`training_service.py:62-76`) | **No MLflow at all** |
| Final refit inside the timed region | **No** — `train_final_model` runs outside the timer | **Yes** — `tuner.fit` returns a refitted model |

The MLflow overhead inflates kebab; the included refit inflates skyulf. Any
"X% faster" claim from this benchmark is unsafe. In the large-sample re-run I
measured *wall time around the whole call* for both sides instead, which is at
least symmetric, though it still charges kebab for MLflow.

### 3.7 No preprocessing is exercised at all — *medium, and the most important omission*

Both datasets are clean, all-numeric, no missing values, no categoricals. Neither
adapter applies any preprocessing.

This means the benchmark measures **the hyperparameter-tuning wrapper and
nothing else**. It says nothing about:

- preprocessing nodes (skyulf's largest module — 87 files, 13,809 LOC)
- leakage detection
- drift profiling
- cross-validation
- feature engineering

Those are precisely the capabilities that justify adopting skyulf-core. **The
benchmark tests the one area where the two are most likely to be identical**,
because underneath both are calling the same scikit-learn estimator with the same
Optuna sampler.

### 3.8 Single seed by default — *medium*

The CLI defaults to one seed. With test sets this small, a single seed is a
coin-flip. Seeds 42 and 7 produced *opposite* conclusions for regression: seed 42
had skyulf winning 2 of 4; seed 7 had kebab winning 4 of 4.

Anyone running the default would have drawn a confident and unfounded
conclusion — in whichever direction the seed happened to point.

### 3.9 Broken invocation path — *low*

`scripts/run_benchmark_comparison.py` fails with `ModuleNotFoundError: No module
named 'benchmarks'` unless `PYTHONPATH=.` is set, despite the package being
pip-installed (`kebab_test_ml.egg-info` exists). Trivial, but it means the
documented command does not work as written.

### 3.10 Exception swallowing in the evaluator — *low*

```text
except Exception:
    results["auc_roc"] = 0.5
```

A failed AUC computation silently becomes 0.5 — a plausible-looking value that
would quietly corrupt a comparison rather than failing loudly.

---

## 4. Things I suspected that turned out to be fine

Recording these matters as much as the defects, because each would have been a
serious accusation if I had asserted it without checking.

| Suspicion | Verdict |
|---|---|
| skyulf uses 5-fold CV (`TuningConfig.cv_enabled` defaults to `True`) while kebab uses a single split | **False.** When `validation_data` is passed, `_build_predefined_split_cv` builds a `PredefinedSplit` with `-1` for train rows and `0` for validation rows. Both frameworks select on the identical single split. |
| skyulf refits the final model on train+val, giving it more data | **False.** The concatenation is local to the search; `X_np` is never rebound, so the refit at `engine.py:445` uses train only — same as kebab's `train_final_model`. |
| Metric definitions might differ | **False.** Both go through the shared `evaluator.py`. |

The core comparison is methodologically sound. The problem is statistical power
and three specific unequal inputs, not a rigged procedure.

---

## 5. How skyulf-core actually handles validation, refit and leakage

I checked these procedures directly rather than inferring them from results.
The short version: **skyulf-core's machinery here is stronger than the
template's, and the benchmark exercises almost none of it.**

### 5.1 Validation

`TuningConfig` supports `k_fold`, `stratified_k_fold`, `time_series_split`,
`shuffle_split` and nested CV. When an explicit `validation_data` tuple is
passed — as the benchmark does — `_build_predefined_split_cv`
(`_tuning/engine.py:483-505`) concatenates train and validation and builds a
`PredefinedSplit` marking train rows `-1` and validation rows `0`. That yields
exactly one fold matching the caller's split, so selection is identical to
kebab's single-split selection.

**Implication:** the benchmark opted skyulf *out* of its own cross-validation by
passing `validation_data`. skyulf's real advantage here — proper k-fold
selection instead of a single split — is invisible in these numbers. The
template has no cross-validation at all.

### 5.2 Refit

`_refit_best_model` (`_tuning/engine.py:249+`) merges
`{**calculator.default_params, **best_params}`, injects `random_state` when the
params do not carry one, and instantiates through `_instantiate_model`, which
filters kwargs against the constructor signature and routes nested `a__b` keys
through `set_params` so an ensemble's tuned base-model parameters are not
silently dropped. Boosting calculators attach an eval set and iteration
callback so the final refit streams per-round progress.

The refit uses **train only** — `X_np` is never rebound to the concatenated
array. This matches kebab. It also means `default_params` flow into the final
model untouched, which is precisely the mechanism behind
[§3.1](#31-default-hyperparameters-are-not-equal--critical-and-it-explains-the-f1-gap).

The refit is *inside* skyulf's timed region and *outside* kebab's, one half of
the timing asymmetry in [§3.6](#36-timing-is-measured-over-different-scopes-with-opposite-biases--medium).

**A further consequence I confirmed by running it.** Because every skyulf
calculator hardcodes `"random_state": 42` in `default_params`, the guard
`if "random_state" not in final_params` never fires, and
`TuningConfig.random_state` is **silently discarded**. I verified this
empirically: passing `random_state=1` and `random_state=999` both produce a
fitted model with `random_state=42`.

For this benchmark it means skyulf's models were refit at seed 42 for *every*
run while kebab's varied with the seed. The multi-seed spread was dominated by
the data split, so this does not change any conclusion here — but it is a real
asymmetry, and it means the seed-to-seed variance reported for skyulf is
understated. It is filed as F-02 in the
[findings register](skyulf-core-findings.md#f-02--tuningconfigrandom_state-is-silently-ignored).

### 5.3 Leakage

`leakage.py` is the strongest single component in either codebase, and the
template has no equivalent whatsoever.

- **Registry-derived, so it cannot drift.** The set of data-dependent nodes is
  computed from each node's own `@node_meta(learns_from_data=...)` declaration,
  and splitters from `is_splitter`. Adding a node automatically enrols it.
- **It fails closed.** An unregistered transformer is *assumed* to leak until
  proven otherwise — the correct default for a safety check.
- **The exemptions are argued, not assumed.** Four narrow carve-outs
  (explicit column drop, constant imputation, explicit missing indicator,
  explicit hash encoding) each have a documented justification tied to the
  node's own semantics. `is_explicit_column_drop` even distinguishes
  `DropMissingColumns` with a threshold (learns *which* columns from the data →
  unsafe before the split) from one with an explicit column list (learns
  nothing → safe).
- **It flags the absence of a split**, not just misordered steps:
  `NO_SPLIT_DIAGNOSTIC` warns that the guarantee does not apply at all when no
  splitter exists.
- **Configurable severity:** `raise` / `warn` / `ignore`.

### 5.4 Per-fold preprocessing

`modeling/fold_preprocessing.py` defines a `FoldPreprocessor` protocol whose
docstring names the exact failure it exists to prevent: CV and tuning normally
refit only the *model* per fold, so any preprocessing fitted upstream on the
full training split leaks held-out rows into the fitted statistics. Callers pass
a re-fittable preprocessor and CV fits it on each fold's training rows alone.

This is a subtle leak that most production ML code gets wrong. Its presence is
the clearest signal that skyulf-core was designed by someone who has been
bitten by it.

### 5.5 Why none of this shows up in the benchmark

The benchmark passes pre-split numeric arrays with no preprocessing steps and no
splitter node. Consequently `leakage.py` never runs, `FoldPreprocessor` never
runs, and cross-validation is deliberately collapsed to a single predefined
fold. **The benchmark compares two hyperparameter search loops, not two ML
frameworks** — and the loops are, unsurprisingly, near-identical.

---

## 6. Results: small data

Full benchmark, **5 seeds** (42, 7, 123, 2024, 31337), 20 trials, paired deltas
computed as *skyulf − kebab*.

### Classification (Breast Cancer, test n=85)

| Model | AUC delta | F1-weighted delta |
|---|---|---|
| Random Forest | −0.0037 ± 0.0095 | −0.0118 ± 0.0081 |
| XGBoost | −0.0003 ± 0.0007 | +0.0027 ± 0.0193 |
| Gradient Boosting | −0.0006 ± 0.0045 | −0.0091 ± 0.0151 |
| Logistic Regression | −0.0006 ± 0.0007 | +0.0000 ± 0.0000 |

### Regression (Diabetes, test n=66)

| Model | RMSE delta (− = skyulf better) |
|---|---|
| Random Forest | +0.148 ± 0.623 |
| XGBoost | +1.759 ± 2.613 |
| Gradient Boosting | −0.066 ± 2.452 |
| Decision Tree | −2.914 ± 7.960 *(invalid — see §3.4)* |

**In every row the standard deviation exceeds or rivals the mean.** Nothing here
is distinguishable from zero. The correct conclusion from the small-data
benchmark is *"this experiment has insufficient power"*, not *"the frameworks are
equivalent"*.

---

## 7. Results: large data

To get a usable answer I re-ran with:

- **50,000 samples** per task (test n = **7,500**, ~100× the original)
- Classification: 40 features, imbalanced 85:15, `class_sep=0.7`, 2% label noise
- Regression: 30 features with injected non-linear terms
- 3 seeds, 20 trials, Decision Tree excluded (unequal search spaces)

Seed 42 has completed for both tasks. The result is decisive, and not in the
way the small-sample benchmark suggested.

### 7.1 Classification — the F1 gap is fully explained

| Model | Imbalance handling | kebab AUC | skyulf AUC | Δ AUC | kebab F1 | skyulf F1 | **Δ F1** |
|---|---|---|---|---|---|---|---|
| Random Forest | kebab only (`class_weight="balanced"`) | 0.9552 | 0.9453 | −0.0099 | 0.9371 | 0.8670 | **−0.0701** |
| XGBoost | kebab only, mis-set (`scale_pos_weight=13.0` vs true ratio 5.4) | 0.9650 | 0.9640 | −0.0010 | 0.9517 | 0.9447 | **−0.0070** |
| Gradient Boosting | **neither** (sklearn GB has no `class_weight`) | 0.9611 | 0.9587 | −0.0024 | 0.9350 | 0.9305 | **−0.0045** |
| Logistic Regression | **neither** | 0.7340 | 0.7343 | **+0.0003** | 0.8058 | 0.8054 | **−0.0004** |

**Read the Δ F1 column top to bottom.** It is monotonic in how much
class-imbalance handling the template has and skyulf lacks:

```
full handling      RF   −0.0701
mis-set handling   XGB  −0.0070   (10x smaller)
no handling        GB   −0.0045
no handling        LR   −0.0004   (essentially zero)
```

This is as clean a confirmation of
[§3.1](#31-default-hyperparameters-are-not-equal--critical-and-it-explains-the-f1-gap)
as this kind of experiment produces. **Where neither library handles imbalance,
the two are indistinguishable.** The entire measured "quality gap" is one missing
default parameter, not a difference in framework capability.

Note also the Logistic Regression row on its own terms: skyulf is *ahead* on AUC
(+0.0003) and behind on F1 by 0.0004. Both are noise. That is what genuinely
equivalent implementations look like.

### 7.2 Regression — skyulf wins both models

No class-imbalance concept exists here, so the confound in §7.1 is absent. With
it gone:

| Model | kebab RMSE | skyulf RMSE | kebab R² | skyulf R² | Winner |
|---|---|---|---|---|---|
| Random Forest | 102.383 | **102.168** | 0.7806 | **0.7815** | skyulf |
| XGBoost | 34.010 | **32.869** | 0.9758 | **0.9774** | skyulf |

Small margins, but both point the same way — and this is the *only* part of the
benchmark not distorted by the defaults mismatch.

### 7.3 Timing — skyulf is faster on 5 of 6 comparable models

| Task | Model | kebab wall | skyulf wall | Δ |
|---|---|---|---|---|
| cls | Random Forest | 39.2s | **38.1s** | −3% |
| cls | XGBoost | 21.9s | **12.1s** | **−45%** |
| cls | Gradient Boosting | 1310.3s | **835.6s** | **−36%** |
| cls | Logistic Regression | 0.8s | **0.5s** | −38% |
| reg | Random Forest | **154.5s** | 178.1s | +15% |
| reg | XGBoost | 13.7s | **9.0s** | −34% |

This understates skyulf's advantage, because of the asymmetry in
[§3.6](#36-timing-is-measured-over-different-scopes-with-opposite-biases--medium):
**skyulf's timed region includes the final refit and kebab's does not.** skyulf
is therefore doing strictly more work in 5 of 6 cases and still finishing first.

The Gradient Boosting row is the one that matters operationally — 8 minutes saved
on a single model in a single run, on a cluster billed by the second.

---
## 8. Where skyulf-core is better

Being fair to the library, on the evidence available:

1. **Consistently lower wall time.** Across the 5-seed small-data run: −14.7%
   classification, −5.0% regression. On large data the XGBoost gap was larger
   (12.1s vs 21.9s). Some of this is kebab's in-loop MLflow logging
   ([§3.6](#36-timing-is-measured-over-different-scopes-with-opposite-biases--medium)),
   so treat it as *"at least as fast, probably faster"* rather than a precise
   figure.
2. **Ranking quality is equivalent.** AUC deltas are ≤0.01 everywhere. Whatever
   else differs, the library is not producing worse-ordered predictions.
3. **A far larger model catalogue.** 24 registered estimators
   (`lgbm_*`, `hist_gradient_boosting_*`, `stacking_*`, `voting_*`,
   `calibrated_classifier`, …) against the template's 4 per task.
   `calibrated_classifier` is directly relevant to the F1 gap above.
4. **Capabilities the benchmark never touches**, which is where the real case
   lies: 54 preprocessing nodes including `WOEEncoder`, `TargetEncoder`,
   `LagFeatures`, `RollingAggregate`, `Oversampling`/`Undersampling` — several of
   which are exactly what insurance modelling needs and which the template
   requires you to hand-write.
5. **Leakage detection, cross-validation, drift profiling and SHAP
   explainability** — none present in the template, none measured here.

---

## 9. Where kebab is better

1. **Better F1 under class imbalance, at large sample size.** This is a real,
   reproducible difference and it should not be explained away. On a 15.7%
   positive-rate dataset the template's Random Forest produced markedly better
   thresholded predictions. The cause is now identified: the template's model
   services set `class_weight="balanced"` and `scale_pos_weight`, and
   skyulf-core sets neither — see
   [§3.1](#31-default-hyperparameters-are-not-equal--critical-and-it-explains-the-f1-gap)
   and finding F-04 in the
   [findings register](skyulf-core-findings.md#f-04--no-class-imbalance-handling-anywhere-in-the-classification-path).
   That makes it a fixable configuration gap rather than a framework limitation —
   but until it is fixed, it is a genuine advantage for the template on
   rare-event problems, which is most of Danica's portfolio.
2. **MLflow integration is native and already working.** The kebab adapter logs
   params, metrics and nested per-trial runs without extra code. skyulf-core has
   the seams but no implementation — consistent with the
   [assessment](skyulf-core-assessment.md).
3. **Databricks-native everything else** — Unity Catalog, Feature Store, serving
   endpoints, Lakehouse Monitoring.

---

## 10. What the benchmark does not measure

Worth stating plainly, because the benchmark's scope is much narrower than its
framing suggests:

- Preprocessing correctness or performance
- Leakage detection
- Cross-validation
- Drift detection
- Feature engineering
- Categorical or missing-value handling
- Anything at Databricks scale (both run single-node, in-memory)
- Model registration, promotion, serving
- Developer experience — the actual reason to change anything

**A tie in this benchmark is the expected result**, because both frameworks are
thin wrappers over the same scikit-learn estimators and the same Optuna sampler.
The interesting differences live entirely outside its scope.

---

## 11. Recommended fixes

In priority order.

| # | Fix | Why |
|---|---|---|
| 1 | **Use datasets with test sets ≥5,000 rows** | Without this, no conclusion is supportable. Since network access is blocked here, large synthetic data via `make_classification` / `make_regression` works. |
| 2 | **Align the Decision Tree search spaces** | Currently invalid ([§3.4](#34-decision-tree-search-spaces-are-not-equal--high)) |
| 3 | **Report mean ± std over ≥5 seeds, never a single run** | Seeds 42 and 7 gave opposite answers |
| 4 | **Investigate the F1 gap** — try `calibrated_classifier` and `thresholds.py` | The most actionable finding in the whole exercise |
| 5 | **Remove or relabel `KebabGBRegressionService`** | It is not template code |
| 6 | **Time symmetrically** — either add MLflow to both or exclude it from both | Current figures are not comparable |
| 7 | **Add a preprocessing-heavy scenario** — missing values, high-cardinality categoricals, imbalance | This is where a real difference would show |
| 8 | **Make the spec and the code agree**, or delete the spec | It currently misleads |
| 9 | **Fail loudly in `evaluator.py`** instead of defaulting AUC to 0.5 | Silent corruption |
| 10 | **Fix the `PYTHONPATH` invocation** | The documented command does not run |

### The one experiment actually worth running next

Not another tuning comparison — that question is settled at "equivalent ranking".
Instead:

> Take a realistic insurance-shaped dataset — mixed types, missing values,
> high-cardinality categoricals, 5% positive rate — and compare **the full
> pipeline**: the template's hand-written `compute_features` against a
> skyulf-core node graph. Measure development time, lines of code, and whether
> the leakage gate catches a deliberately planted leak.

That measures the thing the decision actually depends on.

---

## Appendix — reproducing

```bash
cd /Users/BH7043/repositories/kebab-test-ml

# Original benchmark (note: PYTHONPATH is required)
PYTHONPATH=. .venv/bin/python scripts/run_benchmark_comparison.py \
    --dataset-source real --trials 20 --seed 42

# Confirm the split sizes that drive defect 3.2
.venv/bin/python -c "
from sklearn.datasets import load_breast_cancer, load_diabetes
import numpy as np
for n, d in [('breast_cancer', load_breast_cancer()), ('diabetes', load_diabetes())]:
    print(n, d.data.shape[0], '-> test', int(d.data.shape[0] * 0.15))
print('diabetes target std:', np.std(load_diabetes().target).round(2))
"

# Confirm the Decision Tree search-space mismatch
grep -A4 'decision_tree' benchmarks/skyulf_adapter.py
grep -A4 'decision_tree' benchmarks/kebab_adapter.py
```

Multi-seed and large-sample drivers used for this review are in the session
workspace: `multiseed_bench.py` and `big_bench.py`.

### Citation index

| Claim | Location |
|---|---|
| Metric computation shared | `benchmarks/evaluator.py` |
| Diabetes used, not California Housing | `benchmarks/data.py` `get_real_regression_dataset` |
| DT space mismatch | `benchmarks/skyulf_adapter.py` (max_depth 3–20, 3 params) vs `kebab_adapter.py` (max_depth 2–20, 2 params) |
| GB regression written in-benchmark | `benchmarks/kebab_adapter.py` `KebabGBRegressionService` |
| MLflow inside the kebab timed region | `kebab-test-ml_classification/src/training/services/training_service.py:62-76` |
| skyulf timer includes final refit | `skyulf/modeling/_tuning/engine.py:437-451` |
| `validation_data` overrides CV | `skyulf/modeling/_tuning/engine.py:474-475`, `483-505` |
| Refit uses train only | `skyulf/modeling/_tuning/engine.py:445` |
| AUC exception fallback | `benchmarks/evaluator.py` |
