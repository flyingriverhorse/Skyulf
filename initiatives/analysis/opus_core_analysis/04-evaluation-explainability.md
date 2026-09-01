# Skyulf Deep Audit (Opus) — Evaluation, thresholds & explainability

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

---

## Evaluation, thresholds, explainability

Metric correctness was verified against sklearn ground truth for **30 metrics**;
all standard paths matched exactly. The findings below are edge cases.

### OC-35
### 🟠 High — Multiclass splits missing a class emit binary-only metrics and null curve points

**File:** `modeling/_evaluation/metrics.py:217-237,361-363`,
`_evaluation/classification.py:109-127`

A 3-class model evaluated on a split whose `y_true` contains only two classes is
treated as **binary**: it gains unweighted `precision`/`recall`/`f1` keys, loses
a computable `log_loss`, and emits ROC points with `null` coordinates.

```text
keys: ['f1','f1_weighted','precision','precision_weighted','recall','recall_weighted','roc_auc_ovr']
binary_precision_added 1.0
Metric 'log_loss' failed ... 2 vs 3. Please provide labels.
expected log_loss with labels=[0,1,2] = 0.5736542308362172
ROC (Class 2) points: [(0.0, nan), (1.0, nan)]  →  JSON "y": null
```

**Impact:** The metric contract varies by split composition, so the frontend
receives a different key set than it expects and renders invalid curve points.

**Fix:** Determine binary-vs-multiclass from `model.classes_` (or the probability
column count), not from `y_true`. Pass `labels=classes` to `log_loss`. Skip or
explicitly mark per-class curves whose one-vs-rest target has a single class.

---

### OC-36
### 🟠 High — F1 threshold tuning picks a pathological threshold on single-class validation

**File:** `modeling/_evaluation/thresholds.py:101-111`, `_tuning/refit.py:167-199`

`tune_decision_thresholds` verifies `model.classes_` has two classes but never
checks that the **validation labels** contain both. With no positives, F1 is
tied at 0 for every candidate, and the strict `>` tie-break keeps the first grid
point — a near-zero threshold:

```text
y_val=[0,0,0,0], positive probabilities=[.01,.20,.40,.49]
f1_pos1            threshold 0.009804  pred [1,1,1,1]  pos_rate 1.0
balanced_accuracy  threshold 0.490196  pred [0,0,0,0]  pos_rate 0.0
```

**Impact:** With `tune_threshold=True` on a small or imbalanced split, the model
persists a threshold that classifies nearly everything positive.

**Fix:** Require `np.unique(y_val)` to cover both classes before tuning;
otherwise skip. Tie-break toward the default 0.5 rule.

---

### OC-37
### 🟡 Medium — Binary PR-AUC is dropped for string-labeled classifiers

**File:** `modeling/_evaluation/metrics.py:324-327`

`average_precision_score(y_arr, proba[:, 1])` is called without `pos_label`, so
sklearn defaults to `pos_label=1` and fails on `"no"`/`"yes"` labels:

```text
Metric 'pr_auc' failed: pos_label=1 is not a valid label. It should be one of ['no','yes']
expected pr_auc (pos='yes') = 0.9166666666666665
```

**Fix:** Pass `pos_label=model.classes_[1]`.

---

### OC-38
### ⚪ Low — Clustering metrics treat DBSCAN `-1` noise as a real cluster

**File:** `modeling/_evaluation/metrics.py:432-459`

```text
labels=[0,0,1,1,-1,-1]
with_noise:    n_clusters=3.0  silhouette=0.856246
without_noise: n_clusters=2.0  silhouette=0.858586
```

Currently latent — no shipped clustering model emits `-1` — but it will bite as
soon as density clustering is added.

**Fix:** Add `exclude_noise` support, or document that all labels count.

---
<a id="oc-146"></a>
### OC-146
### 🔴 Critical — Binary `pr_auc` is computed for the wrong class on `{1, n}` labels: reported 0.32 for a model whose true PR-AUC is 0.97

**File:** `modeling/_evaluation/metrics.py:324-326` (`_add_roc_pr_auc_metrics`)

> Found by closing the coverage gap that [17 — file coverage](./17-file-coverage.md)
> declared: `metrics.py` was the top-ranked unread file, flagged as *"metric
> averaging bugs are severe and silent."* It was.

The binary branch scores against `proba[:, 1]` — which is `P(classes_[1])` —
but passes **no `pos_label`**:

```python
if class_count == 2:
    _try_add_metric(metrics, "roc_auc", roc_auc_score, y_arr, proba[:, 1])
    _try_add_metric(metrics, "pr_auc", average_precision_score, y_arr, proba[:, 1])
    return
```

`average_precision_score` defaults to **`pos_label=1`**. When the label set is
`{1, 2}`, `classes_` is `[1, 2]`, so `proba[:, 1]` is `P(y=2)` while the metric
scores class **1** as positive. The two disagree and PR-AUC is computed for the
inverted problem — with **no exception and no warning**.

This is not a hypothetical encoding: `1/2` is how survey exports, Likert-derived
targets, R-style factor codes and many `1=no, 2=yes` CSVs arrive.

**Measured** (`LogisticRegression`, 400 rows, learnable signal; identical data
relabelled — controls included so a clean result proves the probe works):

| label set | reported `pr_auc` | correct | verdict |
|---|---:|---:|---|
| `(0, 1)` *(control)* | 0.9726 | 0.9726 | ok |
| `(-1, 1)` *(control)* | 0.9726 | 0.9726 | ok |
| `(False, True)` *(control)* | 0.9726 | 0.9726 | ok |
| **`(1, 2)`** | **0.3203** | 0.9726 | ⛔ **silently wrong** |
| **`(1, 5)`** | **0.3203** | 0.9726 | ⛔ **silently wrong** |
| `(2, 3)` | *omitted* | 0.9726 | ⚠️ missing (warned) |
| `(0, 2)` | *omitted* | 0.9726 | ⚠️ missing (warned) |
| `(10, 20)` | *omitted* | 0.9726 | ⚠️ missing (warned) |
| `('no', 'yes')` | *omitted* | 0.9726 | ⚠️ missing (warned) |

Two distinct failure modes, one root cause:

1. **Silently wrong** when `1 ∈ classes_` *and* `1` is the **negative** class
   (`classes_[0] == 1`) — `pos_label=1` is a *valid* label, so sklearn raises
   nothing and returns a confidently incorrect number.
2. **Silently missing** for every other non-`{0,1}` binary label set — `pos_label=1`
   isn't in `classes_`, so sklearn raises and `_try_add_metric` omits the key.
   Less damaging, but PR-AUC vanishes from the report with only a server-side log.

`roc_auc` is **unaffected** in all nine cases (0.9714 throughout): `roc_auc_score`
binarises internally against sorted uniques, which happens to agree with
`proba[:, 1]`. The defect is specific to `pr_auc`.

### The report contradicts itself on screen

`_evaluation/classification.py:97` builds the PR curve **correctly**, passing
`pos_label=classes[1]`, then labels it with that class — but attaches
`metrics.get("pr_auc")` as its `auc`. So a single `CurveData` object carries a
curve drawn for class 2 and an AUC computed for class 1:

```text
PR curve name           : PR (Class 2)
reported pr_auc (metric): 0.32033322171188556   <-- class 1
AUC of the plotted curve: 0.97252917456723930   <-- class 2
```

A user sees a chart hugging the top-right corner with "0.32" printed beside it.
The most likely reaction is to distrust the *chart* and discard a strong model.

**Why this is Critical:** it is wrong model output reaching users with no
diagnostic, on a common label encoding, and it inverts the quality signal
(0.97 → 0.32) rather than merely perturbing it.

**Fix** — one argument, and the codebase already knows the pattern. Thirty lines
above, `_add_binary_unweighted_metrics` resolves `pos_label` explicitly *for this
exact reason* ("sklearn's default `pos_label=1` … raises for non-{0,1} binary
labels"), and `classification.py` does the same for the curve. Only this call
site was missed:

```python
if class_count == 2:
    classes_ = getattr(model, "classes_", None)
    pos_label = classes_[1] if classes_ is not None and len(classes_) == 2 else 1
    _try_add_metric(metrics, "roc_auc", roc_auc_score, y_arr, proba[:, 1])
    _try_add_metric(
        metrics, "pr_auc", average_precision_score, y_arr, proba[:, 1],
        pos_label=pos_label,
    )
    return
```

This also fixes failure mode 2, restoring `pr_auc` for string and arbitrary
integer labels. Regression test: assert `pr_auc` is invariant under relabelling
`{0,1} → {1,2} → {"no","yes"}` on fixed data.

---
<a id="oc-147"></a>
### OC-147
### ⚪ Low — `optimize_thresholds` returns a dict shape that bypasses its own documented binary rule, flipping `>=` to `>` on ties

**File:** `modeling/_evaluation/thresholds.py:66-88` (`apply_thresholds`), `:92-108` (`_grid_search_binary`)

`apply_thresholds` documents the binary rule as *"predicts the positive (second)
class when `y_proba[:, 1] >= threshold`"* and implements it in two branches — one
for a bare float, one for a **one**-entry dict.

But `_grid_search_binary` returns a **two**-entry dict
(`{classes[0]: 1 - t, classes[1]: t}`), which matches neither branch and falls
through to the multiclass scaled-argmax path. For binary that path evaluates
`p1/t > p0/(1-t)`, which — since `p0 + p1 == 1` — simplifies to `p1 > t`:
algebraically the documented rule but with **strict** inequality.

Verified: over 500 rows the two paths agree exactly (0 differing rows, F1
0.643087 both ways), so this is *only* a tie-breaking difference — but at a tie
they invert:

```text
p1 == t == 0.5  ->  search rule (>=): [1 1 1 1]
                    apply rule  (> ): [0 0 0 0]
```

Reachable in practice: the grid is `k/102`, so `t = 0.5` is a candidate
(`51/102`), and `p1` of exactly `0.5` is routine for decision trees and small
forests. When both coincide, the score reported by `optimize_thresholds` counts
those rows as positive while production scoring counts them as negative — the
tuned score is then not quite the score you get.

**Fix:** have `_grid_search_binary` return the one-entry form
(`{classes[1]: t}`) that `apply_thresholds` already special-cases, or add
`len(thresholds) == 2 and n_classes == 2` to the binary branch condition. Either
makes the documented `>=` authoritative on both sides.

---
<a id="oc-149"></a>
### OC-149
### 🟠 High — Clustering evaluation crashes on polars when any numeric feature is all-null within one cluster (pandas returns `nan`)

**File:** `modeling/_evaluation/clustering.py:83-88` (`_compute_centroids_polars._column_stats`)

The two dicts built from the same aggregation are guarded inconsistently:

```python
means = casted.select([pl.col(c).mean().alias(c) for c in columns]).row(0)
stds  = casted.select([pl.col(c).std(ddof=1).alias(c) for c in columns]).row(0)
mean_dict = {c: float(v) for c, v in zip(columns, means, strict=True)}          # <-- unguarded
std_dict  = {c: (float(v) if v is not None else 0.0) for c, v in zip(...)}      # <-- guarded
```

Polars returns `None` (not `NaN`) for the mean of an all-null column, so
`float(None)` raises `TypeError`. `_column_stats` is called **per cluster
subset**, so it is enough for one feature to be entirely null *within a single
cluster* — the column can be well-populated overall.

That is not a contrived input. Clusters routinely align with missingness
structure (a measurement taken only for one subpopulation), which is exactly the
case where a feature is fully null inside one cluster.

**Measured** (same frame through both engines; the no-null control passes on
polars, proving the probe is valid):

| case | pandas | polars |
|---|---|---|
| no nulls *(control)* | `{'a': 2.0, 'b': 2.0}`, `{'a': 5.0, 'b': 5.0}` | identical — **MATCH** |
| `b` all-null in cluster 1 | `{'a': 5.0, 'b': nan}` — succeeds | ⛔ `TypeError: float() argument must be a string or a real number, not 'NoneType'` |

The exception is uncaught and propagates out of `evaluate_clustering_model`, so
the entire clustering report is lost rather than one degraded field.

Two independent signals that this is an oversight rather than intent: the `std`
line immediately below guards `None`, and `_auto_profile_label` already
short-circuits on `mean is None` (`if mean is None or not std: continue`) — the
`None` case was anticipated everywhere except here.

**Fix** — mirror the guard already used one line below, choosing `nan` to match
the pandas path rather than `0.0` (a fabricated `0.0` centroid would corrupt the
z-scores in `_auto_profile_label` and invent a "High/Low" profile):

```python
mean_dict = {
    c: (float(v) if v is not None else float("nan"))
    for c, v in zip(columns, means, strict=True)
}
```

---
