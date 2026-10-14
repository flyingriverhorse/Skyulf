# Split, sampling and diagnostic Core nodes

[Back to the preprocessing catalog](preprocessing_recipes.md)

These cards cover the split nodes, training-only resampling and diagnostic
nodes. Generated projects split through `config/training.yml`, not through these
nodes.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

These cards complete the registry inventory. Generated Bundles own splitting
through training configuration; do not insert split/target-split nodes into their
phase recipes. Sampling is a training-only Core operation, not pre-split
eligibility, and diagnostic nodes observe training without transforming features.
The two split-node YAML examples are standalone Core entries. Sampling and
diagnostic entries can accompany a preprocessing recipe; they do not replace
`training.yml` split/model fields.

## TrainTestSplitter

Deprecated alias: `Split`.

**Purpose and effect:** partitions rows into training/test and optional validation datasets, rather
than adding a feature.

**File:** Standalone Core configuration; not a generated project phase recipe. Use
`config/training.yml` for generated project split/target settings.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `test_size` | number | 0.2 | Fraction leaving viable training rows. | Choose test fraction. |
| `validation_size` | number | 0.0 | Nonnegative; test plus validation less than 1. | Choose optional validation fraction. |
| `random_state` | integer/null | 42 | Integer seed or null. | Control random partitioning. |
| `shuffle` | boolean | true | true, false | Shuffle before splitting. |
| `stratify` | boolean | false | true, false; requires usable class labels and shuffle. | Request class-balanced partitions. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |

**Minimal YAML:**

```yaml
- name: traintestsplitter
  transformer: TrainTestSplitter
  params: {test_size: 0.2, validation_size: 0.0, random_state: 42, shuffle: true, stratify: false}
```

**Common mistake:** Generated projects already split through training.yml; do not insert this node
into their phase recipes.

**Details:** Fractions must leave viable training rows and
test+validation must be less than 1. Saves split policy, then returns a
SplitDataset; paired feature/target membership is aligned. Core warns and falls
back to an unstratified split for an absent in-frame label or classes with fewer
than two examples. Other impossible stratified sizes can fail; stratification
cannot be combined with unshuffled random splitting. Generated Bundle split
validation has its own stricter rules. Context global/training. Dependencies: base/sklearn.

## feature_target_split

**Purpose and effect:** `{age, churn}` becomes feature frame `{age}` plus y.

**File:** Standalone Core configuration; not a generated project phase recipe. Use
`config/training.yml` for generated project split/target settings.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `target_column` | string/null | required | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |

**Minimal YAML:**

```yaml
- name: feature_target_split
  transformer: feature_target_split
  params: {target_column: churn}
```

**Common mistake:** Generated projects already route their target; this is a standalone Core helper.

**Details:** Saves the name; separates that column
from X into y without fitting statistics or changing row order/count. Missing
target input cannot supply a training y; prediction paths use their saved target
routing instead. This is not another holdout splitter. Generated Bundles already
manage the target contract. Context row/routing; base dependencies.

## Oversampling

**Purpose and effect:** balances minority classes by duplication or synthetic numeric observations.

**File:** `config/preprocessing.yml`. Training-only row changes need an explicit
[scoring decision](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `method` | string | smote | random_over, smote, adasyn, borderline_smote, svm_smote, kmeans_smote, smote_tomek | Choose training resampler. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |
| `sampling_strategy` | string/number/mapping | auto | imbalanced-learn strategy, binary float ratio or class-count map. | Choose desired class balance; callable is Python-only. |
| `random_state` | integer/null | 42 | Integer seed or null. | Control randomness. |
| `k_neighbors` | integer | 5 | Positive; relevant synthetic methods. | Set minority-neighbor count. |
| `m_neighbors` | integer | 10 | Positive; borderline/SVM methods. | Set danger/noise-neighbor count. |
| `kind` | string | borderline-1 | borderline-1, borderline-2 | Choose BorderlineSMOTE variant. |
| `svm_estimator` | Python estimator/null | null | Python-only object; not a YAML mapping. | Set SVMSMOTE estimator. |
| `out_step` | number | 0.5 | SVMSMOTE setting. | Set extrapolation step. |
| `kmeans_estimator` | integer/Python estimator/null | null | Integer or null in YAML. | Choose KMeansSMOTE clustering. |
| `cluster_balance_threshold` | number/string | 0.1 | Number or auto. | Choose KMeansSMOTE cluster eligibility. |
| `density_exponent` | number/string | auto | Number or auto. | Choose KMeansSMOTE density weighting. |
| `n_jobs` | integer | -1 | Only forwarded to compatible samplers. | Control supported parallelism. |
| `synthetic_weight` | string | unset | class_mean, uniform; required for weighted synthesis. | Assign class mean weight or 1 to synthetic rows. |

**Minimal YAML:**

```yaml
- name: oversampling
  transformer: Oversampling
  params: {method: random_over, sampling_strategy: auto, random_state: 42}
```

**Common mistake:** Synthetic methods need finite numeric data and enough minority neighbors; this
is training-only, not eligibility.

**Details:** SMOTETomek also removes selected rows after synthesis. Only method-relevant fields act.
Saves the sampling policy, not a prediction transform. Requires valid class y;
SMOTE-family methods need finite numeric features and enough minority neighbors.
Output row count/order changes with y kept aligned. Random over-sampling copies
rows; synthetic modes do not preserve one-to-one source identity. Context global;
optional dependency imbalanced-learn. A Python estimator object cannot be placed
directly in YAML; configure supported scalar options or use a Python workflow.

## Undersampling

**Purpose and effect:** reduce majority-class rows while keeping sampled features/labels aligned.

**File:** `config/preprocessing.yml`. Training-only row changes need an explicit
[scoring decision](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `method` | string | random_under_sampling | random_under_sampling, nearmiss, tomek_links, edited_nearest_neighbours | Choose training resampler. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |
| `sampling_strategy` | string/number/mapping | auto | imbalanced-learn strategy, binary float ratio or class-count map. | Choose retained class balance. |
| `random_state` | integer/null | 42 | Integer seed or null. | Control random sampling. |
| `replacement` | boolean | false | Random under-sampling only. | Allow selected duplicates. |
| `version` | integer | 1 | 1, 2, 3; NearMiss only. | Choose NearMiss algorithm. |
| `n_neighbors` | integer | 3 | Positive; neighborhood methods. | Choose neighbor count. |
| `kind_sel` | string | all | all, mode; edited nearest neighbours. | Choose neighbor agreement rule. |
| `n_jobs` | integer | -1 | Only compatible samplers. | Control supported parallelism. |

**Minimal YAML:**

```yaml
- name: undersampling
  transformer: Undersampling
  params: {method: random_under_sampling, sampling_strategy: auto, random_state: 42}
```

**Common mistake:** Resampling changes training rows and y together; do not use it in pre_split or
ordinary prediction.

**Details:** Neighborhood modes need numeric data and enough
observations. Saves policy; removes/selects training rows, including potential
duplicates when replacement is true; corresponding y and supported weights are
selected consistently. Context global; optional imbalanced-learn dependency.

## DatasetProfile

**Purpose and effect:** Inspect training counts, dtypes, missingness and numeric summaries without
changing features.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `None` | — | — | No configuration fields. | Collect training profile metadata; apply passes through. |

**Minimal YAML:**

```yaml
- name: datasetprofile
  transformer: DatasetProfile
  params: {}
```

**Common mistake:** This is a saved training profile, not a fresh prediction-batch drift test.

**Details:** Fit saves row/column counts, dtype names, missing counts and
numeric summary statistics (including binary, constant and all-missing numeric
columns). For `[1, null, 3]`, missing count is 1. Apply is passthrough; it does
not recompute a prediction-batch profile. Metric names differ between pandas
and Polars descriptions. Fit context global, apply context row/passthrough.
Dependencies: base. This is not a distribution-drift decision.

## DataSnapshot

**Purpose and effect:** Capture a small training-row preview without changing features.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `n_rows` | integer | 5 | Nonnegative. | Save the first n training rows as records. |

**Minimal YAML:**

```yaml
- name: datasnapshot
  transformer: DataSnapshot
  params: {n_rows: 5}
```

**Common mistake:** The artifact stores actual row values; choose intentionally whether sample data
should be retained.

**Details:** Fit records the first n training
rows as records; apply passes data through unchanged. If only three rows exist,
the snapshot has three records. Stores actual input values, not just statistics;
choose whether those values belong in your model artifact. Fit depends on input
ordering, apply is row/passthrough. Dependencies: base.
