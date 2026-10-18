# Bounds and outliers

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps apply fixed bounds, clip values or remove outlier rows during
training. Read each card for whether it changes values or removes rows.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

`ManualBounds` removes rows; `ClipValues` changes values and preserves rows.
The learned outlier filters below belong in training, never pre-split. Their appliers
can remove rows. Saved pipeline scoring retains each request row and reports
excluded rows with a null prediction and an `outlier:<step>` reason.

## ManualBounds

**Purpose and effect:** lower 0, upper 100 retains 0 and 100 and removes -1 and 101; missing values
are retained.

**File:** `config/preprocessing.yml`, or `config/pre_split.yml` under the
[narrower admission rules](preprocessing_recipes.md#pre-split-admission-and-scoring). Training-only
row changes need an explicit
[scoring decision](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `bounds` | mapping | {} | Column→{lower, upper}; each bound number/null. | Filter rows outside fixed inclusive bounds. |
| `bounds.<column>.lower, upper` | number/null | null | pre_split requires at least one finite bound per column and lower ≤ upper. | Set each side; omitted sides are unbounded. |

**Minimal YAML:**

```yaml
- name: manualbounds
  transformer: ManualBounds
  params: {bounds: {age: {lower: 0, upper: 120}}}
```

**Common mistake:** Missing values survive; combine with a completeness rule if they are ineligible.

**Details:** Saves fixed bounds. Rows violating any bound are filtered with paired
targets and survivor order preserved. Context: row. Dependencies: base.

## ClipValues

**Purpose and effect:** `[-1, 50, 200]` with bounds 0–100 becomes `[0, 50, 100]`.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `bounds` | mapping | required | Nonempty existing numeric column→{lower, upper}. | Set clipping scope. |
| `bounds.<column>.lower, upper` | number/null | null | At least one finite bound; lower ≤ upper. | Cap values while preserving rows. |

**Minimal YAML:**

```yaml
- name: clipvalues
  transformer: ClipValues
  params: {bounds: {income: {lower: 0, upper: 1000000}}}
```

**Common mistake:** ClipValues preserves rows and is not admitted in pre_split; use ManualBounds to
filter there.

**Details:** Missing values pass through.
Saves fixed bounds, replaces values in place, never filters rows. Integer bounds
must be representable if the integer dtype is preserved; cast explicitly when
fractional arithmetic is intended. Context: row. Dependencies: base. Not admitted in pre-split,
despite using fixed values.

## IQR

**Purpose and effect:** remove observations outside training Q1/Q3 fences.

**File:** `config/preprocessing.yml`. Training-only row changes need an explicit
[scoring decision](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `multiplier` | number | 1.5 | Numeric fence multiplier. | Set distance beyond fitted Q1 and Q3. |

**Minimal YAML:**

```yaml
- name: iqr
  transformer: IQR
  params: {columns: [income], multiplier: 1.5}
```

**Common mistake:** This removes rows; ordinary prediction rejects row-count loss. Use Winsorize to
retain rows.

**Details:** Saves Q1 − multiplier × IQR and Q3 + multiplier × IQR per column. Apply removes
rows outside any learned fence while preserving missing values and aligning y.
A zero IQR gives equal fences. No request-time quartile fit. Context: row against
saved bounds. Dependencies: base.

## ZScore

**Purpose and effect:** remove observations farther than the configured number of fitted standard
deviations from the training mean.

**File:** `config/preprocessing.yml`. Training-only row changes need an explicit
[scoring decision](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `threshold` | number | 3.0 | Numeric standard-deviation threshold. | Set allowed distance from the training mean. |

**Minimal YAML:**

```yaml
- name: zscore
  transformer: ZScore
  params: {columns: [income], threshold: 3.0}
```

**Common mistake:** Do not expect row-removing outlier filters to return one ordinary prediction per
input.

**Details:** Saves mean/standard-deviation statistics and threshold; filters paired rows
using those statistics. Missing values are preserved; unusable/zero scales
cannot provide a meaningful outlier distance. No refit on heldout/request rows.
Context: row. Dependencies: base.

## Winsorize

**Purpose and effect:** values beyond fitted 5th/95th percentiles are capped to the corresponding
saved percentile, retaining every row.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `lower_percentile` | number | 5.0 | 0–100; no greater than upper_percentile. | Learn the lower cap. |
| `upper_percentile` | number | 95.0 | 0–100; no less than lower_percentile. | Learn the upper cap. |
| `percentiles_by_column` | mapping | omitted | Selected column → {lower, upper}; both finite, ordered percentages required. | Override the global tails for individual measurements. |

**Minimal YAML:**

```yaml
- name: winsorize
  transformer: Winsorize
  params: {columns: [income], lower_percentile: 5, upper_percentile: 95}
```

Different measurements can use different tails while other selected columns
keep the global settings:

```yaml
- name: measurement_tails
  transformer: Winsorize
  params:
    columns: [income, tenure]
    lower_percentile: 5
    upper_percentile: 95
    percentiles_by_column:
      income: {lower: 1, upper: 99}
```

**Common mistake:** Use percent values such as 5 and 95, not shares such as 0.05 and 0.95.
Integer columns require finite, integral bounds within their dtype's range. Add an explicit
Casting step to float64 before Winsorize when fractional percentile bounds are intended;
the node does not silently round the bounds or promote integer values.

When `columns` is omitted, automatic numeric selection also includes integer IDs
and counts. If a selected integer column has fractional percentile bounds, the
whole fit raises.
Set `columns` explicitly to keep IDs out of clipping. Cast only the measurements
that should support fractional bounds, for example:

```yaml
- name: fractional_income
  transformer: Casting
  params: {columns: [income], target_type: float64}
- name: cap_income
  transformer: Winsorize
  params: {columns: [income], lower_percentile: 5, upper_percentile: 95}
```

This recipe leaves a separate `customer_id` column unchanged, including integers
above `2**53`; it does not cast every numeric column to float.

**Details:** Saves bounds; applies clipping in place. Missing values remain missing, constant
columns retain their constant. Integer 0th/100th-percentile endpoints remain exact; interpolated
integer quantiles outside the exact float64 range also require explicit Casting. Automatic column
selection follows the same precision rule. Context: row. Dependencies: base.

## EllipticEnvelope

**Purpose and effect:** remove observations classified as outside the fitted robust envelope.

**File:** `config/preprocessing.yml`. Training-only row changes need an explicit
[scoring decision](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `contamination` | number | 0.01 | Greater than 0 and at most 0.5. | Set expected outlier fraction. |
| `random_state` | integer/null | 42 | Integer seed or null. | Control fitting randomness. |

**Minimal YAML:**

```yaml
- name: ellipticenvelope
  transformer: EllipticEnvelope
  params: {columns: [income], contamination: 0.01, random_state: 42}
```

**Common mistake:** This wrapper fits separate envelopes per column; it is not a multivariate
covariance filter across all inputs.

**Details:** Fits a separate sklearn
envelope for each selected column, rather than one multivariate envelope across
all selected columns. Saves these models. Apply filters rows and paired y;
missing values are handled separately from finite fit observations. Small or
degenerate fits can be skipped with a warning. Context: row against saved
models; training fit is population-dependent. Dependencies: base/sklearn.

## IsolationForest

**Purpose and effect:** fit one joint isolation forest across the selected
features and exclude rows classified as outliers. This can detect unusual
combinations that separate per-column filters miss.

**File:** `config/preprocessing.yml`; this learned node does not belong before
the train/test split.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Numeric features; [] selects none. | Select joint axes; implicit target excluded. |
| `contamination` | number/string | 0.01 | (0, 0.5] or auto. | Set the learned outlier threshold. |
| `n_estimators` | integer | 100 | Positive integer. | Set the number of isolation trees. |
| `max_samples` | integer/number/string | auto | Positive count, fraction in (0, 1], or auto. | Set each tree's training population. |
| `random_state` | integer/null | 42 | Valid sklearn seed or null. | Control training randomness. |

```yaml
- name: joint_outliers
  transformer: IsolationForest
  params: {columns: [income, tenure], contamination: 0.01, random_state: 42}
```

**Details:** learns from complete finite training rows and requires at least two
such rows. At training and apply, a row with a missing or infinite value in **any**
selected column is retained without outlier screening, even if another selected
value is extreme. Retention does not mean the detector judged that row normal.
Normalize infinite values and impute missing values before this step if partially
observed rows must also be screened. Fit failures propagate. Saved trees never
refit on request data. Row filtering preserves paired targets and sample-weight
positions, including duplicate indexes. Saved scoring publishes excluded outcomes
in their original positions. Context: local row/filter; native Spark training is
not supported. Worker execution still requires the existing artifact admission.

## LocalOutlierFactor

**Purpose and effect:** learn joint local-density outliers, then compare new
rows with the fixed learned neighbor population.

**File:** `config/preprocessing.yml`; fit only on training data.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Numeric features; [] selects none. | Select joint Euclidean-distance axes. |
| `contamination` | number/string | 0.01 | (0, 0.5] or auto. | Set the learned outlier threshold. |
| `n_neighbors` | integer | 20 | Positive integer; sklearn caps to available neighbors. | Set the local density neighborhood. |

```yaml
- name: density_outliers
  transformer: LocalOutlierFactor
  params: {columns: [income, tenure], n_neighbors: 20, contamination: 0.01}
```

**Common mistake:** training rows and new rows use different sklearn scores.
The pipeline training hook uses `negative_outlier_factor_` against the learned
offset. Saved apply uses novelty prediction for new data. Direct SDK callers
should use `fit_transform_train` for their training representation; calling
novelty apply on training rows is not equivalent.

**Details:** requires two complete finite training rows. At training and apply,
a missing or infinite value in **any** selected column keeps the row without
outlier screening; other observed values in that row are not checked separately.
Normalize infinite values and impute missing values before this step if those
rows must also be screened. Scale features when their units require it. The
model retains its learned neighbor population, so saved artifacts grow with
training size. Targets and sample weights follow the actual training mask;
scoring preserves request positions and exclusion reasons. Context: local
row/filter; worker admission remains explicit, with no native Spark fit support.
