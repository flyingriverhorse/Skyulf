# Bounds and outliers

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps apply fixed bounds, clip values or remove outlier rows during
training. Read each card for whether it changes values or removes rows.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

`ManualBounds` removes rows; `ClipValues` changes values and preserves rows.
The learned outlier filters below belong in training, never pre-split. Their appliers
can remove rows; ordinary prediction rejects row-count loss rather than silently
returning fewer predictions. A scoring eligibility policy must be explicit.

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

**Minimal YAML:**

```yaml
- name: winsorize
  transformer: Winsorize
  params: {columns: [income], lower_percentile: 5, upper_percentile: 95}
```

**Common mistake:** Use percent values such as 5 and 95, not shares such as 0.05 and 0.95.

**Details:** Saves bounds; applies clipping in place. Missing values remain missing, constant
columns retain their constant and numeric percentile bounds may require float
output. Context: row. Dependencies: base.

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
