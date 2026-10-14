# Scaling and numeric transformations

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps rescale numeric columns or apply fixed and fitted numeric
transformations. They fit on training rows and reuse the saved state at
prediction time.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

All four scalers below replace selected numeric columns and save the fitted
statistics. `columns` is an optional list with numeric auto-detection when
absent and no selection for `[]`. Constant scales use 1 instead of dividing by
zero; NaN is not an imputed value. State and column order stay fixed on apply.
Use an imputer first if the model cannot accept missing values. Dependencies:
base; context: row. Each is for `config/preprocessing.yml` only.

## StandardScaler

**Purpose and effect:** Center and scale numeric inputs; [1, 3] becomes [-1, 1] with both options
enabled.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `with_mean` | boolean | true | true, false | Subtract the saved training mean. |
| `with_std` | boolean | true | true, false | Divide by the saved training scale. |

**Minimal YAML:**

```yaml
- name: standardscaler
  transformer: StandardScaler
  params: {columns: [income], with_mean: true, with_std: true}
```

**Common mistake:** Scaling does not impute NaNs; fill them first if the estimator requires finite
inputs.

**Details:** Saves population mean and scale; apply computes `(x - mean) / scale`
with disabled operations omitted. For `[1, 3]`, output is `[-1, 1]`. The two
flags false preserve numeric values (conversion may still affect dtype).
Native Spark supports its explicitly validated numeric contract, with finite
statistics and portable-state checks; this does not extend Spark support to the
other scalers or arbitrary downstream steps.

## MinMaxScaler

**Purpose and effect:** Put numeric inputs on a chosen scale; training [10, 20] becomes [0, 1].

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `feature_range` | pair[number] | [0, 1] | Lower strictly less than upper. | Map training extrema to this range. |

**Minimal YAML:**

```yaml
- name: minmaxscaler
  transformer: MinMaxScaler
  params: {columns: [income], feature_range: [0, 1]}
```

**Common mistake:** Future values can lie outside the requested range; this wrapper does not clip.

**Details:** Saves fitted minima and scale. Training values
`[10, 20]` become `[0, 1]`; a future 30 becomes 2 because there is no public
clipping option. Constant columns use a nonzero denominator. No native Spark
fit/apply declaration for this node.

## MaxAbsScaler

**Purpose and effect:** Scale by maximum absolute training value; [-4, 2] becomes [-1, 0.5].

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |

**Minimal YAML:**

```yaml
- name: maxabsscaler
  transformer: MaxAbsScaler
  params: {columns: [income]}
```

**Common mistake:** This is not centering: use StandardScaler if the mean should become zero.

**Details:** Saves maximum absolute values; `[-4, 2]` becomes
`[-1, 0.5]`. There is no centering, clipping or feature-range parameter. Future
values may exceed absolute 1. Zero columns remain zero.

## RobustScaler

**Purpose and effect:** Center on the median and scale by a quantile span to reduce the influence of
extreme observations.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `quantile_range` | pair[number] | [25.0, 75.0] | Ordered percentiles in 0–100. | Choose the fitted spread. |
| `with_centering` | boolean | true | true, false | Subtract the fitted median. |
| `with_scaling` | boolean | true | true, false | Divide by the fitted quantile span. |

**Minimal YAML:**

```yaml
- name: robustscaler
  transformer: RobustScaler
  params: {columns: [income], quantile_range: [25, 75], with_centering: true, with_scaling: true}
```

**Common mistake:** Percentiles use 0–100, and unit_variance is not an exposed option.

**Details:** Saves medians and quantile spans. The fitted median becomes 0 when
centered; an observation one fitted interquantile span above it becomes 1.
There is no exposed `unit_variance` option.

## SimpleTransformation

**Purpose and effect:** square replaces 3 with 9; log replaces 3 with `log(4)`.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `transformations` | list[mapping] | [] | Ordered rules. | Apply the fixed calculations below. |
| `transformations[].column` | string | required | Existing numeric or coercible column. | Choose the column replaced by the rule. |
| `transformations[].method` | string | required | log, sqrt/square_root, cube_root, reciprocal, square, exp/exponential. | Choose fixed math; log means log1p. |
| `transformations[].clip_threshold` | number | 700 | Used by exp. | Clip before exponentiation. |

**Minimal YAML:**

```yaml
- name: simpletransformation
  transformer: SimpleTransformation
  params:
    transformations: [{column: income, method: log}]
```

**Common mistake:** Use transformations; the old func/columns metadata does not configure this node.

**Details:** Rules run in order and replace their columns. Numeric text
is coerced; invalid text becomes missing. Negative log/sqrt inputs and zero
reciprocals become missing; cube root supports negatives; exp clips above its
threshold. Unknown methods/missing source names are skipped. Saves rules only;
context row. Dependencies: base. The registry's old `func`/`columns` metadata
does not configure this node: use the actual `transformations` list above.

## GeneralTransformation

**Purpose and effect:** ordered fixed math plus learned per-column power normalization, for example
square then Yeo–Johnson.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `transformations` | list[mapping] | [] | Ordered SimpleTransformation rules, box-cox or yeo-johnson. | Apply fixed math and optionally learn power transforms. |
| `transformations[].column` | string | required | Existing input column. | Choose the replaced column. |
| `transformations[].method` | string | required | All SimpleTransformation methods plus box-cox and yeo-johnson. | Choose fixed or fitted math. |
| `transformations[].clip_threshold` | number | 700 | Used by exp. | Clip before exponentiation. |
| `transformations[].standardize` | boolean | true | Used by power rules. | Standardize the power-transformed values. |

**Minimal YAML:**

```yaml
- name: generaltransformation
  transformer: GeneralTransformation
  params:
    transformations:
      - {column: income, method: yeo-johnson, standardize: true}
```

**Common mistake:** Box–Cox requires positive values; inspect warnings when a power rule leaves a
column unchanged.

**Details:** Each rule
fits on the result of prior rules. Saves lambdas and optional standardization
statistics, never refits in apply. Box–Cox requires strictly positive usable
values. A failed power transform is logged and leaves that column unchanged;
this fallback can depend on the batch, so context is global when power rules
are present, row for fixed rules only. Use PowerTransformer when you want a
dedicated power step. Dependencies: base/sklearn.

## PowerTransformer

**Purpose and effect:** learn a monotonic transform making skewed numeric features more nearly
Gaussian.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `method` | string | yeo-johnson | yeo-johnson, box-cox | Choose the learned power family. |
| `standardize` | boolean | true | true, false | Center and scale transformed values. |

**Minimal YAML:**

```yaml
- name: powertransformer
  transformer: PowerTransformer
  params: {columns: [income], method: yeo-johnson, standardize: true}
```

**Common mistake:** Do not choose Box–Cox for zero or negative inputs.

**Details:** Saves fitted lambdas and scale
statistics, replaces selected columns with floats. Yeo–Johnson accepts negative
values; Box–Cox requires positive values. Missing values remain missing and
degenerate/all-null fitting needs a usable training population; inspect warnings
and the resulting state. Context is determined by the fitted operation and its
failure behavior; do not assume worker admission from this mathematical
description. Dependencies: base/sklearn.
