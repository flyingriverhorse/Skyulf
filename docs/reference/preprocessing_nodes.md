# Preprocessing Nodes

This page documents the preprocessing node types supported by `FeatureEngineer`.

## Step schema

Every preprocessing step in the pipeline config uses:

```python
{
  "name": "...",
  "transformer": "TransformerType",
  "params": { ... }
}
```

Where `params` is passed into the node’s Calculator `fit()`.

## Splitters

Example step:

```python
{"name": "split", "transformer": "TrainTestSplitter", "params": {"test_size": 0.2, "random_state": 42, "target_column": "target"}}
```

### TrainTestSplitter

Splits a DataFrame (or `(X, y)` tuple) into `SplitDataset(train, test, validation)`.

Config (`params`):

- `test_size`: float (default 0.2)
- `validation_size`: float (default 0.0)
- `random_state`: int (default 42)
- `shuffle`: bool (default True)
- `stratify`: bool (default False)
- `target_column`: str (required only when splitting a DataFrame and using stratify)

Learned params: none (passes through config).

### feature_target_split

Splits a DataFrame into `(X, y)` (or applies the split to each `SplitDataset` split).

Config:

- `target_column`: str (required)

Learned params: none.

## Cleaning

Example step:

```python
{
  "name": "clean_text",
  "transformer": "TextCleaning",
  "params": {
    "columns": ["free_text"],
    "operations": [
      {"op": "trim", "mode": "both"},
      {"op": "case", "mode": "lower"},
      {"op": "regex", "mode": "collapse_whitespace"}
    ]
  }
}
```

### TextCleaning

Applies a list of string operations.

Config:

- `columns`: list[str] (optional; auto-detects text-like columns)
- `operations`: list[dict]
  - `{ "op": "trim", "mode": "both"|"leading"|"trailing" }`
  - `{ "op": "case", "mode": "lower"|"upper"|"title"|"sentence" }`
  - `{ "op": "remove_special", "mode": "keep_alphanumeric"|"keep_alphanumeric_space"|"letters_only"|"digits_only", "replacement": "" }`
  - `{ "op": "regex", "mode": "collapse_whitespace"|"extract_digits"|"custom", "pattern": "...", "repl": "..." }`

Learned params:

- `columns`
- `operations`

### ValueReplacement

Replaces values in selected columns.

Config:

- `columns`: list[str]
- Either:
  - `mapping`: dict (global mapping) **or** dict[col -> mapping]
  - `to_replace` + `value`
  - `replacements`: list of `{old, new}` (converted into a mapping)

Learned params:

- `columns`, `mapping`, `to_replace`, `value`

### AliasReplacement

Normalizes common textual aliases (boolean/country/custom).

Config:

- `columns`: list[str] (optional; auto-detects text-like columns)
- `alias_type`: `boolean` | `country` | `custom` (also supports legacy `mode`)
- `custom_map`: dict[str, str] (also supports legacy `custom_pairs`)

Learned params:

- `columns`, `alias_type`, `custom_map`

### InvalidValueReplacement

Replaces invalid numeric values.

Config:

- `columns`: list[str]
- `rule`: `negative` | `zero` | `negative_to_nan` | `custom_range` (also supports legacy `mode`)
- `replacement`: any (default NaN)
- `min_value` / `max_value`: used by `custom_range`

Learned params:

- `columns`, `rule`, `replacement`, `min_value`, `max_value`

## Drop & Missing

### Deduplicate

Config:

- `subset`: list[str] | None
- `keep`: `first` | `last` | `none` (mapped to `False`)

Learned params:

- `subset`, `keep`

### DropMissingColumns

Config:

- `missing_threshold`: float (percentage; if > 0, drops columns with missing% >= threshold)
- `columns`: list[str] (explicit columns to drop)

Learned params:

- `columns_to_drop`, `threshold`

### DropMissingRows

Config:

- `subset`: list[str] | None
- `how`: `any` | `all` (ignored if `threshold` provided)
- `threshold`: int | None (min non-null values)

Learned params:

- `subset`, `how`, `threshold`

### MissingIndicator

Adds `{col}_missing` indicator columns.

Config:

- `columns`: list[str] (optional; defaults to all columns with any missing values)

Learned params:

- `columns`

## Imputation

Example step:

```python
{"name": "impute", "transformer": "SimpleImputer", "params": {"strategy": "median", "columns": ["age"]}}
```

### SimpleImputer

Config:

- `strategy`: `mean` | `median` | `most_frequent` | `constant` (also accepts `mode`)
- `columns`: list[str] (optional; numeric auto-detection for mean/median)
- `fill_value`: any (used for `constant`)

Learned params:

- `columns`
- `strategy`
- `fill_values`: dict[col -> value]
- `missing_counts`: dict[col -> count]
- `total_missing`: int

### KNNImputer

Config:

- `columns`: list[str] (numeric)
- `n_neighbors`: int (default 5)
- `weights`: `uniform` | `distance`

Learned params:

- `columns`
- `imputer_object` (sklearn object; pickled in pipeline)
- `n_neighbors`, `weights`

### IterativeImputer

Config:

- `columns`: list[str] (numeric)
- `max_iter`: int (default 10)
- `estimator`: `BayesianRidge` | `DecisionTree` | `ExtraTrees` | `KNeighbors`

Learned params:

- `columns`
- `imputer_object` (sklearn object; pickled in pipeline)
- `estimator`

## Encoding

Example step:

```python
{"name": "encode", "transformer": "OneHotEncoder", "params": {"columns": ["city"], "drop_original": True, "handle_unknown": "ignore"}}
```

### OneHotEncoder

Config:

- `columns`: list[str] (optional; auto-detects categorical columns)
- `drop_first`: bool (default False)
- `max_categories`: int (default 20)
- `handle_unknown`: `ignore` | `error` (default ignore)
- `drop_original`: bool (default True)
- `include_missing`: bool (default False)

Learned params:

- `columns`
- `encoder_object` (sklearn OneHotEncoder)
- `feature_names`: list[str]
- `drop_original`, `include_missing`

### DummyEncoder

Config:

- `columns`: list[str]
- `drop_first`: bool

Learned params:

- `columns`
- `categories`: dict[col -> list[str]]
- `drop_first`

### OrdinalEncoder

Config:

- `columns`: list[str]
- `handle_unknown`: str (default `use_encoded_value`)
- `unknown_value`: int/float (default -1)

Learned params:

- `columns`
- `encoder_object` (sklearn OrdinalEncoder)
- `categories_count`

### LabelEncoder

Encodes either target or selected feature columns.

Config:

- `columns`: optional list[str]
  - if omitted, encodes the provided target `y`
  - if provided, encodes those feature columns (and also target if included)

Learned params:

- `encoders`: dict[col or "__target__" -> sklearn LabelEncoder]
- `classes_count`

### TargetEncoder

Requires a target series (`y`).

Config:

- `columns`: list[str]
- `smooth`: `auto` or numeric
- `target_type`: `auto` or explicit type

Learned params:

- `columns`
- `encoder_object` (sklearn TargetEncoder)

### HashEncoder

Config:

- `columns`: list[str]
- `n_features`: int (default 10)

Learned params:

- `columns`, `n_features`

## Scaling

All scaling nodes accept `columns` (optional; numeric auto-detect) and return learned numeric arrays.

### StandardScaler

Config:

- `columns`: list[str]
- `with_mean`: bool (default True)
- `with_std`: bool (default True)

Learned params:

- `columns`, `mean`, `scale`, `var`, `with_mean`, `with_std`

### MinMaxScaler

Config:

- `columns`: list[str]
- `feature_range`: tuple (default (0, 1))

Learned params:

- `columns`, `min`, `scale`, `data_min`, `data_max`, `feature_range`

### RobustScaler

Config:

- `columns`: list[str]
- `quantile_range`: tuple (default (25.0, 75.0))
- `with_centering`: bool (default True)
- `with_scaling`: bool (default True)

Learned params:

- `columns`, `center`, `scale`, `quantile_range`, `with_centering`, `with_scaling`

### MaxAbsScaler

Config:

- `columns`: list[str]

Learned params:

- `columns`, `scale`, `max_abs`

## Outliers

### IQR

Filters rows outside per-column IQR bounds.

Config:

- `columns`: list[str]
- `multiplier`: float (default 1.5)

Learned params:

- `bounds`: dict[col -> {lower, upper}]
- `warnings`

### ZScore

Config:

- `columns`: list[str]
- `threshold`: float (default 3.0)

Learned params:

- `stats`: dict[col -> {mean, std}]
- `threshold`, `warnings`

### Winsorize

Clips values into per-column percentile bounds.

Config:

- `columns`: list[str]
- `lower_percentile`: float (default 5.0)
- `upper_percentile`: float (default 95.0)

Learned params:

- `bounds`: dict[col -> {lower, upper}]

### ManualBounds

Filters rows outside user-provided bounds.

Config:

- `bounds`: dict[col -> {lower, upper}]

Learned params:

- `bounds`

### EllipticEnvelope

Learns a per-column EllipticEnvelope model and filters outliers.

Config:

- `columns`: list[str]
- `contamination`: float (default 0.01)

Learned params:

- `models`: dict[col -> sklearn model]
- `contamination`, `warnings`

## Transformations

### PowerTransformer

Config:

- `columns`: list[str]
- `method`: `yeo-johnson` | `box-cox`
- `standardize`: bool

Learned params:

- `columns`, `lambdas`, `method`, `standardize`, `scaler_params`

### SimpleTransformation

Config:

- `transformations`: list of `{column, method, clip_threshold?}`
  - methods include `log`, `square_root`, `cube_root`, `reciprocal`, `square`, `exponential`

Learned params:

- `transformations` (passes through)

### GeneralTransformation

Config:

- `transformations`: list of `{column, method, clip_threshold?}`
  - methods include power transforms (`box-cox`, `yeo-johnson`) and the simple methods

Learned params:

- `transformations` with fitted `lambdas`/`scaler_params` where applicable

## Bucketing (Binning)

### GeneralBinning

Creates binned features with configurable strategies.

Config:

- `columns`: list[str] (numeric)
- `strategy`: `equal_width` | `equal_frequency` | `kmeans` | `custom` | `kbins`
- `n_bins` and strategy-specific keys:
  - `equal_width_bins`, `equal_frequency_bins`, `duplicates`
  - `kbins_n_bins`, `kbins_strategy`
  - `custom_bins`: dict[col -> edges]
  - `custom_labels`: dict[col -> labels]
- output formatting:
  - `output_suffix`, `drop_original`, `label_format`, `missing_strategy`, `missing_label`, `include_lowest`, `precision`

Learned params:

- `bin_edges` (dict[col -> edges])
- output formatting settings

### CustomBinning

Config:

- `columns`: list[str]
- `bins`: list[float] (shared edges)
- plus output formatting keys (same as GeneralBinning)

Learned params:

- `bin_edges`

### KBinsDiscretizer

Wrapper around `GeneralBinning` with a KBins-style interface.

Config:

- `columns`: list[str]
- `n_bins`: int
- `strategy`: `uniform` | `quantile` | `kmeans`

Learned params:

- `bin_edges`

## Casting

### Casting

Config:

- Either:
  - `column_types`: dict[col -> dtype]
  - or `columns` + `target_type`
- `coerce_on_error`: bool (default True)

Learned params:

- `type_map`, `coerce_on_error`

## Feature Generation

### PolynomialFeatures

(Alias: `PolynomialFeaturesNode`)

Config:

- `columns`: list[str]
- `auto_detect`: bool
- `degree`: int
- `interaction_only`: bool
- `include_bias`: bool
- `include_input_features`: bool
- `output_prefix`: str

Learned params:

- `columns`, `degree`, `interaction_only`, `include_bias`, `include_input_features`, `output_prefix`, `feature_names`

### FeatureGeneration

(Aliases: `FeatureMath`, `FeatureGenerationNode`)

Config:

- `operations`: list[dict]
- `epsilon`: float (default 1e-9)
- `allow_overwrite`: bool

Learned params:

- `operations`, `epsilon`, `allow_overwrite`

## Feature Selection

### VarianceThreshold

Config:

- `threshold`: float (default 0.0)
- `columns`: list[str] (optional; numeric)
- `drop_columns`: bool (default True)

Learned params:

- `candidate_columns`, `selected_columns`, `variances`, `threshold`, `drop_columns`

### CorrelationThreshold

Config:

- `threshold`: float (default 0.95)
- `correlation_method`: `pearson` | `spearman` | `kendall` (default pearson)
- `columns`: list[str] (numeric)
- `drop_columns`: bool

Learned params:

- `columns_to_drop`, `threshold`, `method`, `drop_columns`

### UnivariateSelection

Config:

- `target_column`: str (if `y` not passed as tuple)
- `problem_type`: `auto` | `classification` | `regression`
- `method`: `select_k_best` | `select_percentile` | `select_fpr` | `select_fdr` | `select_fwe` | `generic_univariate_select`
- selector parameters (depending on method): `k`, `percentile`, `alpha`, `mode`, `param`
- scoring: `score_func` (e.g., `f_classif`, `mutual_info_classif`, …)
- `drop_columns`: bool

Learned params:

- `selected_columns`, `candidate_columns`, `scores`, `pvalues` (when available), plus selector config

### ModelBasedSelection

Config:

- `target_column`: str
- `problem_type`: `auto` | `classification` | `regression`
- `method`: `select_from_model` | `rfe`
- `estimator`: `auto` | `logistic_regression` | `random_forest` | `linear_regression`
- For select_from_model: `threshold`, `max_features`
- For RFE: `n_features_to_select`, `step`
- `drop_columns`: bool

Learned params:

- `selected_columns`, `candidate_columns`, and method-specific metadata

### feature_selection

A higher-level facade node that dispatches to the selection implementations.

## Resampling

### Oversampling

Config:

- `method`: `smote` | `adasyn` | `borderline_smote` | `svm_smote` | `kmeans_smote` | `smote_tomek`
- `target_column`: required if `y` is not provided as tuple
- `sampling_strategy`: `auto` or dict
- `random_state`: int
- method-specific keys: `k_neighbors`, `m_neighbors`, `kind`, `out_step`, `cluster_balance_threshold`, `density_exponent`, `n_jobs`

Learned params: none (passes through config).

### Undersampling

Config:

- `method`: `random_under_sampling` | `nearmiss` | `tomek_links` | `edited_nearest_neighbours`
- `target_column`: required if `y` not provided as tuple
- `sampling_strategy`, `random_state`, `replacement`, `version`, `n_neighbors`, `kind_sel`, `n_jobs`

Learned params: none.

## Inspection

### DatasetProfile

Captures basic dataset stats without modifying data.

Config: none.

Learned params:

- `profile`: rows/columns/dtypes/missing/numeric_stats

### DataSnapshot

Captures the first N rows without modifying data.

Config:

- `n_rows`: int (default 5)

Learned params:

- `snapshot`: list[dict]
