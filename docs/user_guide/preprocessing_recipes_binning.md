# Binning

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps turn numeric columns into bins with general, custom or learned
edges. The shared output parameters come first, then one card per step.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

The three binning nodes share these output parameters. They are for `config/preprocessing.yml` only,
even when supplied fixed edges, because Bundle pre-split has a narrower allowlist.
Dependencies: base/sklearn. Context: row using saved edges.

| Parameter | Type, default and meaning |
| --- | --- |
| `columns` | Numeric name list; absent auto-detects, `[]` no-op. |
| `output_suffix` | String `_binned`; output is `<source><suffix>`. |
| `drop_original` | Bool `false`; true removes only successfully binned sources. |
| `label_format` | String `ordinal`; `ordinal` (alias `bin_index`) emits zero-based bin codes; `range` emits interval labels. Valid `custom_labels` take precedence over this setting. |
| `include_lowest` | Bool `true`; includes the lowest edge in the first interval. Other intervals are right-closed. |
| `precision` | Nonnegative int `3`, range-label display precision. New fitted states retain exact labels if rounding would merge bins. |
| `missing_strategy` | `keep` default or `label`; label replaces missing/out-of-range bin results. |
| `missing_label` | Scalar/string default `Missing`, used by label mode. |

Missing and values outside the outer edges have no bin unless missing label mode
is enabled. Label mode produces strings in Polars, including ordinal code strings,
while pandas can retain numeric codes alongside a string sentinel in object
dtype. Duplicate edges and mismatched custom-label counts must be resolved;
fit failures and output-name collisions raise. All-missing columns have no learned
edges. Do not assume the number of bins always equals the requested count.

## GeneralBinning

**Purpose and effect:** Create numeric bins with learned or per-column fixed edges; five equal-width
bins put 10 and 90 in opposite end bins for 0–100 data.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `output_suffix` | string | _binned | String. | Name output as source plus suffix. |
| `drop_original` | boolean | false | true, false | Remove successfully binned source columns. |
| `label_format` | string | ordinal | ordinal, bin_index, range | Emit numeric codes or interval labels; valid custom labels take precedence where supported. |
| `include_lowest` | boolean | true | true, false | Include the lowest edge in the first interval; intervals are right-closed. |
| `precision` | integer | 3 | Nonnegative. | Choose range-label display precision, retaining exact labels if rounding merges bins. |
| `missing_strategy` | string | keep | keep, label | Keep missing/out-of-range bin results missing or label them. |
| `missing_label` | scalar | Missing | Compatible scalar/string. | Supply label-mode missing value. |
| `strategy` | string | equal_width | equal_width/uniform, equal_frequency, kmeans, custom, kbins | Choose how edges are obtained. |
| `n_bins` | integer | 5 | Positive; sklearn kbins needs at least 2. | Set base bin count. |
| `equal_width_bins, equal_frequency_bins` | integer | n_bins | Positive. | Override count for the corresponding strategy. |
| `duplicates` | string | drop | drop, raise | Handle repeated equal-frequency edges. |
| `kbins_n_bins` | integer | n_bins | At least 2. | Override sklearn bin count. |
| `kbins_strategy` | string | quantile | quantile, uniform, kmeans; equal_width/equal_frequency aliases. | Choose sklearn edge fitting. |
| `custom_bins` | mapping | {} | Column→numeric edge list. | Set per-column fixed edges for custom strategy. |
| `custom_labels` | mapping | {} | Column→labels; count must match intervals. | Override code/range labels. |
| `column_strategies` | mapping | {} | Column→strategy/count/duplicates/kbins/custom options. | Override fit settings per column. |

**Minimal YAML:**

```yaml
- name: generalbinning
  transformer: GeneralBinning
  params: {columns: [age], strategy: equal_width, n_bins: 5}
```

**Common mistake:** A requested bin count can shrink with repeated values; inspect the fitted
output.

**Details:** Noncustom strategies save learned edges. For values 0–100 and five equal-width
bins, 10 belongs to the first and 90 to the last. Unknown strategies and per-column fit failures
raise instead of saving a partial fit. All-missing columns still have no learned edges.
Saved artifacts keep their fitted edges at apply time. The runtime default is equal_width, despite older metadata saying
uniform (these two select the same width strategy).

## CustomBinning

**Purpose and effect:** Apply one fixed edge list; with [0, 18, 65, 120], 18 is in the first
interval and 19 in the second.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `output_suffix` | string | _binned | String. | Name output as source plus suffix. |
| `drop_original` | boolean | false | true, false | Remove successfully binned source columns. |
| `label_format` | string | ordinal | ordinal, bin_index, range | Emit numeric codes or interval labels; valid custom labels take precedence where supported. |
| `include_lowest` | boolean | true | true, false | Include the lowest edge in the first interval; intervals are right-closed. |
| `precision` | integer | 3 | Nonnegative. | Choose range-label display precision, retaining exact labels if rounding merges bins. |
| `missing_strategy` | string | keep | keep, label | Keep missing/out-of-range bin results missing or label them. |
| `missing_label` | scalar | Missing | Compatible scalar/string. | Supply label-mode missing value. |
| `bins` | list[number]/null | null | At least two distinct numeric edges; sorted at fit. | Use one fixed edge list for every selected column; absent/empty is no-op. |

**Minimal YAML:**

```yaml
- name: custombinning
  transformer: CustomBinning
  params: {columns: [age], bins: [0, 18, 65, 120], label_format: range}
```

**Common mistake:** Use GeneralBinning custom strategy when columns need different edges or custom
labels.

**Details:** One sorted
edge list applies to every selected column; absent/empty bins is no-op. At least
two distinct edges are needed. Shares output options above but does not expose
GeneralBinning's per-column fit strategy. For these edges, 18 belongs to the
first interval and 19 to the second. Saves fixed edges; automatic column
selection can still depend on training values. Use GeneralBinning custom
strategy for different edge/label lists per column.

## KBinsDiscretizer

**Purpose and effect:** Learn quantile, uniform or kmeans bin edges and replay them at prediction.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `output_suffix` | string | _binned | String. | Name output as source plus suffix. |
| `drop_original` | boolean | false | true, false | Remove successfully binned source columns. |
| `label_format` | string | ordinal | ordinal, bin_index, range | Emit numeric codes or interval labels; valid custom labels take precedence where supported. |
| `include_lowest` | boolean | true | true, false | Include the lowest edge in the first interval; intervals are right-closed. |
| `precision` | integer | 3 | Nonnegative. | Choose range-label display precision, retaining exact labels if rounding merges bins. |
| `missing_strategy` | string | keep | keep, label | Keep missing/out-of-range bin results missing or label them. |
| `missing_label` | scalar | Missing | Compatible scalar/string. | Supply label-mode missing value. |
| `n_bins` | integer | 5 | At least 2. | Choose sklearn bin count. |
| `strategy` | string | quantile | quantile, uniform, kmeans; equal_width/equal_frequency aliases. | Choose learned edges. |
| `kbins_n_bins` | integer | 5 | At least 2; an explicit n_bins takes precedence. | Set forwarded bin count when n_bins is absent. |
| `kbins_strategy` | string | quantile | quantile, uniform, kmeans; an explicit non-kbins strategy takes precedence. | Set forwarded strategy when strategy is absent. |
| `column_strategies` | mapping | {} | Same per-column fit options as GeneralBinning. | Override individual column strategies. |
| `equal_width_bins, equal_frequency_bins` | integer | n_bins | Positive; relevant to corresponding per-column overrides. | Set overridden strategy counts. |
| `duplicates` | string | drop | drop, raise | Handle repeated edges in equal-frequency overrides. |
| `custom_bins, custom_labels` | mapping | {} | Column→edges or labels; relevant to custom per-column overrides. | Configure forwarded custom strategies. |

**Minimal YAML:**

```yaml
- name: kbinsdiscretizer
  transformer: KBinsDiscretizer
  params: {columns: [age], n_bins: 5, strategy: quantile}
```

**Common mistake:** The sklearn encode option is not exposed; use label_format to control this
wrapper’s output.

**Details:** This wrapper routes to GeneralBinning's `kbins` strategy; it also
accepts its shared output and per-column override options. Saves learned edges,
not a new estimator fit at inference. Quantile bins seek similar training
population sizes; repeated values can reduce the number of distinct bins.
Encoding is ordinal internally; public `label_format` controls final rendering,
not sklearn's sparse one-hot `encode` setting (which is not exposed).
