# Categorical encoding

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps turn categorical columns into numbers with one-hot, ordinal, label,
hash, frequency, target and weight-of-evidence encoders. Rare-category grouping
combines infrequent text values before encoding. The saved mapping fixes the output
columns; new categories at scoring time do not extend it.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

Except LabelEncoder's target-oriented default, omitted `columns` uses categorical auto-detection and
explicit `[]` yields no feature transformation. Select columns explicitly when numeric categories or
Boolean values should be encoded. Encoder state fixes the output schema/mappings; new scoring
categories do not extend it. All cards in this section use `config/preprocessing.yml` (LabelEncoder
also has standalone target use), never pre-split. Dependencies are base/sklearn unless stated
otherwise.

Automatic categorical detection leaves homogeneous pandas Decimal object columns
as numeric inputs, matching native Polars Decimal. Explicit column selection is
unchanged; no implicit conversion of numeric values occurs.

Schema previews for explicit selections advertise Label/Hash `int64`, Ordinal
`float32`, and WOE or binary/continuous Target `float64` outputs. Automatic
category selection, multiclass Target expansion and Hash widths above the signed
64-bit range use runtime schema discovery. Label/Ordinal target-only selections
retain their existing target behavior; no runtime values or saved mappings change.

## FrequencyEncoder

**Purpose and effect:** Replace each category with its training frequency or count.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing categorical columns; [] selects none. | Select features; target is excluded. |
| `mode` | string | frequency | frequency, count | Emit float64 shares or int64 counts in place. |

```yaml
- name: category_frequency
  transformer: FrequencyEncoder
  params: {columns: [segment], mode: frequency}
```

Counts include missing values; the frequency denominator includes **all training
rows**. For `[A, A, null, B]`, saved frequencies are A=0.5, null=0.25 and B=0.25.
Unseen values become zero. Missing values share a canonical key across `None`,
`NaN`, pandas `NA` and Polars null, distinct from the literal string `"nan"`.
Apply uses saved counts without measuring the scoring batch. Empty apply retains
the output dtype; fitting selected columns with zero rows raises an error.

## RareCategoryGrouping

**Purpose and effect:** Keep frequent text categories and group rare or unseen
values under a replacement label. Output columns remain nullable strings.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing text/category columns; [] selects none. | Select features; target is excluded. |
| `min_count` | integer | 2 | At least 1; booleans rejected. | Minimum training count to retain a category. |
| `min_share` | number | 0.0 | Finite value in [0, 1]. | Minimum share of all training rows to retain a category. |
| `replacement` | string | Other | Nonempty text. | Label for rare and unseen values. |

```yaml
- name: group_rare
  transformer: RareCategoryGrouping
  params: {columns: [segment], min_count: 2, min_share: 0.01, replacement: Other}
- name: fill_missing_category
  transformer: SimpleImputer
  params: {columns: [segment], strategy: constant, fill_value: Missing}
- name: encode_groups
  transformer: OneHotEncoder
  params: {columns: [segment], max_categories: null}
```

A category must meet **both** thresholds; equality passes. Missing values are
counted as one category: frequent missing stays null, while rare missing and
unseen values become the replacement. Impute retained missing values before
learned OneHot encoding so pandas nullable strings are supported. An existing
literal `Other` label can intentionally merge with grouped values.

Non-null values must be text, including textual pandas/Polars categorical values.
For numeric category codes, add explicit Casting to string first. Saved membership
is never recomputed on inference batches. Empty apply retains the string dtype;
fitting selected columns with zero rows raises an error. In Canvas, use **Rare
Category Grouping** in the existing Encoding panel.

## OneHotEncoder

**Purpose and effect:** Turn categories into fitted numeric indicator columns, for example segment_A
and segment_B.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing categorical columns; [] selects none. | Select feature columns; omission uses categorical detection. |
| `handle_unknown` | string | ignore | ignore, error | Use all-zero indicators or fail for unseen categories. |
| `drop_first` | boolean | false | true, false | Omit the first fitted category indicator. |
| `max_categories` | integer/null | 20 | At least 2; null disables cap. | Limit fitted category width through infrequent grouping. |
| `prefix_separator` | string | _ | String. | Retained compatibility option; generated names use `_`. |
| `drop_original` | boolean | true | true, false | Remove selected source columns. |
| `include_missing` | boolean | false | true, false | Represent missing values using an escaped category sentinel. |
| `categories` | mapping/null | null | Each selected column maps to an ordered, nonempty list of unique strings. | Fix the vocabulary, including levels absent from training. Requires explicit columns, `max_categories: null` and `include_missing: false`. |
| `column_names` | string | sklearn | sklearn, slug | Keep original category text or produce lowercase ASCII category names. |

**Minimal YAML:**

```yaml
- name: onehotencoder
  transformer: OneHotEncoder
  params: {columns: [segment], handle_unknown: ignore, max_categories: null}
```

**Common mistake:** Set max_categories: null when the default cap of 20 is not intended.

The default cap remains available for local execution. Independent Spark workers and
endpoint admission require `max_categories: null` and `include_missing: false`;
the admission error lists these supported settings. Choose them explicitly before fitting
a pipeline intended for those runtimes.

**Details:** Learns sklearn categories/infrequent grouping and output names. A/B produces
`segment_A`, `segment_B`; output is dense numeric indicators. Unknown categories
with ignore are all-zero. Dropping the first category can produce no output for
a constant column. Missing-category behavior depends on fitted categories and
include_missing; enabling it uses an escaped sentinel so literal sentinel text
and null do not collide. Target selection is excluded. Retained-name collisions
raise. Context: row for the supported fitted configuration.

**Fixed vocabulary:** Keys must match `columns` exactly. Category list order fixes
output order; `drop_first: true` removes the first listed level. Inputs must be
string or categorical columns; numeric values are not converted to strings.
With `handle_unknown: ignore`, both unknown values and nulls produce zeros.
With `error`, either raises, so impute missing values first when needed.

```yaml
- name: fixed_city
  transformer: OneHotEncoder
  params:
    columns: [city]
    categories:
      city: ["Øl", "Åben", "Zürich"]
    column_names: slug
    max_categories: null
    include_missing: false
    handle_unknown: ignore
```

This emits `city_oel`, `city_aaben`, `city_zurich` even when training omits a
listed city. `slug` also works with learned categories. It folds `æ/ø/å/ß` to
`ae/oe/aa/ss`, removes accents, lowercases and replaces punctuation with `_`.
Empty or duplicate generated names fail clearly. Existing artifacts without
the new options keep their original names and encoding.

In ML Canvas, select **Use fixed categories** and enter one category per line
for each selected column. Spaces, commas and order are preserved. The form
disables the category cap and missing-category option for fixed vocabularies.

## DummyEncoder

**Purpose and effect:** Create one fitted dummy column per category with all-zero output for unseen
categories.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing categorical columns; [] selects none. | Select feature columns; omission uses categorical detection. |
| `drop_first` | boolean | false | true, false | Omit the first fitted category indicator. |

**Minimal YAML:**

```yaml
- name: dummyencoder
  transformer: DummyEncoder
  params: {columns: [segment], drop_first: false}
```

**Common mistake:** The OneHotEncoder category-cap and unknown-error options are not exposed here.

**Details:** Saves observed category
lists, replaces selected original columns with fitted dummy columns named from
column and category. A/B becomes one indicator per category; unknown values
produce zeros, and missing values do not invent a new category. Output schema
stays fixed on empty apply. Unlike OneHotEncoder, no category cap, separator,
unknown-error, include-missing or drop-original options are exposed. Context: row.

## OrdinalEncoder

**Purpose and effect:** Replace categories with ordered numeric codes; unknown categories can
receive -1.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing categorical columns; [] selects none. | Select feature columns; omission uses categorical detection. |
| `handle_unknown` | string | use_encoded_value | use_encoded_value, error | Choose how unseen categories are handled. |
| `unknown_value` | number | -1 | Must not collide with assigned codes. | Choose the unseen-category code. |
| `categories_order` | string/list[string]/null | null | Multiline string or list of comma-separated strings, one line per column. | Set category order; malformed explicit orders raise an error. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |

**Minimal YAML:**

```yaml
- name: ordinalencoder
  transformer: OrdinalEncoder
  params: {columns: [segment], handle_unknown: use_encoded_value, unknown_value: -1}
```

**Common mistake:** categories_order expects comma-separated strings, not nested YAML lists.

**Details:** Saves category lists and sklearn encoders.
Replaces A/B with float32 0/1; unseen values use the unknown policy. Newly fitted
feature keys distinguish nulls, numeric categories and literal strings; a missing
category observed at fit can get its own code. Can also encode paired y under
the target contract; avoid using it unintentionally to rewrite model labels.
Context row for current feature artifacts; legacy pandas string-based states
can depend on the batch. An explicit order must have the matching number of lines;
malformed shapes and unsupported `handle_unknown` policies raise an error at fit.
Use comma-separated lines, not nested YAML lists. Empty orders retain automatic ordering.

## LabelEncoder

**Purpose and effect:** Encode explicitly selected category columns as integer labels, or encode the
target in standalone Core use.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Explicit features; absent/empty selects y when available. | Choose features or use the target-oriented default. |
| `missing_code` | integer | -1 | Integer. | Encode missing/unseen feature values. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |

**Minimal YAML:**

```yaml
- name: labelencoder
  transformer: LabelEncoder
  params: {columns: [segment], missing_code: -1}
```

**Common mistake:** Omitting columns targets y rather than auto-detecting feature columns.

**Details:** **Absent/empty columns do not auto-detect
feature columns**: they fit y when available; with no y there may be no-op state.
Explicit feature selection learns one encoder per column; y can be encoded when
selected by its name. Saves vocabularies and target routing. Replaces categories
with integer labels; missing/unseen feature values use the saved missing code
unless represented in the learned vocabulary. Legacy/string-normalization
semantics are retained by old artifacts. Use OrdinalEncoder for multi-feature
encoding with an explicit unknown policy. Context depends on fitted state and
engine; no worker/REST promise for target-changing use.

## HashEncoder

**Purpose and effect:** Replace each category column with one integer hash bucket without learning a
vocabulary.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing categorical columns; [] selects none. | Select feature columns; omission uses categorical detection. |
| `n_features` | integer | 8 | Positive integer; booleans and floating values are rejected. | Set the number of possible integer buckets per source column. |

**Minimal YAML:**

```yaml
- name: hashencoder
  transformer: HashEncoder
  params: {columns: [segment], n_features: 8}
```

**Common mistake:** n_features is the bucket range; it does not create that many output columns.

**Details:** Saves selection, bucket count and normalization version, not a fitted category
vocabulary. Replaces each selected column in place with an integer bucket in `[0, n_features)`; it
does not append `n_features` columns. Collisions are expected, so different categories can share a
bucket. Unknown categories are hashable without refit. Current numeric normalization avoids changing
keys because another row contains null. The target is excluded. Context: row for current supported
scalar state.
Previously saved encoders keep their saved bucket count and assignments, including
the earlier default of 10; the default of 8 applies only to new fits.

## TargetEncoder

**Purpose and effect:** Replace categories with smoothed target statistics, using cross-fitted
training values in the pipeline.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing categorical columns; [] selects none. | Select feature columns; omission uses categorical detection. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |
| `target_type` | string | auto | auto, continuous, binary, multiclass | Choose target interpretation. |
| `smooth` | string/number | auto | auto or nonnegative number. | Smooth category statistics toward the global mean. |
| `cv` | integer | 5 | At least 2; booleans rejected. | Requested maximum cross-fitting folds, capped by available rows/class counts. |
| `shuffle` | boolean | true | true, false | Shuffle rows before assigning cross-fitting folds. |
| `random_state` | integer/null | 42 | 0 through 2^32-1, or null. | Reproducible shuffling; ignored when shuffle is false. |

**Minimal YAML:**

```yaml
- name: targetencoder
  transformer: TargetEncoder
  params: {columns: [segment], target_type: binary, smooth: auto}
```

**Common mistake:** Keep this step inside training folds; do not precompute encodings on the entire
dataset.

**Details:** Requires a usable target during
fit. Saves sklearn target/category statistics and global target mean. Binary or
continuous encoding replaces a column with its smoothed target mean; multiclass
drops it and emits `segment_cls0`, `segment_cls1`, etc. Unseen categories use
the global mean; missing categories can be learned. New fits use the same key for
`None`, `NaN`, pandas `NA` and Polars null, distinct from literal strings such as
`"nan"`. The saved key version makes this stable across engine changes; older
artifacts retain their original sklearn categories. Collision checks protect
multiclass output names. This wrapper fits saved mappings and applies them;
pipeline training separately cross-fits training encodings so a row's own target
does not enter its feature; heldout/prediction uses the full training map. The
fold count is bounded by `cv` and sample/class counts. Defaults remain five folds,
shuffle enabled and seed 42. With shuffle disabled the sklearn seed is unset;
this does not implement grouped or forward-time CV. A null seed permits
nondeterministic shuffled folds. Saved estimators replay their learned maps
without refitting or adopting new CV settings.
Too few examples for safe cross-fitting fail. Keep the entire step inside
training folds too. Context: row against saved maps.

## WOEEncoder

**Purpose and effect:** Replace categories with learned binary log-odds evidence; unseen keys use 0.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing categorical columns; [] selects none. | Select feature columns; omission uses categorical detection. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |
| `regularization` | number | 0.5 | Numeric smoothing strength; zero requires both classes in every category. | Stabilize category class-count logarithms. |

**Minimal YAML:**

```yaml
- name: woeencoder
  transformer: WOEEncoder
  params: {columns: [segment], regularization: 0.5}
```

Zero smoothing is supported when each learned category contains both target
classes. This must also hold in every cross-fitting training complement; a
valid full-data map alone is insufficient. Otherwise fitting raises a clear
`ValueError` recommending positive regularization before computing an undefined
ratio or logarithm. Saved valid mappings replay unchanged.

**Common mistake:** Use exactly two target classes; missing labels or any other
class count raise an error at fit.

**Details:** Requires binary target labels and finite
usable class counts; use smoothing to avoid zero-count logarithms. Learns
category log-odds/evidence maps and information value. Replaces selected columns
with floats; an unseen key uses saved default 0.0 (not an exposed fit-time
`default` knob). Missing categories have their own learned key. Current scalar
keys distinguish numbers from literal strings; older artifacts retain older
string-key semantics. Pipeline training cross-fits training encodings and saves
full-training maps for heldout/prediction. Absent target input still warns and
returns no-op state; a supplied target must have exactly two complete classes.
Context: row for saved-map apply.
