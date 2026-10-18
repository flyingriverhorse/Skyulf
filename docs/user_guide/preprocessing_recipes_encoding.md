# Categorical encoding

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps turn categorical columns into numbers with one-hot, ordinal, label,
hash, target and weight-of-evidence encoders. The saved mapping fixes the output
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
| `categories_order` | string/list[string]/null | null | Multiline string or list of comma-separated strings, one line per column. | Set category order; malformed line count falls back to auto. |
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
can depend on the batch. An explicit order must have the matching number of lines; a malformed shape
falls back to auto. Use comma-separated lines, not nested YAML lists.

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
| `n_features` | integer | 10 | Positive. | Set the number of possible integer buckets per source column. |

**Minimal YAML:**

```yaml
- name: hashencoder
  transformer: HashEncoder
  params: {columns: [segment], n_features: 10}
```

**Common mistake:** n_features is the bucket range; it does not create that many output columns.

**Details:** Saves selection, bucket count and normalization version, not a fitted category
vocabulary. Replaces each selected column in place with an integer bucket in `[0, n_features)`; it
does not append `n_features` columns. Collisions are expected, so different categories can share a
bucket. Unknown categories are hashable without refit. Current numeric normalization avoids changing
keys because another row contains null. The target is excluded. Context: row for current supported
scalar state.

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
the global mean; missing categories can be learned. Collision checks protect
multiclass output names. This wrapper fits saved mappings and applies them;
pipeline training separately cross-fits training encodings so a row's own target
does not enter its feature; heldout/prediction uses the full training map. The
internal fold count is bounded by sample/class counts, not a public YAML knob.
Too few examples for safe cross-fitting fail. Keep the entire step inside
training folds too. Context: row against saved maps.

## WOEEncoder

**Purpose and effect:** Replace categories with learned binary log-odds evidence; unseen keys use 0.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing categorical columns; [] selects none. | Select feature columns; omission uses categorical detection. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |
| `regularization` | number | 0.5 | Numeric smoothing strength. | Stabilize category class-count logarithms. |

**Minimal YAML:**

```yaml
- name: woeencoder
  transformer: WOEEncoder
  params: {columns: [segment], regularization: 0.5}
```

**Common mistake:** Use a binary target; missing or nonbinary labels can return no-op state.

**Details:** Requires binary target labels and finite
usable class counts; use smoothing to avoid zero-count logarithms. Learns
category log-odds/evidence maps and information value. Replaces selected columns
with floats; an unseen key uses saved default 0.0 (not an exposed fit-time
`default` knob). Missing categories have their own learned key. Current scalar
keys distinguish numbers from literal strings; older artifacts retain older
string-key semantics. Pipeline training cross-fits training encodings and saves
full-training maps for heldout/prediction. Missing/nonbinary target warns and
returns no-op state. Context: row for saved-map apply.
