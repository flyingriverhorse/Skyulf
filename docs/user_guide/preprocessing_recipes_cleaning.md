# Missing values, cleaning and types

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps fill missing values, drop incomplete rows or columns, clean text,
replace values and cast types. Seven of them are also admitted in
`config/pre_split.yml`; each card names the file it belongs in. The last section
shows a complete fixed-normalization recipe for `config/pre_split.yml`.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

## SimpleImputer

**Purpose and effect:** `[2, null, 4]` with mean becomes `[2, 3, 4]`.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing columns; mean/median require numeric; [] selects none. | Select features; omitted selection uses strategy-appropriate detection. |
| `strategy` | string | mean | mean, median, most_frequent (mode), constant | Choose learned statistic or fixed fill. |
| `fill_value` | scalar/null | null | Used by constant; specify text/Boolean fills explicitly. | Numeric default is 0; object defaults differ by engine. |

**Minimal YAML:**

```yaml
- name: simpleimputer
  transformer: SimpleImputer
  params: {columns: [income], strategy: mean}
```

**Common mistake:** Do not expect mean imputation to fill an all-null training column; choose an
explicit constant.

**Details:** Mean/median need numeric input; other modes
can select text. State saves selected columns and one fill value per column.
Apply replaces null/NaN in place; mean/median can promote integer columns to
float. A learned all-null fill remains unavailable; use explicit constant filling
for such columns. Most-frequent ties use the smallest comparable value. Context:
row. Dependencies: base. Native Spark accepts numeric mean and compatible
constant fills (text/Boolean need explicit matching values), not all local strategies.

## GroupImputer

**Purpose and effect:** missing income in group A receives A's training mean; an unseen group
receives the global training mean.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `group_by` | string | required | One existing column. | Select the group lookup key. |
| `columns` | list[string]/null | null | Empty/omitted selects eligible columns automatically. | Choose columns to fill; group key and target are excluded. |
| `strategy` | string | mean | mean, median, most_frequent, mode | Learn per-group fills and global fallbacks. |

**Minimal YAML:**

```yaml
- name: groupimputer
  transformer: GroupImputer
  params: {columns: [income], group_by: segment, strategy: mean}
```

**Common mistake:** Do not recompute group means on scoring rows; the saved lookup and global
fallback are reused.

**Details:** Automatic selection excludes
group key and target and selects numeric columns for mean/median, all eligible
columns for mode. State stores group/value pairs and global fallbacks. Missing
group keys, unseen groups and groups without a usable statistic fall back to the
saved global value; an entirely unobserved column can remain missing. Replaces
selected missing cells, preserves observed values, and uses floating values for
numeric statistics. Context: row after fit, not request-time group aggregation.
Dependencies: base. No native Spark fit/apply claim.

## KNNImputer

**Purpose and effect:** fill a missing numeric feature from nearby training observations measured on
their shared observed features.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `n_neighbors` | integer | 5 | Positive. | Choose the number of training neighbors. |
| `weights` | string | uniform | uniform, distance; callable is Python-only. | Weight neighbor contributions. |

**Minimal YAML:**

```yaml
- name: knnimputer
  transformer: KNNImputer
  params: {columns: [income, age], n_neighbors: 5, weights: uniform}
```

**Common mistake:** Do not send text categories into numeric neighbor distances; encode or exclude
them first.

**Details:** State stores the fitted sklearn imputer and training
reference matrix. All-missing training columns are removed from its imputation
matrix and left untouched in the output, with a warning. The full frame retains
its columns. Outputs numeric fills, including float promotion for nullable
integers; null/NaN are missing. Empty apply preserves fitted output width.
Context: row against saved training neighbors. Dependencies: base/sklearn.

## IterativeImputer

**Purpose and effect:** model each incomplete numeric column from the other selected columns, then
replay the fitted chained models.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `max_iter` | integer | 10 | Nonnegative. | Limit chained imputation rounds. |
| `estimator` | string | BayesianRidge | BayesianRidge, DecisionTree/decision_tree, ExtraTrees/extra_trees, KNeighbors/knn. | Choose the fitted predictor; unrecognized names fall back to BayesianRidge. |
| `random_state` | integer/null | 0 | Integer seed or null. | Control estimator randomness. |

**Minimal YAML:**

```yaml
- name: iterativeimputer
  transformer: IterativeImputer
  params: {columns: [income, age], max_iter: 10, estimator: BayesianRidge, random_state: 0}
```

**Common mistake:** Use DecisionTree, not DecisionTreeRegressor, as the configured estimator name.

**Details:** All-missing columns are excluded from the fitted matrix and remain unchanged in
the frame. Saves the fitted imputer, selected columns and estimator choice;
replaces missing cells with numeric predictions without refitting. Empty apply
preserves width. Context: row using saved models. Dependencies: base/sklearn.

## MissingIndicator

**Purpose and effect:** income `[10, null]` adds `income_missing=[0, 1]`.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Empty/omitted selects columns missing at fit. | Choose columns receiving flags. |
| `flag_suffix` | string | _missing | Nonempty string; null/empty uses default. | Name appended 0/1 flags. |

**Minimal YAML:**

```yaml
- name: missingindicator
  transformer: MissingIndicator
  params: {columns: [income], flag_suffix: _missing}
```

**Common mistake:** A newly missing scoring column gets no flag unless it was selected during fit.

**Details:** Saves the selected columns; apply appends integer
0/1 flags for null/NaN and preserves original columns. Generated-name collisions
raise. A column first becoming missing at prediction gets a flag only if selected
at fit. Context: row. Dependencies: base.

## DropMissingColumns

**Purpose and effect:** remove unused columns or columns too incomplete in training, retaining that
same drop set at prediction.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing names; unknown names ignored. | Drop these columns explicitly. |
| `missing_threshold` | number/null | null | 0–100 percent; 0/null disables automatic selection. | Also drop columns at or above this missing percentage. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |

**Minimal YAML:**

```yaml
- name: dropmissingcolumns
  transformer: DropMissingColumns
  params: {columns: [unused_clock], missing_threshold: null}
```

**Common mistake:** The threshold is a percentage, not a 0–1 fraction; an explicit drop can remove
the target.

**Details:** Explicit drops and learned drops are unioned; unknown names are ignored.
Saves that union and removes those columns on apply, never remeasuring request
missingness. Context: row. Dependencies: base.

## DropMissingRows

**Purpose and effect:** rows with a missing required label are removed.

**File:** `config/preprocessing.yml`, or `config/pre_split.yml` under the
[narrower admission rules](preprocessing_recipes.md#pre-split-admission-and-scoring). Training-only
row changes need an explicit
[scoring decision](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `subset` | list[string]/null | null | Core empty/missing/nonmatching means all columns; pre_split requires explicit existing names. | Choose required cells. |
| `how` | string | any | any, all | Drop on any missing value or only all missing, if no threshold is set. |
| `threshold` | integer/null | null | Minimum present count; pre_split validates subset range. | Overrides percentage and how in Core. |
| `missing_threshold` | number/null | null | 0–100 percent; do not combine with threshold in pre_split. | Maximum allowed missing percentage. |

**Minimal YAML:**

```yaml
- name: dropmissingrows
  transformer: DropMissingRows
  params: {subset: [churn], how: any}
```

**Common mistake:** A filter reading the target needs an explicit scoring decision before preview;
set the skip switch for this tutorial.

**Details:** Null and floating NaN count as missing, empty strings do not. No learned
statistics; saves the policy. Removes rows and paired targets by position while
preserving survivor order. Context: row; ordinary prediction skips this training
filter. Dependencies: base.

## Deduplicate

**Purpose and effect:** two matching business records retain the first, last, or neither, as
configured.

**File:** `config/preprocessing.yml`, or `config/pre_split.yml` under the
[narrower admission rules](preprocessing_recipes.md#pre-split-admission-and-scoring). Training-only
row changes need an explicit
[scoring decision](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `subset` | list[string]/null | null | Core: all columns; pre_split: nonempty case-insensitively distinct existing names. | Define a business duplicate. |
| `keep` | string/boolean | first | first, last, none, false | Keep one representative or remove every duplicate. |

**Minimal YAML:**

```yaml
- name: deduplicate
  transformer: Deduplicate
  params: {subset: [segment, age, income], keep: first}
```

**Common mistake:** Do not use this to repair duplicate record keys or conflicting target labels;
Bundle validation rejects those inputs.

**Details:** Comparisons include missing values according to
the dataframe duplicate semantics. Saves the policy, not a global deduplication
database. Filters paired targets with the same positions and preserves survivor
order. Bundle additionally rejects conflicting labels and validates original
record-key uniqueness. Context: global within the supplied population.
Dependencies: base. Not an independent-request uniqueness guarantee.

## TextCleaning

**Purpose and effect:** `" ACME "` becomes `"acme"`.

**File:** `config/preprocessing.yml`, or `config/pre_split.yml` under the
[narrower admission rules](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Core: automatic text selection; pre_split: explicit names. | Choose text columns. |
| `operations` | list[mapping] | [] | Ordered trim, case, remove_special or regex rules; options below. | Apply fixed edits; empty list changes nothing. |

**Minimal YAML:**

```yaml
- name: textcleaning
  transformer: TextCleaning
  params:
    columns: [segment]
    operations: [{op: trim}, {op: case, mode: lower}]
```

**Common mistake:** Declare operations in the intended order; cleaning does not cast non-text
columns.

**Details:** Operation options:

| `op` | Options and defaults |
| --- | --- |
| `trim` | `mode: both` (default), `leading`, `trailing`. |
| `case` | `mode: lower` (default), `upper`, `title`, `sentence`. |
| `remove_special` | `mode: keep_alphanumeric` (default), `keep_alphanumeric_space`, `letters_only`, `digits_only`; `replacement` string default `""`. |
| `regex` | `mode: custom` (default), `collapse_whitespace`, `extract_digits`, `normalize_slash_dates`; custom requires nonempty `pattern`; `repl` string default `""`. |

Saves fixed operations and selection. Nulls remain null; explicit non-text
columns are preserved rather than converted into strings. Replacement is in
place. Python-compatible regex semantics are retained across local engines;
invalid patterns fail validation. Pre-split rejects unsupported operation keys/modes.
Context: row. Dependencies: base.

## AliasReplacement

**Purpose and effect:** known text aliases normalize to one label; `"ACME Inc."` can become `"ACME"`
with a custom map.

**File:** `config/preprocessing.yml`, or `config/pre_split.yml` under the
[narrower admission rules](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Core: automatic text selection; pre_split: explicit names. | Choose columns to normalize. |
| `alias_type` | string | boolean | boolean, country, custom, punctuation | Select normalization rules. |
| `custom_map` | mapping | {} | Normalized alias to replacement; required and nonempty for pre_split custom mode. | Supply custom aliases; other pre_split modes reject this map. |
| `mode` | string/null | null | Core compatibility alias; normalize_boolean/canonicalize_country_codes also recognized. | Used when alias_type is absent; not admitted in pre_split. |
| `custom_pairs` | mapping/null | null | Core map fallback; not admitted in pre_split. | Compatibility input for custom aliases. |

**Minimal YAML:**

```yaml
- name: aliasreplacement
  transformer: AliasReplacement
  params: {columns: [company], alias_type: custom, custom_map: {acmeinc: ACME}}
```

**Common mistake:** Boolean alias normalization produces canonical values; use Casting when you need
Boolean dtype.

**Details:** Matching lowercases text and removes punctuation/spaces; custom map keys
are normalized the same way. Unknown aliases and non-text/null values remain
unchanged. Punctuation mode only removes punctuation, preserving case/spaces.
Custom pre-split mode requires a nonempty custom_map; other pre-split modes reject one.
Core compatibility inputs `mode` (used when alias_type is unset),
`normalize_boolean`/`canonicalize_country_codes` mode names and `custom_pairs`
as a map fallback exist; Pre-split only accepts `columns`, `alias_type`, `custom_map`
and canonical modes. State saves rules; output replaces selected cells. Context:
row. Dependencies: base. Use Casting separately for actual Boolean dtype.

## ValueReplacement

**Purpose and effect:** `"unknown"` becomes null before imputation.

**File:** `config/preprocessing.yml`, or `config/pre_split.yml` under the
[narrower admission rules](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Core: all columns; pre_split: explicit names. | Choose replacement scope. |
| `mapping` | mapping/null | unset | Old→new map, or column→map. | Declare replacements in mapping form. |
| `replacements` | list[mapping]/null | unset | Each item has old and new. | Declare replacements as pairs; Core gives this mode precedence. |
| `to_replace` | scalar | unset | Nonnull old value for pre_split scalar mode. | Select scalar mode when no pair/map mode is active. |
| `value` | scalar/null | unset | JSON-safe in pre_split. | Supply the scalar replacement, including null. |

**Minimal YAML:**

```yaml
- name: valuereplacement
  transformer: ValueReplacement
  params: {columns: [segment], to_replace: unknown, value: null}
```

**Common mistake:** Use exactly one replacement mode in pre_split, and cast integers before
fractional replacements.

**Details:** Defaults are
unset; no active rule means no replacements. Core resolves pairs before mapping
before scalar mode; Bundle pre-split requires exactly one mode and explicit columns.
Pre-split also requires lossless JSON values, forbids colliding old values such as
numeric `1` and `"1"`, and requires a nonnull scalar old value in scalar mode.
Matched cells are replaced, unmatched/null cells otherwise remain. Numeric
replacement preserves integer precision; fractional/out-of-range replacement
of an integer column requires an explicit cast first. Saves rules only.
Context: row. Dependencies: base.

## InvalidValueReplacement

**Purpose and effect:** an income outside fixed bounds becomes missing.

**File:** `config/preprocessing.yml`, or `config/pre_split.yml` under the
[narrower admission rules](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Existing numeric columns; [] selects none. | Select features; omitted selection uses numeric detection. |
| `rule` | string/null | null | negative, negative_to_nan, zero, custom_range; Core also zero_to_nan, percentage_bounds, age_bounds. | Select a fixed invalid-value rule; pre_split allows negative/zero/custom_range only. |
| `replace_inf` | boolean | false | true, false | Replace positive infinity. |
| `replace_neg_inf` | boolean | false | true, false | Replace negative infinity. |
| `min_value, max_value` | number/null | null | Inclusive bounds; pre_split custom_range needs at least one finite bound. | Mark values outside the range invalid. |
| `replacement` | scalar/null | missing | null or a compatible finite scalar in pre_split. | Set invalid cells to this value. |
| `value` | scalar/null | null | Nonnull value overrides replacement. | Compatibility replacement alias. |
| `mode` | string/null | null | Core alias for rule; not admitted in pre_split. | Compatibility rule input. |

**Minimal YAML:**

```yaml
- name: invalidvaluereplacement
  transformer: InvalidValueReplacement
  params: {columns: [income], rule: custom_range, min_value: 0, max_value: 1000000, replacement: null}
```

**Common mistake:** Use rule: custom_range, not outside_range; pre_split requires explicit columns
and JSON-safe replacements.

**Details:** Supported named rules
are `negative`/`negative_to_nan`, `zero`, and `custom_range` (values strictly
outside the specified inclusive bounds). Boundary comparisons are fixed,
not fitted quantiles. Core also normalizes zero_to_nan, percentage_bounds
(0–100), age_bounds (0–120), with explicit bounds overriding presets. Pre-split allows
only negative, zero, custom_range or null with an infinity flag; custom_range
needs a finite lower or upper bound. A nonnull value overrides replacement.
Replaces invalid selected numeric cells in place, preserves existing missing
values and exact integer values where representable. Cast before fractional
replacement in integer columns. Pre-split accepts only its fixed allowlisted fields,
requires explicit columns and JSON-safe replacement (`null`, not NaN/Infinity).
State stores rules only. Context: row. Dependencies: base.

## Casting

**Purpose and effect:** text `"12"` becomes numeric 12; invalid numeric text becomes null with
coercion enabled.

**File:** `config/preprocessing.yml`, or `config/pre_split.yml` under the
[narrower admission rules](preprocessing_recipes.md#pre-split-admission-and-scoring).

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing names paired with target_type. | Apply a common type to selected columns. |
| `target_type` | string | required with columns | Portable types listed below. | Override the same columns in column_types. |
| `column_types` | mapping | {} | Column→type. | Set individual output types. |
| `coerce_on_error` | boolean | true | true, false | Convert invalid input to missing, or request failure. |

**Minimal YAML:**

```yaml
- name: casting
  transformer: Casting
  params: {column_types: {age: float64, active: boolean}, coerce_on_error: true}
```

**Common mistake:** Category casts learn a vocabulary and are not admitted in pre_split.

**Details:** Supported cross-engine names:
`float32`, `float64` (`float`, `double`, `numeric`); `int32`, `int64` (`int`,
`integer`); `string` (`str`, `text`); `boolean` (`bool`); `datetime64[ns]`
(`datetime`, `date`, `datetime64`); `category` (`categorical`). Extra native dtype
spellings are engine-specific and not portable Bundle pre-split contracts. Boolean text
accepts true/yes/1/on/y/t and false/no/0/off/n/f, case-insensitively. Numeric and
nullable dtype conversion can change precision; use explicit float widths.
Category casting learns and saves the vocabulary, with unseen categories
missing on replay; therefore Pre-split rejects it. Other casts save only the fixed type
map. Columns are replaced in place; missing values use the destination's nullable
representation. Context: row. Dependencies: base.

## Fixed normalization followed by eligibility

This **pre_split.yml recipe fragment** demonstrates explicit text and type
normalization before target completeness and business deduplication. Replace
names and rules for your data; do not include record keys in the write columns.

```yaml
clean_known_labels:
  - name: textcleaning
    transformer: TextCleaning
    params: {columns: [segment], operations: [{op: trim}, {op: case, mode: lower}]}
  - name: casting
    transformer: Casting
    params: {columns: [age], target_type: float64, coerce_on_error: true}
  - name: dropmissingrows
    transformer: DropMissingRows
    params: {subset: [churn]}
  - name: deduplicate
    transformer: Deduplicate
    params: {subset: [segment, age, income], keep: first}
```

Deduplicate is not a target-conflict resolution rule: conflicting labels in
one subset group fail. If you need a business tie-breaker or cross-table join,
prepare that upstream with explicit lineage. Unseen text does not automatically
map to an alias; null produced by Casting still needs a deliberate missing-value
policy. Select the recipe with `pre_split_recipe: clean_known_labels`.
