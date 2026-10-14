# Custom factory contracts and ready functions

[Back to the preprocessing catalog](preprocessing_recipes.md)

These cards cover the Core function wrappers behind custom factories and the
ready factories shipped in a generated project. For writing your own factory,
see [custom steps](databricks_custom_steps.md) and
[custom pre-split filters](databricks_custom_pre_split.md).

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

`custom: preprocessing.log_feature` calls a **top-level factory** in the captured
`src/features/preprocessing.py` with YAML `params` as keyword arguments. The
factory returns one valid step dictionary. It is not passed a DataFrame.
The returned step later calls the calculation/learning functions with real
rows. A raw `def transform(df)` returning a frame, a sklearn transformer, a list
of steps, a lambda or a closure is not a valid factory. Use one YAML entry per
factory and wrap calculations with the existing helpers below.

Callbacks receive a private pandas frame, also when the surrounding model uses
Polars; their output is restored to the original engine. Column callbacks must
return one value per input row and declared output, preserving index/order.
An array follows row position; a pandas Series/frame must retain the index.
Empty and null handling belong to the callback, so define and test them.
Learning functions return a dict with string keys and JSON-compatible finite
values; NaN/Infinity and nonstring mapping keys are rejected. Apply never sees y.
Keep I/O out of imports and factories; package dependencies explicitly.

## ColumnFunction

**Purpose and effect:** a fixed calculation adds or replaces columns. The registry ID is code-only;
select the factory in Bundle YAML:

**File:** `config/preprocessing.yml`. Select a `custom` factory; the parameter table describes its
Python helper, not literal YAML callback values.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `name` | string | required | Nonempty step name. | Label the returned step. |
| `output` | string/list[string] | required | Nonempty unique names. | Declare output columns. |
| `replace` | boolean | false | true, false | Allow replacement of existing output names, excluding hidden target. |
| `params` | JSON mapping/null | {} | String keys and finite JSON values. | Pass fixed callback arguments. |
| `inference_context` | string/null | null (unknown) | row, group, window or global declaration accepted by the helper. | Declare actual callback context; declaration is still validated. |
| `function` | Python callable | required | Top-level captured function; not a YAML value. | Compute output columns. |

**Minimal YAML:**

```yaml
- custom: preprocessing.log_feature
  params: {column: income}
```

**Common mistake:** Put Python callbacks in a top-level factory and select custom in YAML; raw
functions are not YAML transformer parameters.

**Details:** Python helper: `column_step(name, function, *, output, replace=False, params=None,
inference_context=None)`. Required `name` string, top-level `function`, `output`
string or nonempty unique string list; `replace` bool default false;
`params` JSON mapping default `{}`; `inference_context` optional row/group/window/
global declaration (default unknown). Callback is `function(df)` when params
empty, or `function(df, params)` otherwise. Saves function reference/config,
no learned state. Output names must be new unless replace is true; overwriting
the hidden target is prohibited. Context depends on actual code, not the helper
name. Dependencies: base plus callback imports. No blanket worker/REST admission.

## FittedFunction

**Purpose and effect:** learn frequencies from training rows, then replace categories using the
saved lookup.

**File:** `config/preprocessing.yml`. Select a `custom` factory; the parameter table describes its
Python helper, not literal YAML callback values.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `name` | string | required | Nonempty step name. | Label the returned step. |
| `output` | string/list[string] | required | Nonempty unique names. | Declare output columns. |
| `replace` | boolean | false | true, false | Allow replacement of existing output names, excluding hidden target. |
| `params` | JSON mapping/null | {} | String keys and finite JSON values. | Pass fixed callback arguments. |
| `inference_context` | string/null | null (unknown) | row, group, window or global declaration accepted by the helper. | Declare actual callback context; declaration is still validated. |
| `learn` | Python callable | required | Top-level captured function. | Learn a JSON dict from training df and y. |
| `apply` | Python callable | required | Top-level captured function. | Compute outputs from df and saved state, without y. |

**Minimal YAML:**

```yaml
- custom: preprocessing.frequency_encoding
  params: {columns: [segment]}
```

**Common mistake:** A saved state does not prove row independence; declare the context your callback
actually requires.

**Details:** Python helper: `fitted_step(name, learn, apply, *, output, replace=False,
params=None, inference_context=None)`. Same name/output/replace/params/context
fields as column_step; required top-level `learn` and `apply` functions replace
the single callback. Calls `learn(df, y)` → JSON dict, and `apply(df, state)` →
outputs; nonempty params adds the params argument last to both. The target is
available only to learn, separate from the feature frame. Saves state once per
fit/fold; apply cannot learn from prediction rows by using the helper alone,
but user code must still honor that intent. Context depends on the callback;
saved state is not proof of row independence. Dependencies: callback-specific.

## RowFilterFunction

**Purpose and effect:** retain rows whose age is inside a declared range.

**File:** `config/pre_split.yml`, through a captured project factory. The parameter table describes
the Python helper used by the factory.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `name` | string | required | Nonempty step name. | Label the returned step. |
| `function` | Python callable | required | Top-level captured fixed filter. | Return an aligned nonmissing Boolean Series. |
| `columns` | list[string] | required | Nonempty unique required source names. | Declare columns read by the filter. |
| `params` | JSON mapping/null | {} | String keys and finite JSON values. | Supply fixed callback arguments. |
| `inference_context` | string/null | null (unknown) | Fixed row-local context required for pre_split admission. | Declare actual callback behavior. |

**Minimal YAML:**

```yaml
- custom: pre_split.value_range
  params: {column: age, minimum: 0, maximum: 120}
```

**Common mistake:** An arbitrary custom transformer is not admitted before splitting; use the
fixed-filter helper and preserve source values.

**Details:** Python helper: `filter_step(name, function, *, columns, params=None,
inference_context=None)`. Required name, top-level function and nonempty unique
string `columns` list; params/context follow the helpers above. Callback returns
a pandas Boolean Series aligned to the input index, one nonmissing value per row;
explicitly fill ambiguous mask values. Saves a function reference/rule and a
declaration with `effect: filter`, required_columns and learns_from_data false.
Bundle admission checks that captured registration and declaration, JSON-safe
params and unchanged source values/dtypes. It does not admit arbitrary custom
transformers before splitting. Filters must be fixed and row-local for pre-split;
do not calculate means, deduplication or request-sized ranks in the callback.
Rows/y are filtered consistently, never edited. Dependencies: callback-specific;
context unknown unless explicitly declared, and still subject to validation.

## preprocessing.log_feature

**Purpose and effect:** Add a nonnegative log feature: [-1, 0, 3, null] becomes
[0, 0, log(4), null].

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `column` | string | required | Existing numeric column. | Add log_<column> without replacing input. |

**Minimal YAML:**

```yaml
- custom: preprocessing.log_feature
  params: {column: income}
```

**Common mistake:** This clips negatives to zero before log1p; it is not the same rule as
SimpleTransformation log.

**Details:** Adds `log_<column>`
without replacing the source, using `log1p(max(value, 0))`. Thus `[-1, 0, 3, null]`
produces `[0, 0, 1.386294..., null]`; positive infinity remains infinity and
nonnumeric text needs casting first. Output is floating. Saves fixed code/params,
no learned state; mathematically row-local, although the shipped factory does
not explicitly declare diagnostic context. Base NumPy/pandas dependencies.

## preprocessing.frequency_encoding

**Purpose and effect:** Use training category frequency as a numeric feature; [A, A, B, null] learns
A=0.5 and B=0.25.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | required | Nonempty unique names; use string categories for JSON lookup keys. | Replace categories with saved training shares. |

**Minimal YAML:**

```yaml
- custom: preprocessing.frequency_encoding
  params: {columns: [segment]}
```

**Common mistake:** Null contributes to the denominator but not a category count; unseen/null
scoring values become 0.

**Details:** Fits
`value_counts / number_of_training_rows`; null is excluded from counts but
included in the denominator. Training `[A, A, B, null]` stores A=0.5, B=0.25;
prediction `[A, NEW, null]` becomes `[0.5, 0, 0]`. Replaces selected columns with
floats; empty training raises, empty apply retains columns. Saves one lookup per
column, never recomputes scoring frequencies. Mathematically row-local apply;
the shipped factory has no explicit context declaration. Base dependencies.

## preprocessing.rare_categories

**Purpose and effect:** Combine rare and unseen categories while preserving missing values.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `column` | string | required | Existing category column; string categories recommended. | Choose values to group. |
| `min_share` | number | 0.05 | Strictly between 0 and 1. | Keep categories at or above this training share. |
| `other` | JSON scalar | Other | Compatible replacement value. | Replace rare and unseen nonnull categories. |

**Minimal YAML:**

```yaml
- custom: preprocessing.rare_categories
  params: {column: segment, min_share: 0.05, other: Other}
```

**Common mistake:** An original category already named Other is not escaped and becomes
indistinguishable from grouped values.

**Details:** Fits category shares using all training rows as denominator, saving category
strings with share **at least** the threshold. Apply keeps known common values
and nulls; rare/unseen values become other. A 2% category and an unseen category
both become Other at a 5% threshold. It replaces the original column (object
on pandas). An original value already equal to Other is not reserved or escaped,
so it becomes indistinguishable from grouped values. Use string categories to
avoid stringification ambiguities. Saves common values; no request refit.
Empty training produces an empty keep set, not a useful learned vocabulary.
Mathematically row-local, undeclared diagnostic context in the shipped factory.

## pre_split.minimum_completeness

**Purpose and effect:** Keep rows with enough observed inputs; [1, null] passes min_present 1 but
fails 2.

**File:** `config/pre_split.yml`, through a captured project factory.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | required | Nonempty unique existing names. | Choose cells counted for completeness. |
| `min_present` | integer | 1 | 1 through number of selected columns; Boolean rejected. | Require this many nonmissing cells. |

**Minimal YAML:**

```yaml
- custom: pre_split.minimum_completeness
  params: {columns: [age, income], min_present: 1}
```

**Common mistake:** Empty text and infinity count as present; completeness is not a value-validity
check.

**Details:** Keeps rows with that many nonmissing cells. Null/NaN are missing; empty text and
infinity count as present. `[1, null]` stays at 1 and is removed at 2. No learned
state; fixed row-local filter, preserves survivor values/order and y alignment.
Empty apply remains empty. Base dependencies.

## pre_split.value_range

**Purpose and effect:** Keep values within fixed inclusive limits while preserving missing values.

**File:** `config/pre_split.yml`, through a captured project factory.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `column` | string | required | Existing comparable column. | Choose eligibility value. |
| `minimum` | comparable scalar | required | Compatible with column and no greater than maximum. | Set inclusive lower bound. |
| `maximum` | comparable scalar | required | Compatible with column and no less than minimum. | Set inclusive upper bound. |

**Minimal YAML:**

```yaml
- custom: pre_split.value_range
  params: {column: age, minimum: 0, maximum: 120}
```

**Common mistake:** Missing values survive this rule; add a completeness filter when missing input
is ineligible.

**Details:** Keeps values in the closed interval **or missing values**. `[35, -1, 130, null]`
keeps 35 and null. It does not validate away missing inputs: combine completeness
when that is required. Reversed/incompatible bounds are not a useful rule;
validate them before training. Saves rule only; row-local filter, empty apply
stays empty. Base dependencies.

## pre_split.allowed_values

**Purpose and effect:** Keep listed categories, for example NL and DE, and remove other rows.

**File:** `config/pre_split.yml`, through a captured project factory.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `column` | string | required | Existing column. | Choose eligibility category. |
| `values` | sequence | required | Configured allowed values; empty sequence admits no rows. | Keep matching values without case normalization. |

**Minimal YAML:**

```yaml
- custom: pre_split.allowed_values
  params: {column: country, values: [NL, DE]}
```

**Common mistake:** NL and nl are different unless normalized first; an ordinary nonnull list
removes missing values.

**Details:** Empty values allow no rows. Uses membership in the configured values, without
case normalization: NL is kept, FR removed. Null is removed for ordinary
nonnull allowed lists; explicitly placing a null-like value in the list follows
pandas membership behavior, so omit null to require a known category. Saves rule
only; row-local filter with original survivor values/order. Base dependencies.

## Asset-backed factories

`preprocessing.city_region` and `pre_split.allowed_countries` are **commented
examples**, not active shipped factories. To use one:

1. Create its UTF-8 JSON asset under `src/features/assets/`, for example
   `countries.json` containing `["NL", "DE", "BE"]`.
2. Add its relative filename to `src/features/assets.json`, for example
   `{"files": ["assets/countries.json"]}`, retaining the existing manifest's
   required fields and other entries.
3. Uncomment the matching import, callback and factory in the phase Python file.
4. Add `custom: pre_split.allowed_countries` (no params) to the selected pre-split recipe,
   or `custom: preprocessing.city_region` to the preprocessing recipe for the city→region lookup.
5. Run smoke/preview and retrain. The asset is captured with the model;
   changing the live file does not change an existing model version.

The country callback filters on `country`; the region callback reads `city` and
adds `region`, with unmapped cities missing. Both use `read_project_asset` from
the captured package. A raw filesystem path, uncaptured network service or
feature-group function is not a substitute for this saved asset contract.
