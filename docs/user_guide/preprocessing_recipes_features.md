# Feature generation and selection

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps add interaction, polynomial and formula features, or keep a subset
of columns by variance, correlation, univariate scores or a model.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

## FeatureInteraction

**Purpose and effect:** income 10 and tenure 3 add `income_x_tenure=30`.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `degree` | integer | 2 | 2, 3, 4 | Generate products of exactly this degree. |
| `interaction_only` | boolean | true | true, false | Exclude repeated inputs/powers when true. |
| `include_bias` | boolean | false | true, false | Add interaction_bias=1. |

**Minimal YAML:**

```yaml
- name: featureinteraction
  transformer: FeatureInteraction
  params: {columns: [income, tenure], degree: 2}
```

**Common mistake:** This generates exactly the selected degree; use PolynomialFeatures for all terms
up to a degree.

**Details:** Generates products of **exactly** the
chosen degree; false interaction_only permits repeated columns/powers.
Saves combinations and names, adds float columns joined in sorted-name order
with `_x_`, and optionally `interaction_bias=1`. Original columns remain. Null
products stay missing; generated-name conflicts raise. Context row; base
dependencies. For terms up to a degree, use PolynomialFeatures.

## PolynomialFeatures

Aliases: `PolynomialFeaturesNode` selects this same owner.

**Purpose and effect:** x=2, y=3 adds squared and cross terms 4, 6, 9.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `auto_detect` | boolean | false | true, false | Detect numeric columns when none are selected, excluding target_column. |
| `target_column` | string/null | null | Existing label name. | Exclude the target from automatic discovery; the pipeline supplies this name. |
| `degree` | integer/pair[integer] | 2 | Nonnegative degree or ordered [min_degree, max_degree]. | Set generated term degrees. |
| `interaction_only` | boolean | false | true, false | Exclude repeated-input powers when true. |
| `include_bias` | boolean | false | true, false | Add a constant term. |
| `include_input_features` | boolean | false | true, false | Append copied degree-1 terms; original columns already survive. |
| `output_prefix` | string | poly | String. | Prefix generated names. |

**Minimal YAML:**

```yaml
- name: polynomialfeatures
  transformer: PolynomialFeatures
  params: {columns: [income, tenure], degree: 2, output_prefix: poly}
```

**Common mistake:** High degrees can create many columns; include_input_features does not control
preservation of original inputs.

**Details:** Saves layout/configuration. Appends terms up to degree, with names such as
`poly_income_pow_2`, `poly_income_tenure`; original columns remain. Optional
bias is a constant term. Numeric finite input is required by sklearn; impute
before expansion. Empty apply derives the same layout. Generated-name collisions
raise. Context row; base/sklearn dependencies. Degree can grow output width rapidly.

## FeatureGeneration

Aliases: `FeatureGenerationNode`, `FeatureMath` select this same owner.

**Purpose and effect:** two numeric columns can add a sum, ratio or saved group statistic; fixed
arithmetic and fitted group operations can share one list.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `operations` | list[mapping] | [] | Ordered operations described below. | Generate new features, allowing later operations to read earlier outputs. |
| `epsilon` | number | 1e-9 | Small numeric guard. | Protect division near zero. |
| `allow_overwrite` | boolean | false | true, false | Overwrite named outputs, or suffix collisions. |
| `operations[].operation_type` | string | arithmetic | arithmetic, ratio, similarity, datetime_extract, group_agg | Choose calculation family. |
| `operations[].input_columns` | list[string] | [] | Existing inputs appropriate to the operation. | Supply primary operands, the numerator or grouping key. |
| `operations[].secondary_columns` | list[string] | [] | Existing inputs appropriate to the operation. | Supply secondary operands, the denominator or grouped value. |
| `operations[].method` | string | Operation-dependent | Choices in the operation table below. | Choose arithmetic, similarity or aggregate behavior. |
| `operations[].output_column` | string/null | null | Optional explicit name. | Name the output; datetime operations may expand it for multiple sources/features. |
| `operations[].output_prefix` | string/null | null | Optional prefix. | Prefix automatically generated output names. |
| `operations[].round_digits` | integer/null | null | Integer. | Round numeric output. |
| `operations[].constants` | list[number] | [] | Arithmetic only. | Add numeric operands. |
| `operations[].fillna` | scalar/null | Effective 0 | Arithmetic only. | Fill missing numeric operands. |
| `operations[].datetime_features` | list[string] | [] | Calendar choices below. | Choose extracted datetime components. |

**Minimal YAML:**

```yaml
- name: featuregeneration
  transformer: FeatureGeneration
  params:
    operations:
      - {operation_type: arithmetic, method: add, input_columns: [income, bonus], output_column: total}
      - {operation_type: group_agg, input_columns: [segment], secondary_columns: [total], method: mean, output_column: segment_mean}
```

**Common mistake:** A fitted group aggregate reuses training lookups. Unseen groups yield missing
for existing aggregate methods; `deviation` uses the saved global training mean.

**Details:** Additional fields and choices:

| Operation | Input and options |
| --- | --- |
| `arithmetic` | `method`: add, subtract, multiply, divide; `constants` numeric list default `[]`; `fillna` scalar/null default effective `0`. Numeric inputs/secondary columns and constants participate in configured order. |
| `ratio` | Sums input columns as numerator and secondary columns as denominator; near-zero/missing denominator uses signed epsilon. |
| `similarity` | First input and first secondary (or second input); `method` ratio, token_sort_ratio or token_set_ratio. Backend is selected and saved at fit, not a user-supplied artifact field. |
| `datetime_extract` | `datetime_features` list default `[]`: year, quarter, month, month_name, week, day, day_name, weekday, is_weekend, hour, minute, second, season, time_of_day. |
| `group_agg` | First input is group key, first secondary is numeric value; `method` default mean, also sum, min, max, count, std, median, deviation. |

The fitted artifact also stores these fields **inside each operation**. They are
created during fit, not configurable YAML parameters. Supplying them in a recipe
does not override fitting: fit replaces them with its own values.

| Saved field | Type and values | How it is created and used |
| --- | --- | --- |
| `operations[].similarity_backend` | `rapidfuzz` or `difflib` | Similarity fitting selects `rapidfuzz` when installed, otherwise `difflib`. Apply reuses the saved choice; a saved rapidfuzz model requires that dependency. With difflib, all method names use its sequence ratio, not rapidfuzz token matching. |
| `operations[].group_agg_mapping` | Mapping or null | Group fitting saves `group_column`, parallel `keys`/`values` lists and `null_value` for the missing-key group. Null records an unresolved fit operation. Apply maps incoming keys to saved numbers; it does not aggregate the request batch. |

Operations run in order; later operations can read earlier outputs. A group
aggregate learns a training lookup (including missing-key group) and reuses it;
an unseen group yields missing for ordinary aggregate methods. There is no
request-time refit. Default generated
names use `<operation_type>_<index>` with optional prefix; datetime outputs use
source/feature names, expanding a supplied name when multiple sources/features
are requested. Existing names receive a numeric suffix unless overwrite is true.
New fits validate operation types, named methods and calendar choices before
publishing state; errors while evaluating supported operations propagate.
`polynomial` is not a supported
operation_type: use PolynomialFeatures. Similarity uses optional rapidfuzz when
available, otherwise difflib; a saved rapidfuzz artifact requires that package
on replay. Null/empty string pairs score 100 when both empty, 0 when one empty.
Context is row for supported numeric/group lookup operations; pandas similarity
can be global due to string rendering. Other dependencies: base.

New fits save `operation_contract_version: 1`. Numeric `datetime_extract`
outputs use nullable `Int64` on both local engines, including invalid dates,
singletons and empty batches. Missing numeric calendar values remain missing.
Unversioned saved artifacts retain their historical dtypes and logged-skip
behavior; refit to adopt the checked operation contract. Arithmetic zero/null
defaults and signed-epsilon division are unchanged.

For `method: deviation`, each output is the source value minus its group's
training mean. Unseen groups and groups with no usable mean use the global
training mean. A learned missing-key group keeps its own mean; a missing source
value stays missing. All outputs are floating point. For example, training
values `[2, 4]` in group A produce mean 3, so a later value 8 produces 5 even
when scored alone. The saved mapping adds `category_key_version: 1` and
`global_value`; typed group keys distinguish Boolean, numeric and text values.
This extends the existing operation; old saved aggregate methods retain their
lookup representation and behavior.

```yaml
- name: income_deviation
  transformer: FeatureMath
  params:
    operations:
      - {operation_type: group_agg, method: deviation, input_columns: [segment], secondary_columns: [income], output_column: income_deviation}
```

## VarianceThreshold

**Purpose and effect:** remove a constant numeric feature with threshold 0.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Numeric candidates, including binary; [] also auto-selects. | Choose candidate features; target is excluded. |
| `threshold` | number | 0.0 | Nonnegative. | Retain variance strictly above this value. |
| `drop_columns` | boolean | true | true, false | Remove rejected candidates, or record selection only. |

**Minimal YAML:**

```yaml
- name: variancethreshold
  transformer: VarianceThreshold
  params: {columns: [age, constant_flag], threshold: 0.0, drop_columns: true}
```

**Common mistake:** If every candidate is constant or all-missing, every candidate can be removed.

**Details:** Learns training variance/retained columns and saves that set.
False records selection but does not remove columns. Unselected unrelated
columns remain; target is excluded. NaNs are ignored for variance estimation;
all-missing candidates are rejected. If no candidate clears the threshold, all
candidates are removed when `drop_columns` is true. This does not impute values. Apply never
remeasures variance. Context row; base/sklearn dependencies.

## CorrelationThreshold

**Purpose and effect:** remove one of two strongly correlated inputs.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Numeric candidates, including binary; [] also auto-selects. | Choose candidate features; target is excluded. |
| `threshold` | number | 0.95 | Normally 0–1. | Choose correlation strength for removing a redundant input. |
| `correlation_method` | string | pearson | pearson, spearman, kendall | Choose fitted correlation measure. |
| `drop_columns` | boolean | true | true, false | Remove rejected candidates, or record selection only. |

**Minimal YAML:**

```yaml
- name: correlationthreshold
  transformer: CorrelationThreshold
  params: {columns: [income, income_copy], threshold: 0.95, correlation_method: pearson}
```

**Common mistake:** Candidate column order can determine which of a correlated pair survives.

**Details:** Saves correlated columns chosen for removal
from training correlations. Column order matters when choosing which redundant
feature survives. Constant/missing correlations do not establish strong
correlation. Apply drops the saved set only, or no columns when drop_columns is
false. Context row; base dependencies.

## UnivariateSelection

**Purpose and effect:** keep the two numeric inputs most associated with y.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Numeric candidates, including binary; [] also auto-selects. | Choose candidate features; target is excluded. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |
| `method` | string | select_k_best | select_k_best, select_percentile, select_fpr, select_fdr, select_fwe, generic_univariate_select; sklearn PascalCase aliases. | Choose selection rule. |
| `score_func` | string/null | auto | f_classif, f_regression, mutual_info_classif, mutual_info_regression, chi2, r_regression. | Choose association score. |
| `problem_type` | string | auto | auto, classification, regression | Resolve default scorer. |
| `allow_missing_target` | boolean | false | true, false | Allow configured missing-target fallback. |
| `drop_columns` | boolean | true | true, false | Remove rejected candidates. |
| `k` | integer/string | 10 | Positive integer or all. | Select-k-best feature count. |
| `percentile` | number | 10 | 0–100. | Select-percentile share. |
| `alpha` | number | 0.05 | Significance level for FPR/FDR/FWE. | Set statistical threshold. |
| `mode` | string | k_best | k_best, percentile, fpr, fdr, fwe | Choose generic selector mode. |
| `param` | number/string | Mode-dependent | Explicit override of k/percentile/alpha. | Set generic selector parameter. |

**Minimal YAML:**

```yaml
- name: univariateselection
  transformer: UnivariateSelection
  params: {columns: [age, income, tenure], method: select_k_best, k: 2, problem_type: classification}
```

**Common mistake:** For low-cardinality numeric regression targets, declare problem_type: regression
instead of relying on auto.

**Details:** Statistical p-value selectors need
a scorer returning p-values. `chi2` uses nonnegative scaled values during fitting.
Requires training y unless explicit missing-target behavior is selected; missing
classification labels fail. Auto problem type uses a cardinality heuristic for
numeric labels: declare regression for low-cardinality count targets. Saves
selected/candidate columns and scores; apply only drops the saved rejected
columns. Context row; base/sklearn dependencies.

## ModelBasedSelection

**Purpose and effect:** retain inputs with sufficient fitted importance or recursively eliminate
inputs down to a requested count.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string]/null | null | Numeric candidates, including binary; [] also auto-selects. | Choose candidate features; target is excluded. |
| `target_column` | string/null | null | Existing label name; paired y can supply the label. | Identify the supervised target, normally supplied by the pipeline. |
| `method` | string | select_from_model | select_from_model, rfe, rfecv, sequential_feature_selection | Choose importance, recursive or sequential selection. |
| `estimator` | string | auto | auto, logistic_regression/logisticregression, linear_regression/linearregression, random_forest/randomforest. | Choose fitted importance estimator. |
| `problem_type` | string | auto | auto, classification, regression | Resolve suitable estimator. |
| `drop_columns` | boolean | true | true, false | Remove rejected candidates. |
| `threshold` | number/string | mean | Sklearn threshold expression, such as mean, median or a multiplier. | Select-from-model importance cutoff. |
| `max_features` | integer/null | null | Select-from-model only. | Cap selected features. |
| `n_features_to_select` | integer/number/string/null | RFE: null; sequential: auto | RFE count/fraction/null; sequential count, fraction in (0,1), or auto. | Set retained features; sequential auto selects half. |
| `k` | integer/null | null | Legacy RFE fallback when n_features_to_select is unset. | Set retained feature count. |
| `step` | integer/number | 1 | Positive count or fraction for RFE/RFECV. | Set elimination size. |
| `cv` | integer | 5 | At least 2; RFECV/sequential only. | Set unshuffled cross-validation folds. |
| `scoring` | string/null | null | sklearn scorer name or estimator default. | Choose the CV selection criterion. |
| `min_features_to_select` | integer | 1 | Positive; RFECV only. | Set the smallest evaluated feature subset. |
| `direction` | string | forward | forward, backward; sequential only. | Add or remove features greedily. |

**Minimal YAML:**

```yaml
- name: modelbasedselection
  transformer: ModelBasedSelection
  params: {columns: [age, income, tenure], method: select_from_model, estimator: random_forest, threshold: mean}
```

**Common mistake:** Supply training y or target_column; a missing target or unsupported estimator
raises an error instead of disabling selection.

**Details:** Uses a resolved estimator suitable for the problem type. Unknown methods, estimators
and problem types raise with supported choices; estimator fit errors propagate. Logistic regression
is supported only for classification. Saves selected columns/importance evidence. Requires numeric
inputs and training y; no candidate columns remains an empty no-op. Candidate
NaNs are filled with zero for selection fitting only; output values are not imputed. Apply preserves
unrelated columns and drops saved rejected candidates only. Context row; base/sklearn dependencies.

RFECV chooses the subset size by cross-validation. Sequential selection chooses
each next feature using cross-validation; its separate `n_features_to_select`
setting does not inherit the legacy RFE `k`. Integer CV uses sklearn's
unshuffled stratified folds for classification and ordinary folds otherwise.
Folds are not automatically reduced: small training populations must choose a
valid lower `cv`. These methods repeat estimator fits and can be expensive on
wide data. Saved inference only uses fixed selected names and does not repeat CV.
Sequential selection has no fitted feature-importance report, so it saves an
empty importance mapping instead of inventing scores.

```yaml
- name: recursive_cv
  transformer: ModelBasedSelection
  params: {method: rfecv, cv: 3, min_features_to_select: 1, step: 1}
```

## feature_selection

**Purpose and effect:** Choose a concrete selection algorithm through a method field; it does not
add another algorithm.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `method` | string | select_k_best | Aliases listed below. | Dispatch to the corresponding concrete selector. |
| `threshold` | number/string, depending on selector | Selected algorithm's default | For `variance`, a nonnegative number (default 0.0); for `correlation_threshold`, a correlation limit (default 0.95). Other selectors define their own fields. | Forwarded unchanged to the chosen calculator; it is not ignored by variance/correlation selection. |
| `Other fields` | See selected card | See selected card | VarianceThreshold, CorrelationThreshold, UnivariateSelection or ModelBasedSelection options. | Forward configuration unchanged. |

**Minimal YAML:**

```yaml
- name: feature_selection
  transformer: feature_selection
  params: {method: variance, columns: [age, constant_flag], threshold: 0.0}
```

**Common mistake:** Use correlation_threshold, not correlation, as the dispatcher method.

**Details:** `variance`/`variance_threshold` route to VarianceThreshold; `correlation_threshold` to
CorrelationThreshold; `select_k_best`, `select_percentile`, `select_fpr`, `select_fdr`,
`select_fwe`, `generic_univariate_select` to UnivariateSelection; `select_from_model`, `rfe`,
`rfecv`, `sequential_feature_selection` to
ModelBasedSelection. The matching card defines required target, defaults, state, output and context.
Unknown methods raise with supported choices. Existing saved artifacts still replay their fitted
selection without resolving the fit method again. Prefer the concrete selector ID when
readability is more valuable than a single dispatcher field.
