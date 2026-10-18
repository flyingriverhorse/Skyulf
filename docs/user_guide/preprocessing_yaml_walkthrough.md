# Build preprocessing with YAML

Configure a generated Databricks project's preprocessing by editing YAML. This
walkthrough uses built-in steps: the shipped `src/features/` Python files can
stay unchanged. It covers fixed eligibility, missing values, categorical
encoding, scaling, outlier exclusion and saved prediction. The examples use
the preprocessing additions in **Skyulf Core 0.9.4**.

Use the [preprocessing catalog](preprocessing_recipes.md) for every step's
parameter types, defaults, allowed values and output contract. The
[complete list below](#find-every-step-and-its-parameters) links all registered
names, including operations that require custom Python or standalone use.

## 1. Start from a generated project

Generate a **single-model classification** project using the
[Bundle setup guide](databricks_bundle.md). Keep its source table, record keys,
registry, training split and deployment settings. This walkthrough replaces
two complete recipe files and updates selected fields in `config/training.yml`.
It does not create a Delta table or deploy an endpoint by itself.

| File | What belongs here | When it runs |
| --- | --- | --- |
| `config/pre_split.yml` | Fixed input normalization and eligibility, such as an age range. | Before splitting training data; the generated scoring policy reuses feature-only rules. |
| `config/preprocessing.yml` | Imputation, learned category maps, scaling, feature selection and learned outlier detection. | Fitted on training rows; fitted again inside training folds when cross-validation fits a fold. Saved state is replayed at prediction. |
| `config/training.yml` | Input columns, target, model and the names of the two selected recipes. | Selects what a training run builds. |

A recipe is an ordered list of steps. Only the named recipe runs. `default: []`
and `none: []` below deliberately do nothing; defining four recipes does not
combine them. Each built-in step has a unique `name`, a case-sensitive
`transformer` and its `params` mapping.

Use a table with these illustrative rows and columns. `customer_id` is the
record key, outside model inputs. The tiny dataset demonstrates the contracts;
it is not a benchmark or a recommendation for production split sizes.

<!-- example:yaml-customer-data -->
```csv
customer_id,age,income,tenure,segment,churn
1,20,1000,1,A,0
2,30,2000,2,A,0
3,40,,3,A,1
4,50,4000,4,B,1
5,60,5000,5,B,0
6,70,,6,B,1
7,25,1500,2,A,0
8,35,2500,3,A,1
9,45,3500,4,B,0
10,55,4500,5,B,1
11,65,5500,6,B,0
12,75,6500,7,B,1
13,22,1200,1,A,0
14,32,2200,2,A,1
15,42,-200,3,A,0
16,52,4200,4,B,1
17,62,5200,5,B,0
18,72,6200,6,B,1
19,27,1700,2,A,0
20,37,2700,3,A,1
21,47,3700,4,B,0
22,57,4700,5,B,1
23,67,5700,6,C,0
24,77,6700,7,B,1
25,24,1400,1,A,0
26,34,2400,2,A,1
27,44,3400,3,C,0
28,54,4400,4,B,1
29,64,5400,5,B,0
30,74,6400,6,B,1
31,28,1800,2,A,0
32,150,3000,3,A,1
```

All labels are present. For delayed or separate labels, configure
[training outcomes](databricks_training_data.md) rather than fabricating labels
in an imputer. Load your table separately, and use enough labeled rows for the
split and CV settings you retain from the generated project.

## 2. Apply fixed rules before the split

Replace **`config/pre_split.yml`** with this complete file:

<!-- example:yaml-customer-pre-split -->
```yaml
version: 1
recipes:
  default: []
  none: []
  eligible:
    - name: income_marker
      transformer: ValueReplacement
      params: {columns: [income], to_replace: -200, value: null}
    - name: age_range
      transformer: ManualBounds
      params: {bounds: {age: {lower: 0, upper: 120}}}
```

The marker `-200` becomes a missing measurement; it is not included in the mean
learned later. The fixed age rule removes row 32 before the training split.
Bounds are inclusive. `ManualBounds` retains missing values; use an explicit
missing-value rule if a missing age should be ineligible rather than imputed.

Both rules read features only. The generated `mode: pre_split` policy in `config/scoring.yml`
can reuse them without editing Python or enabling a target-skip switch.
At scoring time an out-of-range age keeps its result row with
`scoring_status=excluded`, an empty prediction and
`exclusion_reason=pre_split:age_range`.

Pre-split accepts a restricted set of fixed operations: `DropMissingRows`,
`ManualBounds`, `Deduplicate`, `ValueReplacement`, `TextCleaning`,
`AliasReplacement`, `InvalidValueReplacement` and `Casting`. Each has additional
[pre-split constraints](preprocessing_recipes.md#pre-split-admission-and-scoring).
Learned means, quantiles and outlier boundaries belong in preprocessing, not
before the split. If a rule reads the target, follow the explicit
[target-filter policy](databricks_custom_pre_split.md) instead of copying the
feature-only policy in this example.

## 3. Define alternative fitted recipes

Replace **`config/preprocessing.yml`** with this complete file. The recipes
share the same raw input columns, so switch between them by changing one
recipe name in training configuration.

<!-- example:yaml-customer-preprocessing -->
```yaml
version: 1
recipes:
  default: []
  none: []
  customer:
    - name: fill_numeric
      transformer: SimpleImputer
      params: {columns: [age, income, tenure], strategy: mean}
    - name: rare_segments
      transformer: RareCategoryGrouping
      params: {columns: [segment], min_count: 3, replacement: Other}
    - name: segment_frequency
      transformer: FrequencyEncoder
      params: {columns: [segment], mode: frequency}
    - name: numeric_quantiles
      transformer: QuantileTransformer
      params:
        columns: [age, income, tenure]
        n_quantiles: 16
        output_distribution: uniform
        subsample: null
        random_state: 42
    - name: numeric_row_norm
      transformer: Normalizer
      params: {columns: [age, income, tenure], norm: l2}
  fixed_categories:
    - name: fill_numeric
      transformer: SimpleImputer
      params: {columns: [age, income, tenure], strategy: mean}
    - name: scale_measurements
      transformer: StandardScaler
      params:
        columns_regex: '^(age|income|tenure)$'
        dtype: numeric
        exclude_columns: [customer_id]
    - name: fixed_segments
      transformer: OneHotEncoder
      params:
        columns: [segment]
        categories: {segment: [A, B, C]}
        column_names: slug
        max_categories: null
        include_missing: false
        handle_unknown: ignore
  isolation:
    - name: fill_numeric
      transformer: SimpleImputer
      params: {columns: [age, income, tenure], strategy: mean}
    - name: unusual_measurements
      transformer: IsolationForest
      params:
        columns: [age, income, tenure]
        contamination: 0.1
        n_estimators: 40
        random_state: 42
    - name: segment_frequency
      transformer: FrequencyEncoder
      params: {columns: [segment], mode: frequency}
  local_density:
    - name: fill_numeric
      transformer: SimpleImputer
      params: {columns: [age, income, tenure], strategy: mean}
    - name: scale_measurements
      transformer: StandardScaler
      params: {columns: [age, income, tenure]}
    - name: unusual_measurements
      transformer: LocalOutlierFactor
      params: {columns: [age, income, tenure], n_neighbors: 5, contamination: 0.1}
    - name: segment_frequency
      transformer: FrequencyEncoder
      params: {columns: [segment], mode: frequency}
```

Choose the recipe for the behavior you need; these are alternatives, not an
instruction to stack every available transform:

| Recipe | What happens | What the saved model remembers |
| --- | --- | --- |
| `customer` | Impute numeric inputs, group rare segments, encode group frequency, transform each numeric column by its quantiles, then normalize each row's numeric vector. | Training means, frequent-category membership, group frequencies and quantile maps; the selected columns and normalization rule. |
| `fixed_categories` | Impute and standardize numeric measurements; emit exactly `segment_a`, `segment_b`, `segment_c`. | Numeric means/scales and the explicit vocabulary/order, including levels absent from training. |
| `isolation` | Impute, exclude rows outside the learned IsolationForest boundary, and frequency-encode segment for surviving training rows. | The fitted detector and downstream category map. |
| `local_density` | Impute, scale numeric distances, exclude rows using the fitted LocalOutlierFactor detector, then encode segment. | Numeric scales, the novelty detector and downstream category map. |

Quantile transformation changes each column's distribution. L2 normalization
then makes each nonzero selected **row vector** have length one; it does not
standardize columns. They are shown together to demonstrate both operations.
For a random forest neither is automatically necessary; compare validation
results before retaining extra steps.

Rare grouping sends an unseen segment to `Other`. The next frequency encoder
uses the saved frequency of that group, or zero if no such group was learned.
Frequency encoding alone sends an unseen category to zero. Neither recomputes
frequencies from the scoring request.

Fixed one-hot requires explicit string vocabularies, `max_categories: null` and
`include_missing: false`. Here `NEW` or a null segment produces three zeros;
changing `handle_unknown` to `error` rejects them instead. `column_names: slug`
normalizes output names; conflicting normalized names are rejected at fit.

Outlier filtering changes the training population. At prediction, the scoring
workflow preserves an outlier's row and reports
`exclusion_reason=outlier:unusual_measurements`. The first exclusion reason wins:
an age already rejected by the fixed rule keeps `pre_split:age_range`.
`contamination: 0.1` controls the fitted cutoff; it does not promise that exactly
10% of future requests will be excluded. Review the exclusion counts as well as
the model metric before choosing a detector.

## 4. Select and inspect the recipe

Merge this **fragment** into the generated **`config/training.yml`**. Keep its
table, record keys, registry, split, resource and quality settings. Select a
classification quality metric and remove an existing `tuning` block if you want
only the fixed model below. Use the generated model key if it differs from `main`.

<!-- example:yaml-customer-training -->
```yaml
version: 1
defaults:
  task: classification
  input_columns: [age, income, tenure, segment]
  target_column: churn
  pre_split_recipe: eligible
  preprocessing_recipe: customer
models:
  main:
    model:
      type: random_forest_classifier
      params: {n_estimators: 40, max_depth: 6, n_jobs: 1, random_state: 42}
```

Change `preprocessing_recipe` to `fixed_categories`, `isolation` or
`local_density` to train that alternative. A model entry can override the
defaults. Competition candidates can use different preprocessing recipes while
sharing pre-split eligibility and holdout membership; independent-target models
can select their own recipes. Follow the generated layout for your project type.

Run from the generated project directory:

```bash
python src/tools/smoke.py
python src/tools/preview.py --action train
python src/tools/preview.py --action score
python src/tools/preview.py --list-preprocessors
```

Preview resolves the configuration without training the model. Inspect the
selected recipes and input columns, then use your project's generated
[training and scoring commands](databricks_bundle_walkthrough.md).
The registry listing shows metadata defaults; the parameter cards below explain
the full accepted configuration. Successful preview does not prove that a
particular input dataset or serving configuration will be admitted.

### Select columns once, at fit

The `fixed_categories` scaler demonstrates three shared selector options:

| Option | Effect in the example |
| --- | --- |
| `columns_regex: '^(age\|income\|tenure)$'` | Select these exact names; matching uses a regular-expression search, so anchors matter. |
| `dtype: numeric` | Require numeric feature columns as well as a matching name. |
| `exclude_columns: [customer_id]` | Keep identifiers out even if a broader pattern is used later. The key is already outside model inputs here. |

The rules combine with AND and resolve once from training features. Saved
prediction uses the resulting explicit names. An empty selection raises an
error. These rules apply to preprocessing steps that accept `columns`; they do
not apply to pre-split rules or directly to Calculator calls. Native Spark
training uses explicit names. The Canvas uses its explicit column picker.
See the [selection contract](preprocessing_recipes.md#select-columns-from-the-training-schema)
for dtype details and the interaction with an explicit `columns` list.

## 5. Check saved prediction behavior

Send raw model inputs in their recorded types; do not manually reproduce the
mean, scale or category columns in a client. For example:

<!-- example:yaml-customer-query -->
```csv
customer_id,age,income,tenure,segment
101,40,3000,3,A
102,40,,3,NEW
103,150,3000,3,A
104,40,10000000,5000,B
105,40,-200,3,A
```

The source record key associates each result with its input; it is not a model
feature. With `customer` or `fixed_categories`, rows 101, 102, 104 and 105 are
predicted, while 103 is excluded by the age rule. Missing income and the marker
are imputed using the training mean. The two detector recipes can also exclude
104 for its extreme measurements. Their learned boundaries, and therefore other
accepted rows, depend on the training split.

The published scoring table keeps excluded rows, their status and reason, with
null predictions. This is different from asking a bare preprocessing applier to
remove rows. Use the complete saved scoring workflow for that contract. See
[scoring and incremental inputs](databricks_scoring.md) for output columns.

Training saves fitted transformations and the selected project rules with the
model. Editing YAML affects the **next training run**; it does not change an
existing registered version. Load and replay the saved artifact to check the
behavior you intend to deploy.

| Execution path | What must be checked |
| --- | --- |
| pandas / Polars | The selected configuration, input dtypes, missing values and saved replay on the chosen native engine. |
| Spark worker scoring | Admission of the fitted artifact for independent batches, its input/output schema and worker dependencies. This is not distributed training of the estimator. |
| Serving endpoint | Admission of the complete serving workflow, its raw input contract, saved rules and dependencies. A working local pipeline alone does not establish endpoint support. |
| Lag or rolling features | Ordered history must be supplied through the supported context workflow. A model does not fetch prior customer records by itself. |

The recipes use saved state plus individual request rows during prediction.
In contrast, history-dependent operations require additional observations.
Use [preprocessing diagnostics](preprocessing_context.md) and the
[Spark guide](spark.md) for artifact-specific checks. Do not infer support for
every configuration from the existence of a registry entry.

## New nodes and additions to existing nodes

The six new registered operations are **FrequencyEncoder**, **RareCategoryGrouping**,
**QuantileTransformer**, **Normalizer**, **IsolationForest** and
**LocalOutlierFactor**. In the Canvas they are choices within the existing
Encoding, Scaling and Outlier panels, not six separate palette cards.

Other additions are parameters or operations within existing nodes:

| Existing operation | Addition and where to read the complete configuration |
| --- | --- |
| OneHotEncoder | [Fixed string vocabularies and slug output names](preprocessing_recipes_encoding.md#onehotencoder). |
| GroupImputer | [Multiple group keys and hierarchical fallback](preprocessing_recipes_cleaning.md#groupimputer). An unknown full key can fall back to a shorter key, then the saved global statistic. |
| FeatureGeneration | [Group deviation](preprocessing_recipes_features.md#featuregeneration) subtracts a saved group reference from a value. |
| DateFeatures | [Local timezone and cyclic date features](preprocessing_recipes_time_geo.md#datefeatures), such as month sine/cosine. |
| count_vectorizer / tfidf_vectorizer | [Per-column vocabularies](preprocessing_recipes_text.md) separate text fields instead of joining them. |
| Winsorize | [Per-column percentile bounds](preprocessing_recipes_outliers.md#winsorize). |
| ModelBasedSelection | [RFECV and sequential feature selection](preprocessing_recipes_features.md#modelbasedselection), with their own CV and feature-count controls. |
| TargetEncoder | [Cross-fitting fold count, shuffle and seed](preprocessing_recipes_encoding.md#targetencoder). Training encodings remain separate from full-training maps used for prediction. |
| MinMaxScaler / RobustScaler | [Clipping](preprocessing_recipes_scaling.md#minmaxscaler) and [unit-variance adjustment](preprocessing_recipes_scaling.md#robustscaler). |
| Steps accepting columns | [Pattern, dtype and exclusion selectors](preprocessing_recipes.md#select-columns-from-the-training-schema) in Core pipelines and YAML. |

## Find every step and its parameters

Each linked family contains per-step parameter tables with **type, default,
allowed values, effect, minimal YAML and saved-state behavior**. There are 73
registered names and 69 distinct implementations, including four aliases. The
following inventory includes standalone and custom-only operations; registration
is not a promise that every name can be pasted into a Bundle recipe.

<!-- yaml-catalog:start -->

| Parameter reference | Registered names |
| --- | --- |
| [Missing values, cleaning and types](preprocessing_recipes_cleaning.md) | `AliasReplacement`, `Casting`, `Deduplicate`, `DropMissingColumns`, `DropMissingRows`, `GroupImputer`, `InvalidValueReplacement`, `IterativeImputer`, `KNNImputer`, `MissingIndicator`, `SimpleImputer`, `TextCleaning`, `ValueReplacement` |
| [Bounds and outliers](preprocessing_recipes_outliers.md) | `ClipValues`, `EllipticEnvelope`, `IQR`, `IsolationForest`, `LocalOutlierFactor`, `ManualBounds`, `Winsorize`, `ZScore` |
| [Custom factory contracts and ready functions](preprocessing_recipes_custom.md) | `ColumnFunction`, `FittedFunction`, `RowFilterFunction` |
| [Feature generation and selection](preprocessing_recipes_features.md) | `CorrelationThreshold`, `FeatureGeneration`, `FeatureGenerationNode`, `FeatureInteraction`, `FeatureMath`, `ModelBasedSelection`, `PolynomialFeatures`, `PolynomialFeaturesNode`, `UnivariateSelection`, `VarianceThreshold`, `feature_selection` |
| [Binning](preprocessing_recipes_binning.md) | `CustomBinning`, `GeneralBinning`, `KBinsDiscretizer` |
| [Split, sampling and diagnostic Core nodes](preprocessing_recipes_sampling.md) | `DataSnapshot`, `DatasetProfile`, `Oversampling`, `Split`, `TrainTestSplitter`, `Undersampling`, `feature_target_split` |
| [Time and geography](preprocessing_recipes_time_geo.md) | `DateFeatures`, `GeoDistance`, `H3Index`, `LagFeatures`, `RollingAggregate` |
| [Categorical encoding](preprocessing_recipes_encoding.md) | `DummyEncoder`, `FrequencyEncoder`, `HashEncoder`, `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder`, `RareCategoryGrouping`, `TargetEncoder`, `WOEEncoder` |
| [Scaling and numeric transformations](preprocessing_recipes_scaling.md) | `GeneralTransformation`, `MaxAbsScaler`, `MinMaxScaler`, `Normalizer`, `PowerTransformer`, `QuantileTransformer`, `RobustScaler`, `SimpleTransformation`, `StandardScaler` |
| [Text and embeddings](preprocessing_recipes_text.md) | `count_vectorizer`, `hashing_vectorizer`, `sentence_embedder`, `tfidf_vectorizer`, `tokenizer` |

<!-- yaml-catalog:end -->

Most cards show list fragments: paste them under a named recipe in the correct
phase file. `ColumnFunction`, `FittedFunction` and `RowFilterFunction` need a
captured Python factory because YAML cannot supply a callable. Split nodes are
standalone Core operations; let the Bundle own its train/holdout split. Sampling
is training-only, while profile/snapshot nodes are diagnostics. Optional text,
geospatial and sampling packages must be installed where the model runs. Each
card states these restrictions; use the [registry index](preprocessing_recipes.md#registry-index-and-scope)
to jump directly to an individual operation.
