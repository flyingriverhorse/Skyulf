# Preprocessing and pre-split YAML reference

Use this reference to choose a recipe, configure each step, and understand what
the saved model will replay. A recipe is an ordered list. A transformer is a
registered Core operation. A custom factory builds one step from project Python.
These are separate choices: registering an operation does not enable it in every
phase or runtime.

For complete recipe files using built-in operations without editing Python, start
with [Build preprocessing with YAML](preprocessing_yaml_walkthrough.md). This
reference supplies the full parameter tables for the steps used there and every
other registered operation.

For complete custom Python factories and their exact YAML selectors, see
[custom steps](databricks_custom_steps.md) and
[custom pre-split filters](databricks_custom_pre_split.md). For
saved-state diagnostics and lag/rolling continuation, see
[preprocessing context](preprocessing_context.md).

`preview.py --list-preprocessors` displays selected configuration defaults from
the installed Core registry. Use the parameter tables on the family pages for the full set of
supported options and required inputs. Preview's flags and limits are in the
[local tools reference](databricks_project_setup.md#local-tools-and-preview)
and **Local tools** in the generated `GUIDE.md`.

The `feature_selection` wrapper forwards its options
to the selected algorithm: for example, `threshold` works with `method: variance`.

## Choose the phase and recipe

| Location | Work performed |
| --- | --- |
| `config/features.yml` and `src/features/groups/` | Optional upstream Spark features and joins, materialized into Delta tables. |
| `config/pre_split.yml` | Fixed normalization and eligibility filters before holdout/CV splitting. No learned population statistics. |
| `config/preprocessing.yml` | Model transformations fitted on training rows, inside each training fold when CV/search fits a fold. |
| `config/training.yml` | Selects recipes, model inputs, target, split and model settings. |
| `src/features/preprocessing.py` / `pre_split.py` | Shipped factories, selected as `preprocessing.<factory>` or `pre_split.<factory>`. |
| `src/features/<file>.py` | Your own module of factories, selected as `<file>.<factory>`. |

Both phase files use `version: 1` and a `recipes` mapping with a `default` entry.
Recipe names start with a letter, contain only ASCII letters, digits, `_` or `-`,
and are at most 64 characters. An entry named `none` must be an empty list.
Only the selected recipe runs; defining a function or example does not run it.
Built-in steps require a nonempty `name` label, not a Python filename.
For custom factories, `name` is optional and overrides the returned step label.
`transformer` is case-sensitive; `params` is a mapping. Use `custom` instead of
`transformer` for a captured project factory, and give its keyword arguments in
`params`. Do not define the same phase in YAML and a Python `build_preprocessing`
or `build_pre_split_steps` function.

The generated project ships these recipes:

| Phase | Recipe | Ordered steps |
| --- | --- | --- |
| Preprocessing | `default`, `none` | Empty; no automatic filling or encoding. |
| Preprocessing | `example_frequency` | Custom frequency encoding of `category`. |
| Preprocessing | `example_imputer` | Mean imputation of `feature_value`. |
| Preprocessing | `example_imputer_frequency` | Mean imputation, then frequency encoding. |
| Preprocessing | `example_all` | Mean imputation, log feature, rare categories, frequency encoding. This means all shipped examples, not all registered transformers. |
| Pre-split | `default`, `none` | Empty. |
| Pre-split | `example_complete_inputs` | Custom minimum completeness. |
| Pre-split | `example_all` | Minimum completeness, then a fixed value-range filter. |

Set `defaults.preprocessing_recipe` and `defaults.pre_split_recipe` in
`training.yml`; model entries can override these selections. Competition
candidates may use different preprocessing recipes but share eligibility and
holdout membership. Independent-target models can select their own recipes.
Keep the generated model layout rather than inserting `pipeline.preprocessing`
or `pipeline.modeling`: nonempty inline pipeline definitions are rejected.

### Complete phase files for a small customer example

Start with a generated single classification project. Use an input table with
this shape (the table needs enough rows of each class for your split/CV settings):

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
```

Keep `customer_id` as the generated record key, outside model inputs. The
following is a **training.yml fragment**: merge the fields into the existing
file and retain its table, registry, resource, split and quality settings. Remove
an existing `tuning` block to fit just this fixed model. Choose a classification
quality metric.

```yaml
version: 1
defaults:
  task: classification
  input_columns: [age, income, tenure, segment]
  target_column: churn
  preprocessing_recipe: customer
  pre_split_recipe: eligible
models:
  main:
    model:
      type: random_forest_classifier
      params: {n_estimators: 60, max_depth: 6, n_jobs: 1, random_state: 42}
```

Replace `config/pre_split.yml` with this complete file:

```yaml
version: 1
recipes:
  default: []
  none: []
  eligible:
    - name: dropmissingrows
      transformer: DropMissingRows
      params: {subset: [churn]}
    - custom: pre_split.value_range
      params: {column: age, minimum: 0, maximum: 120}
```

Replace `config/preprocessing.yml` with this complete file:

```yaml
version: 1
recipes:
  default: []
  none: []
  customer:
    - name: segment_fill
      transformer: SimpleImputer
      params: {columns: [segment], strategy: most_frequent}
    - name: group_income
      transformer: GroupImputer
      params: {columns: [income], group_by: segment, strategy: mean}
    - name: simpleimputer
      transformer: SimpleImputer
      params: {columns: [age, tenure], strategy: mean}
    - name: featureinteraction
      transformer: FeatureInteraction
      params: {columns: [income, tenure], degree: 2}
    - name: standardscaler
      transformer: StandardScaler
      params: {columns: [age, income, tenure, income_x_tenure]}
    - name: onehotencoder
      transformer: OneHotEncoder
      params: {columns: [segment], handle_unknown: ignore, max_categories: null}
```

Before preview or training, set the complete policy in `config/scoring.yml`:

```yaml
version: 1
mode: pre_split
skip_target_pre_split_steps: true
eligibility: []
outputs: []
combined: []
```

This example deliberately removes missing training labels. The switch declares
that scoring without `churn` skips that target-reading filter; the age filter
still runs. Preview checks this decision even for `--action train`. Keep the
generated scoring mode for this example. A custom scoring implementation is a
separate option when you need a different eligibility contract.

Run these from the generated project root using its configured Python environment:

```powershell
python src/tools/smoke.py
python src/tools/preview.py --action train
python src/tools/preview.py --list-preprocessors
```

Smoke validates declarations without calling custom factories; preview resolves
selected recipes and executes trusted factories, but does not train a model.
Training fills income from the **training partition's** group means, creates
`income_x_tenure`, scales numeric columns using saved training statistics and
replaces `segment` with learned one-hot columns. The resulting values depend on
which rows the split admits; preview cannot predict fitted means. On prediction,
an unseen group uses the saved global income fallback and an unseen segment
produces all-zero one-hot columns with this configuration.

The switch becomes part of the packaged scoring contract. See the phase rules
below before changing it for another dataset.

### Pre-split admission and scoring

The generated training workflow admits exactly these built-in pre-split types:
`DropMissingRows`, `ManualBounds`, `Deduplicate`, `ValueReplacement`,
`TextCleaning`, `AliasReplacement`, `InvalidValueReplacement`, and `Casting`.
It also admits captured project filters with the exact fixed-filter contract.
Even a mathematically fixed scaler or a constant imputer is not automatically
on this allowlist. Learned imputers, encoders, selectors and statistics belong
after splitting and inside each training fold.

Pre-split steps preserve the source columns and survivor identities. Filters
only remove rows and preserve survivor values, dtypes and order. Fixed edits
declare their write columns. They cannot write record keys, source event/result
times, or other protected group/weight metadata. Keys must be unique and
nonmissing before filtering. Deduplication cannot repair duplicate record keys;
it selects among distinct records with an explicitly named business subset and
rejects groups with conflicting targets. Training requires at least four eligible
rows and at least two rows in each final train/holdout partition.

Fixed edits run in declared order before splitting. Their feature-column subset
is projected into the saved model preprocessing; target-column edits have a
separate saved target contract. A column absent from a particular model's inputs
is not added by this projection. Scoring uses saved declarations, not current
editable YAML. In scoring mode `pre_split` (or the pre-split portion of
`combined`), fixed feature edits and eligible row filters are replayed according
to that contract. A target-reading filter cannot silently run without its input:
`skip_target_pre_split_steps: true` in scoring.yml explicitly skips such filters, changing the
eligible scoring population. It does not fabricate labels. Do not rely on
deduplication across separate scoring requests: its comparison population is the
supplied batch. Ordinary Core prediction skips DropMissingRows, Deduplicate, RowFilterFunction
and resampling. Other row-changing operations are guarded against silently
losing prediction rows; scoring eligibility is a separate workflow.

## Read the step cards

### Select columns from the training schema

In `config/preprocessing.yml`, a step that accepts `columns` can also select
columns by name pattern, dtype, or exclusions:

```yaml
version: 1
recipes:
  default:
    - name: scale_measurements
      transformer: StandardScaler
      params:
        columns_regex: '^measure_'
        dtype: numeric
        exclude_columns: [measure_id]
```

The pipeline resolves these rules once against the training features and saves
the resulting column names. Prediction reuses those names; a new column in a
request does not change the fitted selection. The same options work in a
`SkyulfPipeline` preprocessing step or `FeatureEngineer` configuration.

| Option | Meaning |
| --- | --- |
| `columns_regex` | Python regular expression searched within each column name. Use `^` to match the beginning. |
| `dtype` | One of `numeric`, `categorical`, `datetime`, or `boolean`. Numeric includes real numeric and Decimal columns, but excludes booleans, durations, and complex values. |
| `exclude_columns` | List of columns to remove from the selection. |
| `columns` with rules | A nonempty list limits candidates and determines their order. Without one, candidates follow the training frame's order, excluding the target. |

Regex matching uses `re.search`: `amt` selects both `amt` and `max_amt`.
Use `^amt$` to select only the complete name. `dtype: numeric` can also select
integer IDs and counts; use `exclude_columns: [customer_id]` or an explicit
`columns` list to limit the operation to intended measurements.

Name and dtype rules are combined with AND. Exclusions apply before dtype
inspection. An empty result raises an error instead of invoking the node's
automatic selection. An explicit empty `columns` list combined with these rules
starts from all feature columns; without rules, each node keeps its existing
empty-list behavior. Schema preview returns an unknown schema when object values
must be inspected to distinguish Decimal numbers from categories.

These are ordinary preprocessing rules, not `pre_split.yml` options. Native
Spark training currently requires explicit column names. Spark worker inference
from an admitted pandas/Polars-trained model uses the saved resolved selection.
The Canvas continues to use its explicit column picker. Direct Calculator calls
do not resolve pipeline rules automatically; pass explicit `columns` there.

### Read configuration and output contracts

All YAML blocks on the family pages are **list fragments**, to paste under a selected recipe, unless
explicitly marked as standalone Core configuration. Each card names the file to edit. Pre-split
admission is narrower than ordinary preprocessing; training-only steps and diagnostics are marked
explicitly. All fields shown are configuration inputs, not the internal fitted artifact fields. Do
not copy learned `means`, encoders or history seeds into YAML. Changing configuration requires a new
fit and model version.

**Column selection:** use explicit existing names for reproducibility. For the
usual scaler/imputer/encoder `columns` parameter, a missing or `null`
selection auto-detects; explicit `[]` generally means no columns. Selectors,
GroupImputer and MissingIndicator instead auto-select for empty lists. Numeric auto-detection
can omit binary, constant or all-missing columns; an explicit numeric selection
avoids that heuristic. Exceptions are stated in their cards. `target_column`
(string, normally supplied by the pipeline) identifies the label; supervised
nodes can instead receive paired `y`. Avoid selecting the target as a feature.
There is no universal promise that a misspelled column raises: some nodes filter
missing names or become no-ops. Inspect the resolved recipe and output schema.

**Rows and numbers:** unless a card states otherwise, steps preserve row count,
order and paired target. Floating transforms use floating arithmetic, so integer
identifiers or Decimal currency requiring exact arithmetic should not be scaled,
hashed as floats or expanded by numerical features. Null and NaN handling is
node-specific; an all-null training column does not contain a learnable mean.
An empty recipe is identity; an empty fit dataset is not generally a valid model
training request. Empty apply support does not imply empty fit support.

### Runtime support applied to every card

| Runtime | What these cards establish |
| --- | --- |
| Local pandas | Implemented native fit/apply path for the listed Core owners; optional packages and card restrictions still apply. |
| Local Polars | Implemented fit/apply interface for the same owners; sklearn/text/custom callbacks may use a bounded pandas/NumPy conversion. This is not distributed execution. |
| Native Spark FeatureEngineer | Only `SimpleImputer` and `StandardScaler` here declare native Spark fit/apply, under restricted contracts (numeric means/scaling or compatible fixed fills). Other cards do not claim native Spark support. |
| Certified Spark worker inference | Requires a supported complete fitted pipeline, validated state/recipe and worker preflight. Local row context alone is insufficient; the concrete artifact is the unit of admission. No per-card blanket certification is implied. |
| REST / serving endpoint | Requires the serving workflow's admitted complete model, input contract and dependencies. Registry presence, successful local replay, or Spark worker support does not grant REST admission. |

Context in the cards describes apply-time dependence: `row` means saved state plus
one input row, `window` means ordered observations, `global` means the whole
supplied batch matters. Fit always needs its appropriate training population.
Use [preprocessing context checks](preprocessing_context.md) for artifact-specific
inspection and [Spark execution](spark.md) for distributed boundaries. The
examples do not certify every parameter/engine/serving combination.

## Step families

Each family page has one card per step: purpose, file, parameter table, minimal
YAML, a common mistake and saved-state behavior. The
[registry index](#registry-index-and-scope) below links every registered ID to
its card.

| Family | Steps | Page |
| --- | --- | --- |
| <span id="missing-values-fixed-cleaning-and-types"></span><span id="fixed-normalization-followed-by-eligibility"></span>Missing values, cleaning and types | Imputers, missing indicators, row and column drops, deduplication, text cleaning, replacements, casting | [Open page](preprocessing_recipes_cleaning.md) |
| <span id="scaling-and-numeric-transformations"></span>Scaling and numeric transformations | Standard, min-max, max-abs and robust scalers; simple, general and power transformations | [Open page](preprocessing_recipes_scaling.md) |
| <span id="bounds-and-outliers"></span>Bounds and outliers | Manual bounds, clipping, IQR, z-score, winsorizing, elliptic envelope | [Open page](preprocessing_recipes_outliers.md) |
| <span id="categorical-encoding"></span>Categorical encoding | One-hot, dummy, ordinal, label, hash, target and WOE encoders | [Open page](preprocessing_recipes_encoding.md) |
| <span id="binning"></span>Binning | General, custom and k-bins discretizers | [Open page](preprocessing_recipes_binning.md) |
| <span id="feature-generation-and-selection"></span>Feature generation and selection | Interactions, polynomial and formula features, feature selection | [Open page](preprocessing_recipes_features.md) |
| <span id="time-and-geography"></span>Time and geography | Date features, lags, rolling aggregates, geo distance, H3 cells | [Open page](preprocessing_recipes_time_geo.md) |
| <span id="text-and-embeddings"></span>Text and embeddings | Tokenizer, count, TF-IDF and hashing vectorizers, sentence embeddings | [Open page](preprocessing_recipes_text.md) |
| <span id="split-sampling-and-diagnostic-core-nodes"></span>Split, sampling and diagnostic Core nodes | Split nodes, oversampling, undersampling, diagnostics | [Open page](preprocessing_recipes_sampling.md) |
| <span id="custom-factory-contracts-and-ready-functions"></span>Custom factory contracts and ready functions | Column, fitted and row-filter functions; shipped ready factories; assets | [Open page](preprocessing_recipes_custom.md) |

## Registry index and scope

This index covers 73 registered IDs and 69 distinct Calculator/Applier owners.
Four additional names are aliases, not new algorithms. Follow each link for
parameters and phase restrictions; presence here does not grant pre-split/Spark/REST
admission.

| Registered ID | Canonical card |
| --- | --- |
| <span id="aliasreplacement"></span>AliasReplacement | [AliasReplacement](preprocessing_recipes_cleaning.md#aliasreplacement) |
| <span id="casting"></span>Casting | [Casting](preprocessing_recipes_cleaning.md#casting) |
| <span id="clipvalues"></span>ClipValues | [ClipValues](preprocessing_recipes_outliers.md#clipvalues) |
| <span id="columnfunction"></span>ColumnFunction | [ColumnFunction: custom factory only](preprocessing_recipes_custom.md#columnfunction) |
| <span id="correlationthreshold"></span>CorrelationThreshold | [CorrelationThreshold](preprocessing_recipes_features.md#correlationthreshold) |
| <span id="custombinning"></span>CustomBinning | [CustomBinning](preprocessing_recipes_binning.md#custombinning) |
| <span id="datasnapshot"></span>DataSnapshot | [DataSnapshot: diagnostic](preprocessing_recipes_sampling.md#datasnapshot) |
| <span id="datasetprofile"></span>DatasetProfile | [DatasetProfile: diagnostic](preprocessing_recipes_sampling.md#datasetprofile) |
| <span id="datefeatures"></span>DateFeatures | [DateFeatures](preprocessing_recipes_time_geo.md#datefeatures) |
| <span id="deduplicate"></span>Deduplicate | [Deduplicate](preprocessing_recipes_cleaning.md#deduplicate) |
| <span id="dropmissingcolumns"></span>DropMissingColumns | [DropMissingColumns](preprocessing_recipes_cleaning.md#dropmissingcolumns) |
| <span id="dropmissingrows"></span>DropMissingRows | [DropMissingRows](preprocessing_recipes_cleaning.md#dropmissingrows) |
| <span id="dummyencoder"></span>DummyEncoder | [DummyEncoder](preprocessing_recipes_encoding.md#dummyencoder) |
| <span id="ellipticenvelope"></span>EllipticEnvelope | [EllipticEnvelope: training filter](preprocessing_recipes_outliers.md#ellipticenvelope) |
| <span id="featuregeneration"></span>FeatureGeneration | [FeatureGeneration](preprocessing_recipes_features.md#featuregeneration) |
| FeatureGenerationNode | [Alias of FeatureGeneration](preprocessing_recipes_features.md#featuregeneration) |
| <span id="featureinteraction"></span>FeatureInteraction | [FeatureInteraction](preprocessing_recipes_features.md#featureinteraction) |
| FeatureMath | [Alias of FeatureGeneration](preprocessing_recipes_features.md#featuregeneration) |
| <span id="fittedfunction"></span>FittedFunction | [FittedFunction: custom factory only](preprocessing_recipes_custom.md#fittedfunction) |
| <span id="generalbinning"></span>GeneralBinning | [GeneralBinning](preprocessing_recipes_binning.md#generalbinning) |
| <span id="generaltransformation"></span>GeneralTransformation | [GeneralTransformation](preprocessing_recipes_scaling.md#generaltransformation) |
| <span id="geodistance"></span>GeoDistance | [GeoDistance](preprocessing_recipes_time_geo.md#geodistance) |
| <span id="groupimputer"></span>GroupImputer | [GroupImputer](preprocessing_recipes_cleaning.md#groupimputer) |
| FrequencyEncoder | [FrequencyEncoder](preprocessing_recipes_encoding.md#frequencyencoder) |
| RareCategoryGrouping | [RareCategoryGrouping](preprocessing_recipes_encoding.md#rarecategorygrouping) |
| QuantileTransformer | [QuantileTransformer](preprocessing_recipes_scaling.md#quantiletransformer) |
| Normalizer | [Normalizer](preprocessing_recipes_scaling.md#normalizer) |
| <span id="h3index"></span>H3Index | [H3Index](preprocessing_recipes_time_geo.md#h3index) |
| <span id="hashencoder"></span>HashEncoder | [HashEncoder](preprocessing_recipes_encoding.md#hashencoder) |
| <span id="isolationforest"></span>IsolationForest | [IsolationForest](preprocessing_recipes_outliers.md#isolationforest) |
| <span id="localoutlierfactor"></span>LocalOutlierFactor | [LocalOutlierFactor](preprocessing_recipes_outliers.md#localoutlierfactor) |
| <span id="iqr"></span>IQR | [IQR: training filter](preprocessing_recipes_outliers.md#iqr) |
| <span id="invalidvaluereplacement"></span>InvalidValueReplacement | [InvalidValueReplacement](preprocessing_recipes_cleaning.md#invalidvaluereplacement) |
| <span id="iterativeimputer"></span>IterativeImputer | [IterativeImputer](preprocessing_recipes_cleaning.md#iterativeimputer) |
| <span id="kbinsdiscretizer"></span>KBinsDiscretizer | [KBinsDiscretizer](preprocessing_recipes_binning.md#kbinsdiscretizer) |
| <span id="knnimputer"></span>KNNImputer | [KNNImputer](preprocessing_recipes_cleaning.md#knnimputer) |
| <span id="labelencoder"></span>LabelEncoder | [LabelEncoder](preprocessing_recipes_encoding.md#labelencoder) |
| <span id="lagfeatures"></span>LagFeatures | [LagFeatures](preprocessing_recipes_time_geo.md#lagfeatures) |
| <span id="manualbounds"></span>ManualBounds | [ManualBounds](preprocessing_recipes_outliers.md#manualbounds) |
| <span id="maxabsscaler"></span>MaxAbsScaler | [MaxAbsScaler](preprocessing_recipes_scaling.md#maxabsscaler) |
| <span id="minmaxscaler"></span>MinMaxScaler | [MinMaxScaler](preprocessing_recipes_scaling.md#minmaxscaler) |
| <span id="missingindicator"></span>MissingIndicator | [MissingIndicator](preprocessing_recipes_cleaning.md#missingindicator) |
| <span id="modelbasedselection"></span>ModelBasedSelection | [ModelBasedSelection](preprocessing_recipes_features.md#modelbasedselection) |
| <span id="onehotencoder"></span>OneHotEncoder | [OneHotEncoder](preprocessing_recipes_encoding.md#onehotencoder) |
| <span id="ordinalencoder"></span>OrdinalEncoder | [OrdinalEncoder](preprocessing_recipes_encoding.md#ordinalencoder) |
| <span id="oversampling"></span>Oversampling | [Oversampling: training only](preprocessing_recipes_sampling.md#oversampling) |
| <span id="polynomialfeatures"></span>PolynomialFeatures | [PolynomialFeatures](preprocessing_recipes_features.md#polynomialfeatures) |
| PolynomialFeaturesNode | [Alias of PolynomialFeatures](preprocessing_recipes_features.md#polynomialfeatures) |
| <span id="powertransformer"></span>PowerTransformer | [PowerTransformer](preprocessing_recipes_scaling.md#powertransformer) |
| <span id="robustscaler"></span>RobustScaler | [RobustScaler](preprocessing_recipes_scaling.md#robustscaler) |
| <span id="rollingaggregate"></span>RollingAggregate | [RollingAggregate](preprocessing_recipes_time_geo.md#rollingaggregate) |
| <span id="rowfilterfunction"></span>RowFilterFunction | [RowFilterFunction: custom filter only](preprocessing_recipes_custom.md#rowfilterfunction) |
| <span id="simpleimputer"></span>SimpleImputer | [SimpleImputer](preprocessing_recipes_cleaning.md#simpleimputer) |
| <span id="simpletransformation"></span>SimpleTransformation | [SimpleTransformation](preprocessing_recipes_scaling.md#simpletransformation) |
| Split | [Deprecated alias of TrainTestSplitter](preprocessing_recipes_sampling.md#traintestsplitter) |
| <span id="standardscaler"></span>StandardScaler | [StandardScaler](preprocessing_recipes_scaling.md#standardscaler) |
| <span id="targetencoder"></span>TargetEncoder | [TargetEncoder](preprocessing_recipes_encoding.md#targetencoder) |
| <span id="textcleaning"></span>TextCleaning | [TextCleaning](preprocessing_recipes_cleaning.md#textcleaning) |
| <span id="traintestsplitter"></span>TrainTestSplitter | [TrainTestSplitter: standalone split](preprocessing_recipes_sampling.md#traintestsplitter) |
| <span id="undersampling"></span>Undersampling | [Undersampling: training only](preprocessing_recipes_sampling.md#undersampling) |
| <span id="univariateselection"></span>UnivariateSelection | [UnivariateSelection](preprocessing_recipes_features.md#univariateselection) |
| <span id="valuereplacement"></span>ValueReplacement | [ValueReplacement](preprocessing_recipes_cleaning.md#valuereplacement) |
| <span id="variancethreshold"></span>VarianceThreshold | [VarianceThreshold](preprocessing_recipes_features.md#variancethreshold) |
| <span id="woeencoder"></span>WOEEncoder | [WOEEncoder](preprocessing_recipes_encoding.md#woeencoder) |
| <span id="winsorize"></span>Winsorize | [Winsorize](preprocessing_recipes_outliers.md#winsorize) |
| <span id="zscore"></span>ZScore | [ZScore: training filter](preprocessing_recipes_outliers.md#zscore) |
| <span id="count_vectorizer"></span>count_vectorizer | [count_vectorizer](preprocessing_recipes_text.md#count_vectorizer) |
| <span id="feature_selection"></span>feature_selection | [feature_selection: dispatcher](preprocessing_recipes_features.md#feature_selection) |
| <span id="feature_target_split"></span>feature_target_split | [feature_target_split: standalone target routing](preprocessing_recipes_sampling.md#feature_target_split) |
| <span id="hashing_vectorizer"></span>hashing_vectorizer | [hashing_vectorizer](preprocessing_recipes_text.md#hashing_vectorizer) |
| <span id="sentence_embedder"></span>sentence_embedder | [sentence_embedder](preprocessing_recipes_text.md#sentence_embedder) |
| <span id="tfidf_vectorizer"></span>tfidf_vectorizer | [tfidf_vectorizer](preprocessing_recipes_text.md#tfidf_vectorizer) |
| <span id="tokenizer"></span>tokenizer | [tokenizer](preprocessing_recipes_text.md#tokenizer) |

For implementation-level contracts, see the generated
[preprocessing API reference](../reference/api/preprocessing/index.md).
Use the API version matching the installed package; output naming and legacy
artifact compatibility can differ across versions. New YAML edits affect only
new training runs. Saved model state, captured Python/assets and selected
declarations define existing inference behavior.
