# Preprocessing leakage audit

This audit covers **62 registered preprocessing types: 60 transformations and
the two train/test splitter registrations, `TrainTestSplitter` and `Split`**.
Aliases have separate rows because saved pipelines can contain either spelling.
The four stacking/voting ensemble registrations are models and are outside this
inventory.

The audit distinguishes learning from training values, applying fixed rules,
and operations that depend on other rows. It covers the implementations and
parameter modes listed here; it is not a guarantee about every possible input,
custom operation, or the provenance of a dataset.

## What the leakage check means

Registry `learns_from_data` metadata is a conservative description of a node's
capabilities. The configured operation is authoritative: for example,
`GeneralTransformation` can learn a power-transform parameter or apply a fixed
logarithm. `skyulf.leakage.step_learns_from_data` resolves these differences for
the core and backend. The frontend mirrors the parameter rules for immediate
feedback and receives registry flags from the backend.

For evaluation, fit learned preprocessing after establishing the train/test
boundary. Learn its state from training rows and reuse that state for validation,
test, and inference. During cross-validation and tuning, fit preprocessing again
inside each training fold. A transformer fitted once on the whole outer training
set is not automatically safe for its inner validation folds.

`learns_from_data=False` is not a guarantee that an operation is safe in every
experiment. Lagged features, rolling windows, target-derived inputs, and data
prepared outside the pipeline require additional reasoning about availability,
time, and provenance.

## Registered-node inventory

The **Metadata** column records the registry flag. **Behavior and placement**
describes the actual fit/apply contract. "Learned" means that an active operation
needs training-only fitting. The explicit-empty exceptions described after the
table still apply.

| # | Registered type | Metadata | Behavior and placement |
| --- | --- | --- | --- |
| 1 | `AliasReplacement` | False | Applies configured/domain alias mappings to individual values. Fixed conversion; no training vocabulary is fitted. |
| 2 | `Casting` | True | `category`/`categorical` casts freeze training vocabularies; unseen values become missing. Other casts are fixed conversions. Resolve per-column overrides before classifying the mode. |
| 3 | `CorrelationThreshold` | True | Learns which features to retain from training correlations. Replays the selected columns; an empty selection still invokes automatic selection. |
| 4 | `CustomBinning` | True | Bin edges are configured. With an explicit column list, including `[]`, the operation is fixed. Omitted/null columns invoke value-based column discovery and remain learned. |
| 5 | `DataSnapshot` | False | Reports a snapshot/statistics while preserving the modeling data. Reporting is not fitting a model feature. |
| 6 | `DatasetProfile` | False | Reports dataset statistics while preserving the modeling data. Using held-out reports to make modeling decisions can still contaminate an evaluation. |
| 7 | `DateFeatures` | False | Extracts configured calendar features from each value. No training distribution is learned. |
| 8 | `Deduplicate` | True | Filters across rows. Conservatively gated because duplicate relationships affect the retained dataset, even without a numerical fitted statistic. |
| 9 | `DropMissingColumns` | True | A positive `missing_threshold` learns which columns to drop, including when candidate columns are explicit. Otherwise the drop list is configured and fixed. |
| 10 | `DropMissingRows` | False | Applies configured missingness rules to each row. It does not estimate a training statistic; removing evaluation rows still changes the evaluated population. |
| 11 | `DummyEncoder` | True | Learns training categories and emits that fixed indicator schema. Unknown categories give zeros; `drop_first` drops the training vocabulary's first category. |
| 12 | `EllipticEnvelope` | True | Fits the training outlier estimator and reuses it. Finite rows are filtered even when another row is invalid; invalid rows are preserved rather than disabling filtering for the batch. |
| 13 | `FeatureGeneration` | True | `group_agg` learns training group tables and replays them. Arithmetic, ratio, similarity, and datetime extraction use fixed rules. Mixed lists are learned if any operation is learned. |
| 14 | `FeatureGenerationNode` | True | Alias of feature generation, with the same group-aggregation versus fixed-operation distinction. |
| 15 | `FeatureInteraction` | False | Builds a polynomial interaction basis from configured degree/options and input schema. The basis is not a statistic of the feature values. |
| 16 | `FeatureMath` | True | Alias of feature generation. Group aggregation learns state; arithmetic, ratio, similarity, and datetime extraction are fixed. |
| 17 | `GeneralBinning` | True | Learns data-derived bin edges or discretization state from training values and reuses them. Changing the held-out distribution must not refit edges. |
| 18 | `GeneralTransformation` | True | Yeo-Johnson and Box-Cox learn parameters. The eight fixed math method names listed below are row-local. Every rule in a mixed list matters. |
| 19 | `GeoDistance` | False | Computes configured distances from each row's coordinates. No training distribution is fitted. |
| 20 | `H3Index` | False | Maps each coordinate pair to an H3 cell using configured resolution. Optional dependency availability does not make the operation learned. |
| 21 | `HashEncoder` | True | Hashes values into configured buckets. Explicit columns, including `[]`, are exempt. Automatic selection is conservatively gated, although its categorical detector inspects dtypes rather than category counts. |
| 22 | `IQR` | True | Learns training quartiles/bounds and applies the stored bounds. Held-out outliers cannot move them. |
| 23 | `InvalidValueReplacement` | False | Replaces values according to configured invalid-value rules. No distribution or category vocabulary is fitted. |
| 24 | `IterativeImputer` | True | Fits imputation models on training data and reuses those models for other splits. |
| 25 | `KBinsDiscretizer` | True | Registered binning alias. Its active data-derived binning modes learn training edges/state and replay them. |
| 26 | `KNNImputer` | True | Stores/fits the training reference data used to impute missing features. Held-out rows must not become fitting references. |
| 27 | `LabelEncoder` | True | Omitted/null/empty columns encode only the available target or do nothing. Explicit feature columns learn category mappings. An exact known-target selection is target-only. |
| 28 | `LagFeatures` | False | Uses positive lags within the configured ordering/groups. Depends on other rows even though no fitted distribution is stored; requires a valid temporal/history protocol. |
| 29 | `ManualBounds` | False | Applies user-supplied bounds. Limits are not estimated from the current batch. |
| 30 | `MaxAbsScaler` | True | Learns training maximum absolute values and applies the stored scale. |
| 31 | `MinMaxScaler` | True | Learns training extrema and applies the stored scaling parameters. Held-out extrema cannot redefine the scale. |
| 32 | `MissingIndicator` | True | Omitted/null/`[]` discovers which columns contain missing values. A nonempty explicit list fixes the indicator columns and is exempt. |
| 33 | `ModelBasedSelection` | True | Fits a selector/model on training features and, where required, training targets. Replays the selected feature set. |
| 34 | `OneHotEncoder` | True | Learns categories and optional frequency-based grouping. Replay retains the training schema; unknowns follow the configured ignore/error policy. |
| 35 | `OrdinalEncoder` | True | Omitted/null columns auto-detect and learn feature categories. Explicit `[]` is target-only/no-op; an exact known-target selection is also exempt. Explicit feature selections remain conservatively learned, including configured category orders. |
| 36 | `Oversampling` | True | Learns/changes the training sample distribution. Resampling belongs only on training data, never validation or test rows. |
| 37 | `PolynomialFeatures` | True | Automatic selection learns eligible columns when `auto_detect` is enabled and columns are omitted/empty. Explicit nonempty columns or disabled/omitted `auto_detect` remain fixed; the polynomial basis itself fits no feature-value statistics. |
| 38 | `PolynomialFeaturesNode` | True | Polynomial-feature alias with the same operation-dependent automatic-selection rule and fixed explicit-column mode. |
| 39 | `PowerTransformer` | True | Fits Yeo-Johnson or Box-Cox parameters on training values. Disabling standardization does not make the fitted power parameter fixed. |
| 40 | `RobustScaler` | True | Learns training robust location/scale statistics and reuses them. |
| 41 | `RollingAggregate` | False | Computes ordered/grouped rolling features and includes the current row. Sequence-dependent; a current-target input can leak the answer directly. |
| 42 | `SimpleImputer` | True | Mean, median, and most-frequent strategies learn training statistics. Constant filling uses a configured value and is exempt. |
| 43 | `SimpleTransformation` | False | Applies the configured simple mathematical function to individual values. No fitted power-transform parameter is learned. |
| 44 | `Split` | False | Deprecated alias of `TrainTestSplitter`. Establishes the same train/test boundary. |
| 45 | `StandardScaler` | True | Learns training centering/scaling statistics and reuses them for held-out rows. |
| 46 | `TargetEncoder` | True | Learns target statistics. Pipeline training receives out-of-fold encodings; held-out/inference rows use the full-training artifact. Unknown categories use the training prior. |
| 47 | `TextCleaning` | False | Trim, case, special-character, and regex operations act on individual values. Omitted/null columns use text-dtype discovery; explicit `[]` does nothing. |
| 48 | `TrainTestSplitter` | False | Creates the train/test boundary. Splitting features from the target alone is not an equivalent boundary. |
| 49 | `Undersampling` | True | Learns/changes the training sample distribution. Validation and test populations must remain outside resampling. |
| 50 | `UnivariateSelection` | True | Fits feature scores/selection on training data and training targets when the method needs them. Replays the selected columns. |
| 51 | `ValueReplacement` | False | Applies explicitly configured value replacements. It does not learn the replacement mapping from the batch. |
| 52 | `VarianceThreshold` | True | Learns feature variances and the retained feature set from training rows. An empty configured list does not establish a no-op exemption. |
| 53 | `WOEEncoder` | True | Learns binary-target log-odds. Training rows receive complement-fold mappings; held-out rows use the full-training artifact. Unknown categories receive the configured fallback, currently zero. |
| 54 | `Winsorize` | True | Learns training clipping limits and reuses them. Held-out extreme values cannot change fitted limits. |
| 55 | `ZScore` | True | Learns training location/scale and applies the stored outlier rule. |
| 56 | `count_vectorizer` | True | A nonempty explicit feature-column list fits vocabulary and document-frequency restrictions. Omitted/null/empty or target-only selections do nothing. Apply only transforms with the training vocabulary. |
| 57 | `feature_selection` | True | Generic feature-selection registration. Its active selector learns training-dependent state; an empty list is not a blanket no-op exemption. |
| 58 | `feature_target_split` | False | Separates `X` from `y`. Does not create independent training and evaluation row sets. |
| 59 | `hashing_vectorizer` | False | Requires explicit text columns and constructs configured token hashes with rowwise normalization. No vocabulary is learned; omitted/null/empty selections do nothing. |
| 60 | `sentence_embedder` | False | Requires explicit text columns and applies pretrained weights without fitting the corpus. Empty/target-only selections avoid model loading. The optional model boundary is tested with a deterministic substitute. |
| 61 | `tfidf_vectorizer` | True | A nonempty explicit feature-column list fits vocabulary and IDF. Omitted/null/empty or target-only selections do nothing. Held-out text reuses training vocabulary and weights. |
| 62 | `tokenizer` | False | Requires explicit text columns. Word/character analyzers and token counts are rowwise; omitted/null/empty selections do nothing. |

## Parameter distinctions that change placement

### Fixed math and fitted power transforms

`GeneralTransformation` stores its rules under `transformations`. The fixed
method names are `log`, `sqrt`, `square_root`, `cube_root`, `reciprocal`, `square`,
`exp`, and `exponential`. A Yeo-Johnson or Box-Cox rule makes the operation
learned, even if another rule is fixed. Empty rule lists do nothing. Unknown
methods do not receive the fixed-operation exemption.

Feature-generation registrations use `operations`. `arithmetic`, `ratio`,
`similarity`, and `datetime_extract` are fixed; an omitted `operation_type`
defaults to arithmetic. `group_agg` fits a group table on training rows and
replays it. An unknown operation does not receive a fixed-operation exemption.

### Empty selections are node-specific

An explicit `columns: []` is a no-op for `OneHotEncoder`, `DummyEncoder`,
`TargetEncoder`, `WOEEncoder`, `PowerTransformer`, `StandardScaler`,
`MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `SimpleImputer`, `KNNImputer`,
`IterativeImputer`, `GeneralBinning`, `KBinsDiscretizer`, `CustomBinning`, `IQR`,
`ZScore`, `Winsorize`, and `EllipticEnvelope`.

This rule must not be generalized to `MissingIndicator`, feature selectors,
or resampling. Their empty selections do not mean the same thing. Label and
ordinal encoders can still transform the target when feature columns are empty.
An empty column selection is also different from a selected column containing
zero rows or no valid text tokens; those inputs can still fail a fit.

All five text feature nodes require nonempty **explicit** columns. They do not
auto-select text features when `columns` is omitted. The resolver removes both
the configured target column and the name of a supplied target series. If only
target columns remain, the node does nothing; otherwise it fits/transforms the
remaining feature columns. `drop_original` does not authorize removing the
target or creating derived target features.

### Casting and column discovery

Casting first reads `column_types`, then overrides the named entries when a
nonempty `columns` list and `target_type` are supplied. Check the effective map,
not only the initial configuration: an override can introduce or remove the
last categorical cast. Empty `columns` does not cancel a categorical entry in
`column_types`. Datetime conversion is performed with row-local mixed-format
handling so another row does not choose the interpretation of a value.

Fixed bin edges do not make automatic column discovery fixed.
`CustomBinning` with omitted/null columns still examines values to select
columns. Conversely, `HashEncoder` uses dtype-based categorical discovery;
its automatic mode remains conservatively classified as learned. Hashing
itself does not fit a vocabulary, and collecting unique values during apply
is an optimization rather than learning new buckets.

## Pipeline responsibilities

The **core library** owns the operation classification and fitted-state
semantics. Its leakage checks reject active learned preprocessing before the
evaluation boundary. Calculators fit training state; appliers reuse it.
Target/WOE training hooks produce cross-fitted training representations, and
tuning must rebuild learned preprocessing within each fold. Calling a
supervised encoder's plain `fit` and then `apply` on the same training rows is
not a substitute for its training hook.

The **backend** applies the shared core classification to submitted graphs and
validates the execution scope before starting work. It propagates known target
context so text feature nodes can remove target columns before fitting, handles
split-aware training execution, and returns actionable leakage errors. A
client-side check does not replace this server-side enforcement.
A per-node `target_column` hint cannot grant a leakage exemption; the
pipeline/branch target context is authoritative.

The **frontend** combines registry flags with the mirrored parameter rules.
It can block a bad connection before a job is submitted. Per-model actions
check the selected model's upstream graph; a separate unsafe branch must not
block a safe selected model. Failed server submissions retain the detailed
reason so the user can correct the pipeline after a toast disappears.

## Limits and artifact compatibility

- These checks cannot infer an unnamed target, detect every target-derived
  feature, or reconstruct how an uploaded dataset was prepared. Known target
  exclusion protects the named inputs, not all possible proxies for the answer.
- `LagFeatures` and `RollingAggregate` need an explicit history and validation
  protocol. Positive lag alone is not proof of correct data availability;
  rolling windows include the current row. Neither metadata nor this audit
  guarantees temporal ordering or external-data provenance.
- Reporting nodes preserve the modeling data, but repeatedly inspecting test
  profiles and changing modeling decisions still uses evaluation information.
- Deduplication and missing-row filtering alter which observations survive.
  The conservative deduplication guard does not establish entity isolation or
  decide the right evaluation population for a particular application.
- Legacy config-only group-aggregation artifacts require refitting. Serving
  must not rebuild the missing group statistics from an inference batch.
- Older categorical-casting artifacts without stored vocabularies cannot
  acquire a historical training vocabulary retrospectively. Refit them before
  relying on frozen-category behavior.
- New hash-encoding artifacts version their numeric normalization. They hash
  an integral float consistently with the same integral numeric value in a
  different batch dtype. Legacy artifacts retain historical conversion so an
  upgrade does not silently change a trained model's buckets. Refit to obtain
  the new batch-invariant behavior.
- Pretrained sentence-embedding weights are not fitted by this node. The audit
  tests its integration boundary and missing-dependency behavior with a
  deterministic substitute; it does not claim integration testing of a
  downloaded external model or all hardware-dependent numerical behavior.
- Category ordering, unseen values, optional dependencies, and invalid inputs
  are covered by focused regressions. That coverage is not an exhaustive proof
  for every parameter combination or third-party library version.

## Source and regression coverage

The paths below are relative to the repository root.

- `skyulf-core/skyulf/leakage.py`: authoritative operation classification.
- `skyulf-core/skyulf/preprocessing/`: actual calculators, appliers, and
  supervised training hooks audited above.
- `backend/ml_pipeline/_execution/_leakage_validation.py`: graph enforcement
  using the shared operation rules.
- `frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts`: canvas
  preflight, registry fallback, and parameter-specific exceptions.
- `skyulf-core/tests/test_cases/leakage/registry_nodes.json`: registry inventory
  contract. `operation_modes.json` in the same directory covers meaningful
  parameter-specific learned/fixed modes.
- `skyulf-core/tests/unit/test_cleaning_operation_leakage.py`: cleaning,
  imputation, scaling, binning, selection, and outlier semantics.
- `skyulf-core/tests/unit/test_encoding_operation_leakage.py`: category/text
  fitting and replay, unknown values, target exclusion, cross-fitting, batching,
  and optional embedding dependency behavior.
- `skyulf-core/tests/unit/test_feature_operation_leakage.py`: feature
  operations, fitted group state, and row/sequence behavior.
- `skyulf-core/tests/integration/test_leakage_fixture_contract.py` and
  `skyulf-core/tests/integration/test_leakage_operation_contract.py`: shared
  inventory and operation contracts exercised through core integration paths.
- `skyulf-core/tests/integration/test_core_pipeline_tuning_leakage.py`: actual
  preprocessing fits within tuning folds.
- `tests/integration/test_leakage_submission.py`,
  `tests/integration/test_leakage_operation_contract.py`, and
  `tests/integration/test_leakage_graph_semantics.py`: server submissions,
  parameter contracts, and graph execution semantics.
- `frontend/ml-canvas/src/core/utils/pipelineLeakageOperationModes.test.ts`,
  `frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.test.ts`, and
  `frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.test.tsx`: bundled
  and fetched registry modes, selected-model scope, and actionable submission
  feedback.

The regression strategy checks effects: changing held-out data must not refit
training state, changing held-out labels must not change features, and replay
must preserve the fitted schema and unknown-value policy. Registry flags alone
are insufficient evidence for these properties.

## Validation record for this audit

The final consolidated runs completed on 2026-09-08:

| Layer | Passing tests | Scope |
|---|---:|---|
| Core | 3,476 | Entire unit directory plus leakage JSON contracts, preprocessing integration, native tuning/CV refit, raw text target context, and repeated row boundaries |
| Backend | 1,608 | Admission and operation JSON contracts, real graph execution, fold refit/stress, gate units, and preprocessing/pipeline units |
| Frontend | 193 | Leakage preflight, operation modes, selected training context, and error-detail regressions |

These are 5,277 passing test cases across the consolidated suites, not a claim
that every possible input or graph has been exhausted. The 69 skipped core cases
are opt-in or disabled performance/benchmark cases, not failed leakage cases.
Dependency, alias, dataframe-conversion, and build-chunk warnings remain.

Changed-source Ruff and ty checks passed. Frontend lint and the production build
passed. Example 09's script ran successfully; its notebook ran all 12 code cells
in a real Jupyter kernel with all assertions passing. Six SVG diagrams, three
embedded notebook SVG attachments, and the executed notebook schema were also
validated. Mermaid sources accompany the SVGs; mkdocs was deliberately not run.

Backend submission tests replace external dispatch/storage services rather than
launch a production Celery/DB deployment. Pretrained embedding behavior uses a
model double. Legacy numeric HashEncoder artifacts preserve historical buckets;
refit to adopt the new versioned, batch-invariant numeric normalization.
