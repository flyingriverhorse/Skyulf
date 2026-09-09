# Preprocessing placement and leakage

Use this guide to decide where a preprocessing node belongs relative to the
**Train-Test Split** node. It covers **62 registered preprocessing IDs**:
60 transformations and two row-split registrations. Aliases have separate entries
because older saved pipelines can contain either spelling.

Placement depends on the configured operation, not just the node's display name.
The same node can apply a fixed formula in one configuration and learn a statistic
in another. These rules describe supported implementations and parameter modes;
they do not guarantee that an arbitrary dataset or experiment is leakage-free.

## Start with the training boundary

A practical linear pipeline is:

```text
Loader -> Fixed rules -> Train-Test Split -> Learned preprocessing -> Train / Tune
```

Learned preprocessing must fit using **training rows only**. Apply the resulting
state to validation, test and inference rows without fitting it again. Examples
include scaling statistics, imputation values, category vocabularies, feature
selection, bin edges, target statistics and group lookup tables.

Cross-validation introduces an inner boundary. Fit learned preprocessing again
inside each training fold, then apply it to that fold's validation rows. A scaler
fitted once on the whole outer training set has already seen the inner validation
folds. Putting it after the outer split alone does not resolve that problem.

Resampling is stricter: oversampling and undersampling act **only on training
rows**, including the training partition of each CV fold. Do not resample the
validation or test population.

**Feature-Target Split is not Train-Test Split.** Separating X from y does not
reserve any rows for evaluation. `TrainTestSplitter` and its legacy `Split` alias
create the row boundary checked by the leakage gate.

## Read the placement labels

| Label | Meaning | What to do |
| --- | --- | --- |
| Before split (fixed) | Uses configured rules, schema or a pretrained model without fitting the current dataset. | May precede the split when its inputs and configuration are valid for prediction. |
| Depends on operation | Some configurations are fixed; others learn from data. | Inspect every configured operation and selection rule. |
| After split | Active preprocessing learns state or is conservatively gated. | Fit on training rows and refit inside CV training folds. Resampling is training-only. |
| Time/history review | Uses neighboring or earlier rows without fitting a distribution. | Check ordering, entity groups, horizon, history availability and split strategy. |
| Creates row split | Establishes the train/test row boundary. | Place learned feature preprocessing downstream and choose an appropriate split protocol. |

"Before split" is permission to apply a fixed operation, not a universal safety
claim. A ratio that includes the answer, a timestamp recorded after an outcome,
or a manual threshold chosen from test results can leak even though the operation
fits no statistic. Reporting statistics is not feature fitting, but using test
reports to select features or tune a model contaminates the evaluation.

## Configuration changes that change placement

### General Transformation and Power Transformer

Fixed logarithms, square roots and squares do not fit a parameter from the dataset.
Yeo-Johnson and Box-Cox do. Put those learned methods after the row split and fit
them within each CV training fold. Turning off standardization does not make the
power parameter fixed.

Inspect the entire transformation list. If one entry learns, the mixed node learns.
The name "General Transformation" alone cannot establish its placement.

### Polynomial Features and its alias

`PolynomialFeatures` and `PolynomialFeaturesNode` build a fixed mathematical
basis, but their optional automatic column selection learns from the fitting rows.

| Configuration | Placement |
|---|---|
| `auto_detect: true`, with `columns` omitted or `[]` | After the row split: discover eligible columns from training data only. |
| Explicit nonempty `columns`, even with `auto_detect: true` | Before split is allowed: column selection and polynomial math are fixed. |
| `auto_detect` omitted or `false` | Before split is allowed: automatic discovery is disabled. |

The registry conservatively marks both aliases as learned; the configured
operation determines placement. `FeatureInteraction` is a separate fixed node
and does not inherit this automatic-selection rule.

### Feature Generation and its aliases

Arithmetic, ratios, similarity and datetime extraction are fixed calculations.
`group_agg` fits a group lookup on training rows and reuses it on held-out rows.
Unknown groups become missing instead of obtaining statistics from the test batch;
add suitable downstream missing-value handling when needed.

This distinction applies to `FeatureGeneration`, `FeatureGenerationNode` and
`FeatureMath`. A mixed operation list containing a learned group aggregate belongs
after the split, even when its other entries are arithmetic. Fixed expressions
still require inputs that exist at prediction time.

### Casting, imputation and column deletion

| Configuration | Placement reason |
| --- | --- |
| Casting numeric/string/datetime values | Fixed conversion. |
| Casting to `category`/`categorical` | Fits a training vocabulary; unseen categories become missing. Resolve per-column overrides first. |
| Simple Imputer with a configured constant | Fixed fill rule. |
| Simple Imputer with mean, median or most-frequent strategy | Fits training statistics. |
| Drop Missing Columns without a positive threshold | Fixed configured deletion. |
| Drop Missing Columns with positive `missing_threshold` | Learns which columns to remove, even with explicit candidate columns. |

### Column selection is part of the operation

Do not treat omitted columns, `null` and `[]` as interchangeable:

| Node | Omitted/null selection | Explicit empty selection | Explicit nonempty selection |
| --- | --- | --- | --- |
| Custom Binning | Value-based column discovery; learned. | Fixed/no-op. | Configured columns and edges are fixed. |
| Hash Encoder | Automatic selection; conservatively gated. | Fixed/no-op. | Fixed hashing of selected columns. |
| Missing Indicator | Discovers columns containing missing values; learned. | Also automatic discovery; learned. | Fixed indicator columns. |
| Label Encoder | Available target only, or no-op. | Available target only, or no-op. | Feature columns learn mappings; exact known-target selection is exempt. |
| Ordinal Encoder | Discovers and learns feature categories. | Target-only/no-op. | Feature encoding remains learned, including configured category orders; exact known-target selection is exempt. |
| Count and TF-IDF Vectorizers | No-op; no automatic text discovery. | No-op. | Feature text learns vocabulary and, for TF-IDF, IDF. |
| Hashing Vectorizer, Tokenizer, Sentence Embedder | No-op. | No-op. | Fixed processing or pretrained embeddings. |
| Text Cleaning | Discovers text dtypes for fixed cleaning. | No-op. | Fixed cleaning of selected text columns. |

Target-only exemptions require an authoritative target in the current pipeline or
branch. A target name from an unrelated branch is not evidence that an encoder is
operating only on labels. Text feature processing excludes the authoritative target.

An explicit `[]` is also a supported no-op for `OneHotEncoder`, `DummyEncoder`,
`TargetEncoder`, `WOEEncoder`, `PowerTransformer`, `StandardScaler`, `MinMaxScaler`,
`MaxAbsScaler`, `RobustScaler`, `SimpleImputer`, `KNNImputer`, `IterativeImputer`,
`GeneralBinning`, `KBinsDiscretizer`, `CustomBinning`, `IQR`, `ZScore`, `Winsorize`
and `EllipticEnvelope`. This is a specific implementation contract, not a rule to
apply to other nodes. Selectors, resamplers and Missing Indicator do not receive a
general empty-list exemption.

### Target encoders, lags and rolling windows

`TargetEncoder` learns target statistics. Training rows in the pipeline receive
out-of-fold encodings; held-out and inference rows use the full-training artifact.
Unknown categories fall back to the training prior. `WOEEncoder` uses complement-fold
mappings for training rows and the full-training mapping for held-out rows, with the
configured unknown-category fallback, currently zero. These protections do not remove
the need for the outer row boundary and CV preprocessing refits.

`LagFeatures` and `RollingAggregate` do not fit a distribution, but they depend on
other rows. Use the correct time ordering, entity groups, prediction horizon and
history protocol. A rolling aggregate includes the current row: a current-target
input can expose the answer directly. A lagged target is useful only when that
target observation is actually available at the intended prediction time. Merely
moving a temporal node after a random split does not establish a valid protocol.

## Complete registered-node catalog

The placement of learned entries below describes active feature processing. The
explicit-empty and target-only exceptions above still apply. Aliases remain separate.

| Registered ID | Placement | Behavior and practical rule |
| --- | --- | --- |
| `AliasReplacement` | Before split (fixed) | Apply configured/domain aliases to each value; no fitted vocabulary. |
| `Casting` | Depends on operation | Fixed casts may precede split; categorical vocabularies fit on train. Resolve per-column overrides. |
| `CorrelationThreshold` | After split | Learn retained columns from training correlations; empty selection still invokes discovery. |
| `CustomBinning` | Depends on operation | Explicit columns and configured edges are fixed; omitted/null columns learn value-based selection. |
| `DataSnapshot` | Before split (fixed) | Report a snapshot without changing modeling data; do not tune from held-out reports. |
| `DatasetProfile` | Before split (fixed) | Report statistics while preserving data; held-out-driven modeling decisions can still leak. |
| `DateFeatures` | Before split (fixed) | Extract configured calendar fields from each timestamp. |
| `Deduplicate` | After split | Conservatively gated cross-row filtering. Separately prevent duplicate entities from crossing evaluation splits. |
| `DropMissingColumns` | Depends on operation | Positive missing threshold learns deletion; otherwise use fixed configured columns. |
| `DropMissingRows` | Before split (fixed) | Apply fixed per-row missingness rules. Removing evaluation rows changes the scored population. |
| `DummyEncoder` | After split | Learn training categories and fixed indicator schema; unknown categories yield zeros. |
| `EllipticEnvelope` | After split | Fit training outlier estimator and reuse it. Filtering held-out rows changes evaluation coverage. |
| `FeatureGeneration` | Depends on operation | Fixed arithmetic/ratio/similarity/datetime; `group_agg` learns training lookup. |
| `FeatureGenerationNode` | Depends on operation | Feature-generation alias with the same mixed-operation rules. |
| `FeatureInteraction` | Before split (fixed) | Build a polynomial basis from schema and configured options. |
| `FeatureMath` | Depends on operation | Feature-generation alias; any learned group aggregate makes the node learned. |
| `GeneralBinning` | After split | Learn bin edges or discretization state on training values and replay them. |
| `GeneralTransformation` | Depends on operation | Fixed math may precede split; Yeo-Johnson/Box-Cox fit parameters. Inspect all rules. |
| `GeoDistance` | Before split (fixed) | Compute configured distances from each row's coordinates. |
| `H3Index` | Before split (fixed) | Map coordinates to cells at a configured resolution. |
| `HashEncoder` | Depends on operation | Explicit columns use fixed hashing; automatic selection is conservatively gated. |
| `IQR` | After split | Learn training quartiles and bounds; held-out outliers cannot move them. |
| `InvalidValueReplacement` | Before split (fixed) | Apply configured invalid-value replacement rules. |
| `IterativeImputer` | After split | Fit imputation models on training data and reuse them. |
| `KBinsDiscretizer` | After split | Binning alias; active data-derived modes learn training edges/state. |
| `KNNImputer` | After split | Fit/store training reference rows; do not use held-out rows as fitting references. |
| `LabelEncoder` | Depends on operation | Default/empty target-only or no-op; explicit feature columns learn category mappings. |
| `LagFeatures` | Time/history review | Positive lags require valid ordering, groups, available history and prediction horizon. |
| `ManualBounds` | Before split (fixed) | Apply supplied bounds; do not choose them by inspecting test outcomes. |
| `MaxAbsScaler` | After split | Fit training maximum absolute values and reuse the scale. |
| `MinMaxScaler` | After split | Fit training extrema; held-out extrema cannot redefine the range. |
| `MissingIndicator` | Depends on operation | Nonempty explicit columns are fixed; omitted/null/[] learns missing-column discovery. |
| `ModelBasedSelection` | After split | Fit selector/model on training features and targets as required. |
| `OneHotEncoder` | After split | Fit categories/frequency grouping and preserve the training output schema. |
| `OrdinalEncoder` | Depends on operation | Default learns feature categories; explicit [] or exact known-target selection is exempt. |
| `Oversampling` | After split, training only | Resample each training partition only, never validation or test rows. |
| `PolynomialFeatures` | Depends on operation | `auto_detect: true` with omitted/empty columns learns column selection; otherwise fixed polynomial math. |
| `PolynomialFeaturesNode` | Depends on operation | Same automatic-selection rule as `PolynomialFeatures`; explicit nonempty columns remain fixed. |
| `PowerTransformer` | After split | Fit Yeo-Johnson/Box-Cox parameters even when standardization is disabled. |
| `RobustScaler` | After split | Fit robust training location and scale statistics. |
| `RollingAggregate` | Time/history review | Ordered/grouped rolling features include the current row; current-target input can leak directly. |
| `SimpleImputer` | Depends on operation | Configured constant is fixed; mean/median/most-frequent fits training statistics. |
| `SimpleTransformation` | Before split (fixed) | Apply configured simple math without fitting power parameters. |
| `Split` | Creates row split | Legacy alias of `TrainTestSplitter`. |
| `StandardScaler` | After split | Fit training centering/scaling statistics and refit within CV folds. |
| `TargetEncoder` | After split | OOF training encoding; full-training artifact for held-out/inference, training prior for unknowns. |
| `TextCleaning` | Before split (fixed) | Fixed per-value cleaning; omitted/null detects text dtypes, [] does nothing. |
| `TrainTestSplitter` | Creates row split | Create the actual train/test row boundary with a suitable split protocol. |
| `Undersampling` | After split, training only | Preserve the original validation and test populations. |
| `UnivariateSelection` | After split | Fit feature scores/selection on training rows and targets as required. |
| `ValueReplacement` | Before split (fixed) | Apply explicitly configured value replacements. |
| `VarianceThreshold` | After split | Fit feature variances and selection; [] is not a general no-op exemption. |
| `WOEEncoder` | After split | Complement-fold training mappings; full-training held-out mapping and configured unknown fallback. |
| `Winsorize` | After split | Fit training clipping limits; held-out extreme values cannot change them. |
| `ZScore` | After split | Fit training location/scale and reuse the stored outlier rule. |
| `count_vectorizer` | Depends on operation | Nonempty feature selection learns vocabulary; omitted/null/[] is no-op. |
| `feature_selection` | After split | Dispatch to configured learned selector; preserve training-only and fold-refit boundaries. |
| `feature_target_split` | Before split (fixed) | Separate X from y; this is not a train/test row boundary. |
| `hashing_vectorizer` | Before split (fixed) | Fixed hash space; explicit nonempty text columns required. |
| `sentence_embedder` | Before split (fixed) | Apply pretrained embeddings; explicit nonempty columns required, external model provenance remains separate. |
| `tfidf_vectorizer` | Depends on operation | Nonempty feature selection learns vocabulary/IDF; omitted/null/[] is no-op. |
| `tokenizer` | Before split (fixed) | Fixed tokenization; explicit nonempty text columns required. |

## Supported graph shapes and current policy

Prefer a linear chain when the feature operations are sequential. Backend fold
reconstruction also supports these branch forms when their requirements hold:

- A common row splitter forks into linear transformer branches that preserve row
  alignment and join directly into the model. Avoid additional splitters, nested
  merges and disallowed row-changing steps in the branches.
- Genuinely fixed branches from a shared loader merge at the first row splitter,
  followed by a reconstructable downstream chain. The resolver retains merged raw
  input and outer training row selection for fold-local fitting.

Supported topology does not establish correct feature ownership. Preserve intended
column selection and merge order; overlapping outputs can replace earlier branch
columns. A join into an intermediate encoder is not the same supported form as a
direct join into the model. See the [core/backend guide](leakage_core_backend.md)
for diagrams and exact reconstruction requirements.

With the default `raise` policy, a detected definite preprocessing-placement
violation is rejected by the backend admission gate with HTTP 400 and no jobs
created. Unsupported fold reconstruction involving learned preprocessing also
stops execution by default, but that capability failure can occur after a job has
been created. These are separate outcomes.

Explicit `metadata.on_leakage="warn"` or `"ignore"` can permit the legacy fallback
to pre-transformed data. Its CV scores can be optimistically biased, and its
`fold_refit_fallback` diagnostic describes the missing guarantee. Choosing a weaker
policy does not make the graph reconstructable or repair the statistical problem.

Read the **Leakage Gate** and **Fold Refit Audit** separately. Green results provide
evidence for the checks performed, not proof of feature provenance, entity
independence, valid temporal assumptions or an untouched test-selection process.

Core-only CV callers must fit a fresh pipeline on each fold's raw training
partition; cross-validating an estimator over a globally preprocessed matrix does
not automatically refit the core preprocessing steps.

## Where to use this reference

The canvas **How pipelines work** dialog contains a searchable
**Preprocessing & Leakage** tab with the same registered-node coverage. Search by
display name, registry ID, category or operation, and filter by placement.

For implementation evidence and the exact fit/apply contracts, see the
[preprocessing leakage audit](preprocessing_leakage_audit.md). For graph diagrams,
admission and fold-reconstruction details, see
[Leakage: core and backend](leakage_core_backend.md).
