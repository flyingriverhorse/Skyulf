/** User-facing placement guidance; execution decisions remain in the leakage classifier. */
export type PreprocessingPlacement = 'before' | 'conditional' | 'after' | 'time' | 'split';

export interface PreprocessingPlacementEntry {
  id: string;
  name: string;
  category: string;
  placement: PreprocessingPlacement;
  rule: string;
}

export const placementLabels: Record<PreprocessingPlacement, string> = {
  before: 'Before split (fixed)',
  conditional: 'Depends on operation',
  after: 'After split',
  time: 'Time/history review',
  split: 'Creates row split',
};

const featureGenerationRule =
  'Arithmetic, ratio, similarity and datetime extraction are fixed operations. ' +
  'group_agg learns a training group lookup and must run after the split; unknown groups become missing. ' +
  'A mixed operation list is learned if any operation learns.';
const polynomialRule =
  'Builds a polynomial basis from the input schema and configured degree/options. ' +
  'No feature-value distribution is fitted. Only use inputs available at prediction time.';
const polynomialSelectionRule =
  'With auto_detect enabled and columns omitted or [], learns which columns are eligible from the data; ' +
  'place after the row split. An explicit nonempty column list, or auto_detect disabled/omitted, ' +
  'keeps selection fixed. Polynomial math itself fits no feature-value statistics.';

const rows: ReadonlyArray<readonly [string, string, string, PreprocessingPlacement, string]> = [
  ['AliasReplacement', 'Alias Replacement', 'Cleaning', 'before',
    'Applies configured or domain alias mappings to individual values. No training vocabulary is fitted.'],
  ['Casting', 'Type Casting', 'Cleaning', 'conditional',
    'Numeric, string and datetime conversions are fixed. category/categorical casts learn a training vocabulary; put them after the split. Resolve per-column overrides before deciding. Unseen categories become missing.'],
  ['CorrelationThreshold', 'Correlation Threshold', 'Feature selection', 'after',
    'Learns retained features from training correlations and replays that selection. An empty selection still invokes automatic selection.'],
  ['CustomBinning', 'Custom Binning', 'Binning', 'conditional',
    'Configured edges with explicit columns, including [], are fixed. Omitted/null columns trigger value-based discovery, so that mode belongs after the split.'],
  ['DataSnapshot', 'Data Snapshot', 'Reporting', 'before',
    'Reports a snapshot while preserving modeling data. Viewing a report is not feature fitting, but choosing model changes from held-out reports can contaminate evaluation.'],
  ['DatasetProfile', 'Dataset Profile', 'Reporting', 'before',
    'Reports dataset statistics while preserving modeling data. Keep held-out reports out of feature and model selection decisions.'],
  ['DateFeatures', 'Date Features', 'Feature engineering', 'before',
    'Extracts configured calendar features from individual values without fitting a training distribution. The source timestamp must be available at prediction time.'],
  ['Deduplicate', 'Deduplicate', 'Row operations', 'after',
    'Filters across rows and is conservatively gated because duplicate relationships affect retained data. Separately prevent duplicate entities or records from crossing evaluation splits; placement alone does not establish independence.'],
  ['DropMissingColumns', 'Drop Missing Columns', 'Missing values', 'conditional',
    'A positive missing_threshold learns which columns to drop, even with explicit candidate columns: put it after the split. Otherwise the configured deletion is fixed and can precede the split.'],
  ['DropMissingRows', 'Drop Missing Rows', 'Missing values', 'before',
    'Applies configured missingness rules to each row without learning a statistic. Dropping evaluation rows changes the measured population; row-changing branches can also break merge alignment.'],
  ['DummyEncoder', 'Dummy Encoder', 'Encoding', 'after',
    'Learns training categories and fixes the indicator schema. Unknown categories produce zeros; drop_first removes the first training category.'],
  ['EllipticEnvelope', 'Elliptic Envelope', 'Outliers', 'after',
    'Fits an outlier estimator on training rows and reuses it. Filtering evaluation rows changes the population being scored; do not silently treat the filtered score as a full-population result.'],
  ['FeatureGeneration', 'Feature Generation', 'Feature engineering', 'conditional', featureGenerationRule],
  ['FeatureGenerationNode', 'Feature Generation (alias)', 'Feature engineering', 'conditional', featureGenerationRule],
  ['FeatureInteraction', 'Feature Interaction', 'Feature engineering', 'before', polynomialRule],
  ['FeatureMath', 'Feature Math (alias)', 'Feature engineering', 'conditional', featureGenerationRule],
  ['GeneralBinning', 'General Binning', 'Binning', 'after',
    'Fits data-derived bin edges or discretization state on training values. Reuse the same edges for validation, test and inference.'],
  ['GeneralTransformation', 'General Transformation', 'Transformations', 'conditional',
    'Fixed math such as logarithm, sqrt and square can precede the split. Yeo-Johnson and Box-Cox learn parameters and belong after the split. Inspect every entry in a mixed transformation list; one learned method makes the node learned.'],
  ['GeoDistance', 'Geo Distance', 'Geospatial', 'before',
    'Calculates configured distances from each row\'s coordinates. It does not fit the current dataset; coordinates must already be known when predicting.'],
  ['H3Index', 'H3 Index', 'Geospatial', 'before',
    'Maps coordinates to a cell at a configured resolution without fitting the dataset. Optional dependency availability is separate from leakage placement.'],
  ['HashEncoder', 'Hash Encoder', 'Encoding', 'conditional',
    'Hashing into configured buckets is fixed with explicit columns, including []. Automatic column selection is conservatively gated and belongs after the split.'],
  ['IQR', 'IQR Outliers', 'Outliers', 'after',
    'Learns training quartiles and bounds, then applies those stored limits. Held-out outliers must not move the bounds.'],
  ['InvalidValueReplacement', 'Invalid Value Replacement', 'Cleaning', 'before',
    'Replaces values using configured invalid-value rules, without learning a distribution or category vocabulary.'],
  ['IterativeImputer', 'Iterative Imputer', 'Missing values', 'after',
    'Fits imputation models on training data and reuses those models for validation, test and inference. Refit inside each cross-validation training fold.'],
  ['KBinsDiscretizer', 'K-Bins Discretizer', 'Binning', 'after',
    'A registered binning alias whose active data-derived modes fit edges or discretization state on training values. Replay the fitted state on held-out data.'],
  ['KNNImputer', 'KNN Imputer', 'Missing values', 'after',
    'Fits or stores the training reference rows used to impute missing values. Held-out rows must not become fitting references.'],
  ['LabelEncoder', 'Label Encoder', 'Encoding', 'conditional',
    'Omitted/null/empty columns encode the available target only, or do nothing. Explicit feature columns learn category mappings and belong after the split. An exact known-target selection is target-only, not a feature-statistics fit.'],
  ['LagFeatures', 'Lag Features', 'Time series', 'time',
    'Uses positive lags within configured ordering and groups. Review entity boundaries, time order, prediction horizon and available history. No fitted distribution does not mean no leakage; a random split or unavailable past labels can invalidate the experiment.'],
  ['ManualBounds', 'Manual Bounds', 'Outliers', 'before',
    'Uses user-supplied bounds without estimating batch limits. Bounds chosen by inspecting held-out outcomes still contaminate evaluation.'],
  ['MaxAbsScaler', 'Max-Abs Scaler', 'Scaling', 'after',
    'Learns maximum absolute values from training rows and reuses that scale on all held-out rows.'],
  ['MinMaxScaler', 'Min-Max Scaler', 'Scaling', 'after',
    'Learns training minima and maxima. Test extrema must not redefine the scaling range.'],
  ['MissingIndicator', 'Missing Indicator', 'Missing values', 'conditional',
    'A nonempty explicit column list fixes the indicator schema and can precede the split. Omitted/null/[] discovers which columns contain missing values, so those modes belong after the split.'],
  ['ModelBasedSelection', 'Model-Based Selection', 'Feature selection', 'after',
    'Fits a selector/model using training features and, when required, training targets. Replay the selected feature set and refit selection inside each CV training fold.'],
  ['OneHotEncoder', 'One-Hot Encoder', 'Encoding', 'after',
    'Learns categories and optional frequency grouping from training rows. Keep the fitted output schema; unknown categories follow the configured ignore/error policy.'],
  ['OrdinalEncoder', 'Ordinal Encoder', 'Encoding', 'conditional',
    'Omitted/null columns auto-detect and learn feature categories. Explicit [] is target-only/no-op; exact known-target selection is also exempt. Explicit feature columns remain conservatively learned, including configured category orders.'],
  ['Oversampling', 'Oversampling', 'Resampling', 'after',
    'Resample training rows only, including only the training partition of each CV fold. Never oversample validation or test rows; their original population is the evaluation target.'],
  ['PolynomialFeatures', 'Polynomial Features', 'Feature engineering', 'conditional', polynomialSelectionRule],
  ['PolynomialFeaturesNode', 'Polynomial Features (alias)', 'Feature engineering', 'conditional', polynomialSelectionRule],
  ['PowerTransformer', 'Power Transformer', 'Transformations', 'after',
    'Fits Yeo-Johnson or Box-Cox parameters using training values. Disabling standardization does not make the fitted power parameter fixed.'],
  ['RobustScaler', 'Robust Scaler', 'Scaling', 'after',
    'Learns robust location and scale statistics on training values, then reuses them on held-out data.'],
  ['RollingAggregate', 'Rolling Aggregate', 'Time series', 'time',
    'Computes ordered/grouped rolling features and includes the current row. Using the current target as an input can reveal the answer directly. Review time order, groups, horizon, split strategy and which observations are available for each prediction.'],
  ['SimpleImputer', 'Simple Imputer', 'Missing values', 'conditional',
    'Constant filling is fixed when its value is configured. Mean, median and most_frequent learn training statistics and belong after the split and inside each CV training fold.'],
  ['SimpleTransformation', 'Simple Transformation', 'Transformations', 'before',
    'Applies a configured simple mathematical function to each value. It does not fit power-transform parameters; source features must be available at prediction time.'],
  ['Split', 'Train-Test Split (legacy alias)', 'Boundaries', 'split',
    'Deprecated alias of TrainTestSplitter. It establishes the same row boundary; retain this ID when reading older saved pipelines.'],
  ['StandardScaler', 'Standard Scaler', 'Scaling', 'after',
    'Learns training centering and scaling statistics. Reuse fitted parameters on held-out data and refit them inside each CV training fold.'],
  ['TargetEncoder', 'Target Encoder', 'Encoding', 'after',
    'Learns target statistics. Pipeline training uses out-of-fold encodings; held-out/inference rows use the full-training artifact, with the training prior for unknown categories. Keep the outer split and CV refit boundaries intact.'],
  ['TextCleaning', 'Text Cleaning', 'Text', 'before',
    'Trim, case, special-character and regex operations are fixed per value. Omitted/null columns discover text dtypes; explicit [] does nothing.'],
  ['TrainTestSplitter', 'Train-Test Split', 'Boundaries', 'split',
    'Creates the train/test row boundary. Choose a split strategy suited to the data. Separating X from y with Feature-Target Split is not an equivalent evaluation boundary.'],
  ['Undersampling', 'Undersampling', 'Resampling', 'after',
    'Resample training rows only, including each CV training partition. Never undersample validation or test rows; preserve their evaluation population.'],
  ['UnivariateSelection', 'Univariate Selection', 'Feature selection', 'after',
    'Fits feature scores and selection on training data, including training targets when the method requires them. Replay the selected columns.'],
  ['ValueReplacement', 'Value Replacement', 'Cleaning', 'before',
    'Uses explicitly configured value replacements. The replacement mapping is not learned from the current batch.'],
  ['VarianceThreshold', 'Variance Threshold', 'Feature selection', 'after',
    'Learns feature variances and retained columns from training rows. An empty configured selection does not establish a no-op exemption.'],
  ['WOEEncoder', 'Weight of Evidence Encoder', 'Encoding', 'after',
    'Learns binary-target log-odds. Training rows receive complement-fold mappings; held-out rows use the full-training artifact. Unknown categories use the configured fallback, currently zero.'],
  ['Winsorize', 'Winsorize', 'Outliers', 'after',
    'Learns clipping limits from training rows. Held-out extreme values must not change the fitted limits.'],
  ['ZScore', 'Z-Score Outliers', 'Outliers', 'after',
    'Learns training location and scale and applies the stored outlier rule to held-out rows.'],
  ['count_vectorizer', 'Count Vectorizer', 'Text', 'conditional',
    'An explicit nonempty feature-column list learns the training vocabulary and belongs after the split. Omitted/null/[] is a no-op, not automatic text discovery. Authoritative target-only columns are excluded from feature processing.'],
  ['feature_selection', 'Feature Selection (dispatcher)', 'Feature selection', 'after',
    'Dispatches to the configured feature selector. Active selection learns training scores, correlations, variances or model state; fit it after the split and inside CV training folds. An empty selection is not a universal no-op.'],
  ['feature_target_split', 'Feature-Target Split', 'Boundaries', 'before',
    'Separates feature columns X from target y without separating training rows from test rows. It does not satisfy the train/test leakage boundary; add Train-Test Split before learned preprocessing.'],
  ['hashing_vectorizer', 'Hashing Vectorizer', 'Text', 'before',
    'Uses a fixed hash space without learning a vocabulary. Requires explicit nonempty text columns; omitted/null/[] does nothing.'],
  ['sentence_embedder', 'Sentence Embedder', 'Text', 'before',
    'Applies a pretrained embedding model without fitting the current dataset. Requires explicit nonempty columns; omitted/null/[] does nothing. The guard cannot establish the pretrained model\'s data provenance.'],
  ['tfidf_vectorizer', 'TF-IDF Vectorizer', 'Text', 'conditional',
    'An explicit nonempty feature-column list learns the training vocabulary and IDF and belongs after the split. Omitted/null/[] is a no-op. Authoritative target-only columns are excluded from feature processing.'],
  ['tokenizer', 'Tokenizer', 'Text', 'before',
    'Applies configured tokenization rules without fitting a training distribution. Requires explicit nonempty columns; omitted/null/[] does nothing.'],
];

/** Includes aliases because every serialized preprocessing registration is searchable. */
export const preprocessingPlacementCatalog: readonly PreprocessingPlacementEntry[] = rows.map(
  ([id, name, category, placement, rule]) => ({ id, name, category, placement, rule }),
);
