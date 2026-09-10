import { StepType as BackendStepType } from '../../constants/stepTypes';
import type { NodeConverter } from './types';

const convertDatasetNode: NodeConverter = (node) => {
  return {
    stepType: BackendStepType.DATA_LOADER,
    params: {
      dataset_id: node.data.datasetId,
    }
  };
};

const convertImputationNode: NodeConverter = (node) => {
  let stepType = 'unknown';
  let params: Record<string, unknown> = {};
  const method = node.data.method || 'simple';
  if (method === 'knn') {
    stepType = 'KNNImputer';
    params = {
      columns: node.data.columns,
      n_neighbors: node.data.n_neighbors,
      weights: node.data.weights
    };
  } else if (method === 'iterative') {
    stepType = 'IterativeImputer';
    params = {
      columns: node.data.columns,
      max_iter: node.data.max_iter,
      estimator: node.data.estimator,
      // Legacy graphs may lack the field entirely — omit it
      // instead of hardcoding a fallback so core's documented
      // default (IterativeImputerCalculator) is the single owner.
      ...(node.data.random_state != null
        ? { random_state: node.data.random_state }
        : {}),
    };
  } else {
    stepType = 'SimpleImputer';
    params = {
      columns: node.data.columns,
      strategy: node.data.strategy,
      fill_value: node.data.fill_value
    };
  }
  return { stepType, params };
};

const convertSimpleImputer: NodeConverter = (node) => {
  return {
    stepType: 'SimpleImputer',
    params: node.data || {}
  };
};

const convertDropMissingColumns: NodeConverter = (node) => {
  return {
    stepType: 'DropMissingColumns',
    params: {
      columns: node.data.columns || [],
      missing_threshold: node.data.missing_threshold
    }
  };
};

const convertDropMissingRows: NodeConverter = (node) => {
  const stepType = 'DropMissingRows';
  // UI semantics: "drop rows missing MORE than X%". A 0% threshold
  // (or the checkbox) means "any missing" -> how="any".
  const dropRowsData = node.data as { drop_if_any_missing?: boolean; missing_threshold?: number };
  const dropAny = dropRowsData.drop_if_any_missing === true
    || dropRowsData.missing_threshold == null
    || dropRowsData.missing_threshold <= 0;
  const params = dropAny
    ? { how: 'any' }
    : { missing_threshold: dropRowsData.missing_threshold };
  return { stepType, params };
};

const convertDeduplicate: NodeConverter = (node) => {
  return {
    stepType: 'Deduplicate',
    params: {
      subset: node.data.subset,
      keep: node.data.keep
    }
  };
};

const convertCasting: NodeConverter = (node) => {
  return {
    stepType: 'Casting',
    params: {
      column_types: node.data.column_types
    }
  };
};

const convertMissingIndicator: NodeConverter = (node) => {
  return {
    stepType: 'MissingIndicator',
    params: {
      columns: node.data.columns,
      flag_suffix: node.data.flag_suffix
    }
  };
};

const scalerTypes = new Map([
  ['minmax', 'MinMaxScaler'], ['maxabs', 'MaxAbsScaler'], ['robust', 'RobustScaler'],
]);

/** Translate scalar UI bounds into the backend's tuple parameters. */
function scalerRanges(method: unknown, config: Record<string, unknown>): Record<string, unknown> {
  if (method === 'minmax') {
    return { feature_range: [config.feature_range_min ?? 0, config.feature_range_max ?? 1] };
  }
  if (method === 'robust') {
    return { quantile_range: [config.quantile_range_min ?? 25, config.quantile_range_max ?? 75] };
  }
  return {};
}

const convertScaleNumericFeatures: NodeConverter = (node) => {
  const config = node.data && typeof node.data === 'object' ? node.data : {};
  const method = config.method || 'standard';
  return {
    stepType: scalerTypes.get(method as string) ?? 'StandardScaler',
    params: { ...config, ...scalerRanges(method, config) },
  };
};

const convertEncoding: NodeConverter = (node) => {
  let stepType = 'unknown';
  let params: Record<string, unknown> = {};
  const method = node.data.method;
  if (method === 'onehot') stepType = 'OneHotEncoder';
  else if (method === 'dummy') stepType = 'DummyEncoder';
  else if (method === 'label') stepType = 'LabelEncoder';
  else if (method === 'ordinal') stepType = 'OrdinalEncoder';
  else if (method === 'target') stepType = 'TargetEncoder';
  else if (method === 'hash') stepType = 'HashEncoder';
  else if (method === 'woe') stepType = 'WOEEncoder';
  else stepType = 'OneHotEncoder'; // Default

  params = node.data;
  return { stepType, params };
};

const convertTrainTestSplitter: NodeConverter = (node) => {
  return {
    stepType: 'TrainTestSplitter',
    params: node.data || {}
  };
};

const convertLabelEncoding: NodeConverter = (node) => {
  return {
    stepType: 'LabelEncoder',
    params: node.data || {}
  };
};

const convertFeatureTargetSplit: NodeConverter = (node) => {
  return {
    stepType: 'feature_target_split',
    params: node.data || {}
  };
};

const convertFeatureSelection: NodeConverter = (node) => {
  return {
    stepType: 'feature_selection',
    params: node.data || {}
  };
};

const convertOutlier: NodeConverter = (node) => {
  let stepType = 'unknown';
  let params: Record<string, unknown> = {};
  const method = node.data.method || 'iqr';
  if (method === 'iqr') stepType = 'IQR';
  else if (method === 'zscore') stepType = 'ZScore';
  else if (method === 'winsorize') stepType = 'Winsorize';
  else if (method === 'elliptic_envelope') stepType = 'EllipticEnvelope';
  else stepType = 'IQR';
  params = node.data;
  return { stepType, params };
};

const convertTransformationNode: NodeConverter = (node) => {
  const stepType = 'GeneralTransformation';

  // Flatten transformations: { columns: ['a', 'b'], method: 'log' } -> [{ column: 'a', method: 'log' }, { column: 'b', method: 'log' }]
  const rawTransformations = (node.data.transformations || []) as unknown[];
  const flattenedTransformations = [];

  for (const r of rawTransformations) {
    const rule = r as Record<string, unknown>;
    if (rule.columns && Array.isArray(rule.columns)) {
      for (const col of (rule.columns as string[])) {
        flattenedTransformations.push({
          column: col,
          method: rule.method,
          ...(rule.params as Record<string, unknown>)
        });
      }
    }
  }

  const params = { transformations: flattenedTransformations };
  return { stepType, params };
};

const convertBinningNode: NodeConverter = (node) => {
  return {
    stepType: 'GeneralBinning',
    params: {
      columns: node.data.columns,
      strategy: node.data.strategy,
      n_bins: node.data.n_bins,
      label_format: node.data.label_format,
      output_suffix: node.data.output_suffix,
      drop_original: node.data.drop_original,
      custom_bins: node.data.custom_bins, // For custom strategy
      custom_labels: node.data.custom_labels, // For custom strategy
      precision: node.data.precision
    }
  };
};

const convertResamplingNode: NodeConverter = (node) => {
  let stepType = 'unknown';
  let params: Record<string, unknown> = {};
  const type = node.data.type || 'oversampling';
  if (type === 'oversampling') {
    stepType = 'Oversampling';
  } else {
    stepType = 'Undersampling';
  }
  params = node.data;
  return { stepType, params };
};

const convertFeatureGenerationNode: NodeConverter = (node) => {
  return {
    stepType: 'FeatureMath',
    params: {
      operations: node.data.operations
    }
  };
};

const convertPolynomialFeaturesNode: NodeConverter = (node) => {
  return {
    stepType: 'PolynomialFeatures',
    params: {
      columns: node.data.columns,
      degree: node.data.degree,
      interaction_only: node.data.interaction_only,
      include_bias: node.data.include_bias,
      output_prefix: node.data.output_prefix,
      include_input_features: node.data.include_input_features
    }
  };
};

const convertFeatureInteractionNode: NodeConverter = (node) => {
  return {
    stepType: 'FeatureInteraction',
    params: {
      columns: node.data.columns,
      degree: node.data.degree,
      interaction_only: node.data.interaction_only,
      include_bias: node.data.include_bias
    }
  };
};

const convertTimeSeriesNode: NodeConverter = (node) => {
  let stepType = 'unknown';
  let params: Record<string, unknown> = {};
  const method = node.data.method;
  if (method === 'rolling') stepType = 'RollingAggregate';
  else if (method === 'date') stepType = 'DateFeatures';
  else stepType = 'LagFeatures';
  params = node.data;
  return { stepType, params };
};

const convertTextCleaning: NodeConverter = (node) => {
  return {
    stepType: 'TextCleaning',
    params: node.data
  };
};

const convertTextNode: NodeConverter = (node) => {
  return {
    stepType: node.data.definitionType as string,
    params: node.data
  };
};

const convertValueReplacement: NodeConverter = (node) => {
  return {
    stepType: 'ValueReplacement',
    params: node.data
  };
};

const convertAliasReplacement: NodeConverter = (node) => {
  return {
    stepType: 'AliasReplacement',
    params: node.data
  };
};

const convertInvalidValueReplacement: NodeConverter = (node) => {
  return {
    stepType: 'InvalidValueReplacement',
    params: node.data
  };
};

const convertDataPreview: NodeConverter = () => {
  return {
    stepType: 'data_preview',
    params: {}
  };
};

/** Retain generic parameters for unrecognized saved node types. */
export const convertUnknownNode: NodeConverter = (node) => {
  console.warn(`Unknown node type: ${node.data.definitionType}`);
  return {
    stepType: 'Unknown',
    params: node.data && typeof node.data === 'object' ? node.data : {},
  };
};

export const preprocessingConverters = new Map<string, NodeConverter>([
  ['dataset_node', convertDatasetNode],
  ['imputation_node', convertImputationNode],
  ['simple_imputer', convertSimpleImputer],
  ['drop_missing_columns', convertDropMissingColumns],
  ['DropMissingColumns', convertDropMissingColumns],
  ['drop_missing_rows', convertDropMissingRows],
  ['DropMissingRows', convertDropMissingRows],
  ['deduplicate', convertDeduplicate],
  ['Deduplicate', convertDeduplicate],
  ['casting', convertCasting],
  ['Casting', convertCasting],
  ['MissingIndicator', convertMissingIndicator],
  ['missing_indicator', convertMissingIndicator],
  ['scale_numeric_features', convertScaleNumericFeatures],
  ['encoding', convertEncoding],
  ['TrainTestSplitter', convertTrainTestSplitter],
  ['label_encoding', convertLabelEncoding],
  ['feature_target_split', convertFeatureTargetSplit],
  ['feature_selection', convertFeatureSelection],
  ['outlier', convertOutlier],
  ['TransformationNode', convertTransformationNode],
  ['BinningNode', convertBinningNode],
  ['ResamplingNode', convertResamplingNode],
  ['FeatureGenerationNode', convertFeatureGenerationNode],
  ['PolynomialFeaturesNode', convertPolynomialFeaturesNode],
  ['FeatureInteractionNode', convertFeatureInteractionNode],
  ['TimeSeriesNode', convertTimeSeriesNode],
  ['TextCleaning', convertTextCleaning],
  ['count_vectorizer', convertTextNode],
  ['tfidf_vectorizer', convertTextNode],
  ['hashing_vectorizer', convertTextNode],
  ['tokenizer', convertTextNode],
  ['sentence_embedder', convertTextNode],
  ['ValueReplacement', convertValueReplacement],
  ['value_replacement', convertValueReplacement],
  ['AliasReplacement', convertAliasReplacement],
  ['InvalidValueReplacement', convertInvalidValueReplacement],
  ['data_preview', convertDataPreview],
]);
