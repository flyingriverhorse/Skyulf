import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Node } from 'react-flow-renderer';
import {
  FeatureMathNodeSignal,
  FeatureNodeParameter,
  OutlierMethodName,
  OutlierNodeSignal,
  TransformerAuditNodeSignal,
  PipelinePreviewSchema,
  SkewnessColumnDistribution,
  ScalingMethodName,
  PolynomialFeaturesNodeSignal,
  FeatureSelectionNodeSignal,
  triggerFullDatasetExecution,
  type BinningColumnRecommendation,
  type BinningExcludedColumn,
} from '../api';
import {
  DropMissingColumnsSection,
  useDropMissingColumns,
} from './node-settings/nodes/drop_col_rows/DropMissingColumnsSection';
import { DropMissingRowsSection } from './node-settings/nodes/drop_col_rows/DropMissingRowsSection';
import { RemoveDuplicatesSection } from './node-settings/nodes/remove_duplicates/RemoveDuplicatesSection';
import { DEFAULT_KEEP_STRATEGY, type KeepStrategy } from './node-settings/nodes/remove_duplicates/removeDuplicatesSettings';
import { CastColumnTypesSection } from './node-settings/nodes/cast_column/CastColumnTypesSection';
import { DataSnapshotSection } from './node-settings/nodes/dataset/DataSnapshotSection';
import { DatasetProfileSection, useDatasetProfileController } from './node-settings/nodes/dataset/datasetProfile';
import { NodeSettingsHeader } from './node-settings/layout/NodeSettingsHeader';
import { NodeSettingsFooter } from './node-settings/layout/NodeSettingsFooter';
import { ensureArrayOfString } from './node-settings/sharedUtils';
import {
  formatMetricValue,
  formatMissingPercentage,
  formatNumericStat,
  formatModeStat,
  getPriorityClass,
  getPriorityLabel,
} from './node-settings/formatting';
import {
  BINNED_SAMPLE_PRESETS,
  normalizeBinningConfigValue,
  type BinnedDistributionBin,
  type BinnedDistributionCard,
  type BinnedSamplePresetValue,
  type BinningLabelFormat,
  type BinningMissingStrategy,
  type BinningStrategy,
  type KBinsEncode,
  type KBinsStrategy,
} from './node-settings/nodes/binning/binningSettings';
import { BinNumericColumnsSection } from './node-settings/nodes/binning/BinNumericColumnsSection';
import { BinningInsightsSection } from './node-settings/nodes/binning/BinningInsightsSection';
import { BinnedDistributionSection } from './node-settings/nodes/binning/BinnedDistributionSection';
import {
  IMPUTATION_METHOD_OPTIONS,
  buildDefaultOptionsForMethod,
  isNumericImputationMethod,
  normalizeImputationStrategies,
  sanitizeOptionsForMethod,
  serializeImputationStrategies,
  type ImputationMethodOption,
  type ImputationStrategyConfig,
  type ImputationStrategyMethod,
  type ImputationStrategyOptions,
} from './node-settings/nodes/imputation/imputationSettings';
import { ImputationStrategiesSection, type ImputationSchemaDiagnostics } from './node-settings/nodes/imputation/ImputationStrategiesSection';
import { SCALING_METHOD_ORDER, normalizeScalingConfigValue } from './node-settings/nodes/scaling/scalingSettings';
import { ScalingInsightsSection } from './node-settings/nodes/scaling/ScalingInsightsSection';
import { SkewnessInsightsSection } from './node-settings/nodes/skewness/SkewnessInsightsSection';
import {
  dedupeSkewnessTransformations,
  normalizeSkewnessTransformations,
  type SkewnessTransformationConfig,
} from './node-settings/nodes/skewness/skewnessSettings';
import {
  resolveMissingIndicatorSuffix,
  buildMissingIndicatorInsights,
  type MissingIndicatorInsights,
} from './node-settings/nodes/missing_indicator/missingIndicatorSettings';
import { MissingIndicatorInsightsSection } from './node-settings/nodes/missing_indicator/MissingIndicatorInsightsSection';
import { ReplaceAliasesSection } from './node-settings/nodes/replace_aliases/ReplaceAliasesSection';
import {
  ALIAS_MODE_OPTIONS,
  DEFAULT_ALIAS_MODE,
  normalizeAliasStrategies,
  resolveAliasMode,
  serializeAliasStrategies,
  type AliasMode,
  type AliasStrategyConfig,
} from './node-settings/nodes/replace_aliases/replaceAliasesSettings';
import { StandardizeDatesSection } from './node-settings/nodes/standardize_date/StandardizeDatesSection';
import {
  DATE_MODE_OPTIONS,
  normalizeDateFormatStrategies,
  resolveDateMode,
  serializeDateFormatStrategies,
  type DateMode,
  type DateFormatStrategyConfig,
} from './node-settings/nodes/standardize_date/standardizeDateSettings';
import { TrimWhitespaceSection } from './node-settings/nodes/trim_white_space/TrimWhitespaceSection';
import { RemoveSpecialCharactersSection } from './node-settings/nodes/remove_special_char/RemoveSpecialCharactersSection';
import { NormalizeTextCaseSection } from './node-settings/nodes/normalize_text/NormalizeTextCaseSection';
import { RegexCleanupSection } from './node-settings/nodes/regex_node/RegexCleanupSection';
import { ReplaceInvalidValuesSection } from './node-settings/nodes/replace_invalid_values/ReplaceInvalidValuesSection';
import { FeatureMathSection } from './node-settings/nodes/feature_math/FeatureMathSection';
import {
  DATETIME_FEATURE_OPTIONS,
  buildFeatureMathSummaries,
  createFeatureMathOperation,
  FeatureMathOperationDraft,
  FeatureMathOperationType,
  getMethodOptions,
  normalizeFeatureMathOperations,
  serializeFeatureMathOperations,
} from './node-settings/nodes/feature_math/featureMathSettings';
import { LabelEncodingSection } from './node-settings/nodes/label_encoding/LabelEncodingSection';
import { TargetEncodingSection } from './node-settings/nodes/target_encoding/TargetEncodingSection';
import { HashEncodingSection } from './node-settings/nodes/hash_encoding/HashEncodingSection';
import { PolynomialFeaturesSection } from './node-settings/nodes/polynomial_features/PolynomialFeaturesSection';
import { FeatureSelectionSection } from './node-settings/nodes/feature_selection/FeatureSelectionSection';
import { OrdinalEncodingSection } from './node-settings/nodes/ordinal_encoding/OrdinalEncodingSection';
import { DummyEncodingSection } from './node-settings/nodes/dummy_encoding/DummyEncodingSection';
import { OneHotEncodingSection } from './node-settings/nodes/one_hot_encoding/OneHotEncodingSection';
import { TransformerAuditSection } from './node-settings/nodes/transformer_audit/TransformerAuditSection';
import { FeatureTargetSplitSection } from './node-settings/nodes/modeling/FeatureTargetSplitSection';
import { TrainTestSplitSection } from './node-settings/nodes/modeling/TrainTestSplitSection';
import { TrainModelDraftSection } from './node-settings/nodes/modeling/TrainModelDraftSection';
import { ModelTrainingSection } from './node-settings/nodes/modeling/ModelTrainingSection';
import { EvaluationPackSection } from './node-settings/nodes/modeling/EvaluationPackSection';
import { ModelRegistrySection } from './node-settings/nodes/modeling/ModelRegistrySection';
import { HyperparameterTuningSection } from './node-settings/nodes/modeling/HyperparameterTuningSection';
import { ClassResamplingSection, type ResamplingSchemaGuard } from './node-settings/nodes/resampling/ClassResamplingSection';
import { useCatalogFlags } from './node-settings/hooks/useCatalogFlags';
import { useScalingInsights } from './node-settings/hooks/useScalingInsights';
import { useBinningInsights } from './node-settings/hooks/useBinningInsights';
import { useSkewnessInsights } from './node-settings/hooks/useSkewnessInsights';
import { useBinnedDistribution } from './node-settings/hooks/useBinnedDistribution';
import { usePipelinePreview } from './node-settings/hooks/usePipelinePreview';
import { useDropColumnRecommendations } from './node-settings/hooks/useDropColumnRecommendations';
import { useAliasConfiguration } from './node-settings/hooks/useAliasConfiguration';
import { useOutlierRecommendations } from './node-settings/hooks/useOutlierRecommendations';
import { useNumericColumnAnalysis } from './node-settings/hooks/useNumericColumnAnalysis';
import { useNumericRangeSummaries } from './node-settings/hooks/useNumericRangeSummaries';
import { usePruneColumnSelections } from './node-settings/hooks/usePruneColumnSelections';
import { useTextCleanupConfiguration } from './node-settings/hooks/useTextCleanupConfiguration';
import { useReplaceInvalidConfiguration } from './node-settings/hooks/useReplaceInvalidConfiguration';
import { useStandardizeDatesConfiguration } from './node-settings/hooks/useStandardizeDatesConfiguration';
import { useScalingConfiguration } from './node-settings/hooks/useScalingConfiguration';
import { useOutlierConfiguration } from './node-settings/hooks/useOutlierConfiguration';
import { useBinningConfiguration } from './node-settings/hooks/useBinningConfiguration';
import { useImputationConfiguration } from './node-settings/hooks/useImputationConfiguration';
import { useModelingConfiguration } from './node-settings/hooks/useModelingConfiguration';
import {
  useSkewnessConfiguration,
  type SkewnessDistributionView,
} from './node-settings/hooks/useSkewnessConfiguration';
import {
  arraysAreEqual,
  cloneConfig,
  inferColumnSuggestions,
  normalizeConfigBoolean,
  pickAutoDetectValue,
  stableStringify,
} from './node-settings/utils/configParsers';
import { useLabelEncodingRecommendations } from './node-settings/hooks/useLabelEncodingRecommendations';
import { useTargetEncodingRecommendations } from './node-settings/hooks/useTargetEncodingRecommendations';
import { useHashEncodingRecommendations } from './node-settings/hooks/useHashEncodingRecommendations';
import { useOrdinalEncodingRecommendations } from './node-settings/hooks/useOrdinalEncodingRecommendations';
import { useDummyEncodingRecommendations } from './node-settings/hooks/useDummyEncodingRecommendations';
import { useOneHotEncodingRecommendations } from './node-settings/hooks/useOneHotEncodingRecommendations';
import { formatCellValue, formatColumnType, formatRelativeTime } from './node-settings/utils/formatters';
import { DATA_CONSISTENCY_GUIDANCE } from './node-settings/utils/guidance';
import { HistogramSparkline } from './node-settings/utils/HistogramSparkline';
import {
  sanitizeConstantsList,
  sanitizeDatetimeFeaturesList,
  sanitizeIntegerValue,
  sanitizeNumberValue,
  sanitizeStringList,
  sanitizeTimezoneValue,
} from './node-settings/utils/sanitizers';
import { OutlierInsightsSection } from './node-settings/nodes/outlier/OutlierInsightsSection';
import { OUTLIER_METHOD_ORDER } from './node-settings/nodes/outlier/outlierSettings';
import { useNodeMetadata } from './node-settings/hooks/useNodeMetadata';
import { ConnectionRequirementsSection } from './node-settings/layout/ConnectionRequirementsSection';
import { useConnectionReadiness } from './node-settings/hooks/useConnectionReadiness';
import { useParameterCatalog } from './node-settings/hooks/useParameterCatalog';
import { useFilteredParameters } from './node-settings/hooks/useFilteredParameters';
import { useNodeConfigState } from './node-settings/hooks/useNodeConfigState';
import { useParameterHandlers } from './node-settings/hooks/useParameterHandlers';
import { useTargetEncodingDefaults } from './node-settings/hooks/useTargetEncodingDefaults';

type NodeSettingsModalProps = {
  node: Node;
  onClose: () => void;
  onUpdateConfig: (nodeId: string, config: Record<string, any>) => void;
  onUpdateNodeData?: (nodeId: string, dataUpdates: Record<string, any>) => void;
  sourceId?: string | null;
  graphSnapshot?: {
    nodes: any[];
    edges: any[];
  } | null;
  onResetConfig?: (nodeId: string, config?: Record<string, any> | null) => void;
  defaultConfigTemplate?: Record<string, any> | null;
  isResetAvailable?: boolean;
};

export const NodeSettingsModal: React.FC<NodeSettingsModalProps> = ({
  node,
  onClose,
  onUpdateConfig,
  onUpdateNodeData,
  sourceId,
  graphSnapshot,
  onResetConfig,
  defaultConfigTemplate,
  isResetAvailable = false,
}) => {
  const { metadata, title } = useNodeMetadata(node);

  const {
    connectionInfo,
    connectedHandleKeys,
    connectedOutputHandleKeys,
    evaluationSplitConnectivity,
    connectionReady,
    gatedSectionStyle,
    gatedAriaDisabled,
  } = useConnectionReadiness(node, graphSnapshot ?? null);

  const {
    catalogType,
    isDataset,
    isImputerNode,
    isMissingIndicatorNode,
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isReplaceInvalidValuesNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isStandardizeDatesNode,
    isLabelEncodingNode,
    isTargetEncodingNode,
    isHashEncodingNode,
    isFeatureTargetSplitNode,
    isTrainTestSplitNode,
    isTrainModelDraftNode,
    isModelEvaluationNode,
    isModelRegistryNode,
    isHyperparameterTuningNode,
    isClassUndersamplingNode,
    isClassOversamplingNode,
    isOrdinalEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
    isFeatureMathNode,
    isCastNode,
    isBinningNode,
    isScalingNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
    isTransformerAuditNode,
    isOutlierNode,
    isSkewnessNode,
    isSkewnessDistributionNode,
    isBinnedDistributionNode,
    isDataConsistencyNode,
    isInspectionNode,
    isRemoveDuplicatesNode,
    isDropMissingColumnsNode,
    isDropMissingRowsNode,
    isDropMissingNode,
  } = useCatalogFlags(node);
  const datasetBadge = isDataset;
  const isClassResamplingNode = isClassUndersamplingNode || isClassOversamplingNode;

  const imputationMethodOptions = useMemo<ImputationMethodOption[]>(() => IMPUTATION_METHOD_OPTIONS, []);

  const imputationMethodValues = useMemo(
    () => imputationMethodOptions.map((option) => option.value),
    [imputationMethodOptions]
  );

  const parameters = useMemo<FeatureNodeParameter[]>(() => {
    const raw = node?.data?.parameters;
    if (!Array.isArray(raw)) {
      return [];
    }
    return raw
      .filter((parameter) => Boolean(parameter?.name))
      .map((parameter) => ({ ...parameter }));
  }, [node]);

  const getParameter = useParameterCatalog(parameters);
  const getParameterIf = (condition: boolean, name: string) => (condition ? getParameter(name) : null);

  const dataConsistencyHint = useMemo(
    () => DATA_CONSISTENCY_GUIDANCE[catalogType] ?? null,
    [catalogType]
  );

  const nodeId = typeof node?.id === 'string' ? node.id : '';
  const canResetNode = useMemo(
    () => Boolean(isResetAvailable && !isDataset && defaultConfigTemplate),
    [defaultConfigTemplate, isDataset, isResetAvailable]
  );

  const binningColumnsParameter = getParameterIf(isBinningNode, 'columns');
  const scalingColumnsParameter = getParameterIf(isScalingNode, 'columns');
  const removeDuplicatesColumnsParameter = getParameterIf(isRemoveDuplicatesNode, 'columns');
  const removeDuplicatesKeepParameter = getParameterIf(isRemoveDuplicatesNode, 'keep');
  const replaceAliasesCustomPairsParameter = getParameterIf(isReplaceAliasesNode, 'custom_pairs');
  const trimWhitespaceColumnsParameter = getParameterIf(isTrimWhitespaceNode, 'columns');
  const trimWhitespaceModeParameter = getParameterIf(isTrimWhitespaceNode, 'mode');
  const removeSpecialColumnsParameter = getParameterIf(isRemoveSpecialCharsNode, 'columns');
  const removeSpecialModeParameter = getParameterIf(isRemoveSpecialCharsNode, 'mode');
  const removeSpecialReplacementParameter = getParameterIf(isRemoveSpecialCharsNode, 'replacement');
  const replaceInvalidColumnsParameter = getParameterIf(isReplaceInvalidValuesNode, 'columns');
  const replaceInvalidModeParameter = getParameterIf(isReplaceInvalidValuesNode, 'mode');
  const replaceInvalidMinValueParameter = getParameterIf(isReplaceInvalidValuesNode, 'min_value');
  const replaceInvalidMaxValueParameter = getParameterIf(isReplaceInvalidValuesNode, 'max_value');
  const regexCleanupColumnsParameter = getParameterIf(isRegexCleanupNode, 'columns');
  const regexCleanupModeParameter = getParameterIf(isRegexCleanupNode, 'mode');
  const regexCleanupPatternParameter = getParameterIf(isRegexCleanupNode, 'pattern');
  const regexCleanupReplacementParameter = getParameterIf(isRegexCleanupNode, 'replacement');
  const normalizeCaseColumnsParameter = getParameterIf(isNormalizeTextCaseNode, 'columns');
  const normalizeCaseModeParameter = getParameterIf(isNormalizeTextCaseNode, 'mode');

  const labelEncodingColumnsParameter = isLabelEncodingNode ? getParameter('columns') : null;
  const labelEncodingAutoDetectParameter = isLabelEncodingNode ? getParameter('auto_detect') : null;
  const labelEncodingMaxUniqueParameter = isLabelEncodingNode ? getParameter('max_unique_values') : null;
  const labelEncodingOutputSuffixParameter = isLabelEncodingNode ? getParameter('output_suffix') : null;
  const labelEncodingDropOriginalParameter = isLabelEncodingNode ? getParameter('drop_original') : null;
  const labelEncodingMissingStrategyParameter = isLabelEncodingNode ? getParameter('missing_strategy') : null;
  const labelEncodingMissingCodeParameter = isLabelEncodingNode ? getParameter('missing_code') : null;

  const hashEncodingColumnsParameter = isHashEncodingNode ? getParameter('columns') : null;
  const hashEncodingAutoDetectParameter = isHashEncodingNode ? getParameter('auto_detect') : null;
  const hashEncodingMaxCategoriesParameter = isHashEncodingNode ? getParameter('max_categories') : null;
  const hashEncodingBucketsParameter = isHashEncodingNode ? getParameter('n_buckets') : null;
  const hashEncodingOutputSuffixParameter = isHashEncodingNode ? getParameter('output_suffix') : null;
  const hashEncodingDropOriginalParameter = isHashEncodingNode ? getParameter('drop_original') : null;
  const hashEncodingEncodeMissingParameter = isHashEncodingNode ? getParameter('encode_missing') : null;

  const polynomialColumnsParameter = getParameterIf(isPolynomialFeaturesNode, 'columns');
  const polynomialAutoDetectParameter = getParameterIf(isPolynomialFeaturesNode, 'auto_detect');
  const polynomialDegreeParameter = getParameterIf(isPolynomialFeaturesNode, 'degree');
  const polynomialIncludeBiasParameter = getParameterIf(isPolynomialFeaturesNode, 'include_bias');
  const polynomialInteractionOnlyParameter = getParameterIf(isPolynomialFeaturesNode, 'interaction_only');
  const polynomialIncludeInputFeaturesParameter = getParameterIf(
    isPolynomialFeaturesNode,
    'include_input_features'
  );
  const polynomialOutputPrefixParameter = getParameterIf(isPolynomialFeaturesNode, 'output_prefix');

  const featureSelectionColumnsParameter = getParameterIf(isFeatureSelectionNode, 'columns');
  const featureSelectionAutoDetectParameter = getParameterIf(isFeatureSelectionNode, 'auto_detect');
  const featureSelectionTargetColumnParameter = getParameterIf(isFeatureSelectionNode, 'target_column');
  const featureSelectionMethodParameter = getParameterIf(isFeatureSelectionNode, 'method');
  const featureSelectionScoreFuncParameter = getParameterIf(isFeatureSelectionNode, 'score_func');
  const featureSelectionProblemTypeParameter = getParameterIf(isFeatureSelectionNode, 'problem_type');
  const featureSelectionKParameter = getParameterIf(isFeatureSelectionNode, 'k');
  const featureSelectionPercentileParameter = getParameterIf(isFeatureSelectionNode, 'percentile');
  const featureSelectionAlphaParameter = getParameterIf(isFeatureSelectionNode, 'alpha');
  const featureSelectionThresholdParameter = getParameterIf(isFeatureSelectionNode, 'threshold');
  const featureSelectionModeParameter = getParameterIf(isFeatureSelectionNode, 'mode');
  const featureSelectionEstimatorParameter = getParameterIf(isFeatureSelectionNode, 'estimator');
  const featureSelectionStepParameter = getParameterIf(isFeatureSelectionNode, 'step');
  const featureSelectionMinFeaturesParameter = getParameterIf(isFeatureSelectionNode, 'min_features');
  const featureSelectionMaxFeaturesParameter = getParameterIf(isFeatureSelectionNode, 'max_features');
  const featureSelectionDropUnselectedParameter = getParameterIf(
    isFeatureSelectionNode,
    'drop_unselected'
  );

  const featureMathErrorHandlingParameter = getParameterIf(isFeatureMathNode, 'error_handling');
  const featureMathAllowOverwriteParameter = getParameterIf(isFeatureMathNode, 'allow_overwrite');
  const featureMathDefaultTimezoneParameter = getParameterIf(isFeatureMathNode, 'default_timezone');
  const featureMathEpsilonParameter = getParameterIf(isFeatureMathNode, 'epsilon');
  const resamplingMethodParameter = getParameterIf(isClassResamplingNode, 'method');
  const resamplingTargetColumnParameter = getParameterIf(isClassResamplingNode, 'target_column');
  const resamplingSamplingStrategyParameter = getParameterIf(
    isClassResamplingNode,
    'sampling_strategy'
  );
  const resamplingRandomStateParameter = getParameterIf(isClassResamplingNode, 'random_state');
  const resamplingKNeighborsParameter = getParameterIf(isClassOversamplingNode, 'k_neighbors');
  const resamplingReplacementParameter = getParameterIf(isClassUndersamplingNode, 'replacement');
  const featureTargetSplitTargetColumnParameter = getParameterIf(
    isFeatureTargetSplitNode,
    'target_column'
  );
  const hasTrainOrTuningNode = isTrainModelDraftNode || isHyperparameterTuningNode;
  const trainModelTargetColumnParameter = getParameterIf(hasTrainOrTuningNode, 'target_column');
  const trainModelProblemTypeParameter = getParameterIf(hasTrainOrTuningNode, 'problem_type');
  const trainModelModelTypeParameter = getParameterIf(hasTrainOrTuningNode, 'model_type');
  const trainModelHyperparametersParameter = getParameterIf(
    isTrainModelDraftNode,
    'hyperparameters'
  );
  const hyperparameterTuningSearchStrategyParameter = getParameterIf(
    isHyperparameterTuningNode,
    'search_strategy'
  );
  const hyperparameterTuningSearchIterationsParameter = getParameterIf(
    isHyperparameterTuningNode,
    'search_iterations'
  );
  const hyperparameterTuningSearchRandomStateParameter = getParameterIf(
    isHyperparameterTuningNode,
    'search_random_state'
  );
  const hyperparameterTuningScoringMetricParameter = getParameterIf(
    isHyperparameterTuningNode,
    'scoring_metric'
  );
  const trainModelCvEnabledParameter = getParameterIf(hasTrainOrTuningNode, 'cv_enabled');
  const trainModelCvStrategyParameter = getParameterIf(hasTrainOrTuningNode, 'cv_strategy');
  const trainModelCvFoldsParameter = getParameterIf(hasTrainOrTuningNode, 'cv_folds');
  const trainModelCvShuffleParameter = getParameterIf(hasTrainOrTuningNode, 'cv_shuffle');
  const trainModelCvRandomStateParameter = getParameterIf(hasTrainOrTuningNode, 'cv_random_state');
  const trainModelCvRefitStrategyParameter = getParameterIf(
    isTrainModelDraftNode,
    'cv_refit_strategy'
  );

  const targetEncodingColumnsParameter = isTargetEncodingNode ? getParameter('columns') : null;
  const targetEncodingTargetColumnParameter = isTargetEncodingNode ? getParameter('target_column') : null;
  const targetEncodingAutoDetectParameter = isTargetEncodingNode ? getParameter('auto_detect') : null;
  const targetEncodingMaxCategoriesParameter = isTargetEncodingNode ? getParameter('max_categories') : null;
  const targetEncodingOutputSuffixParameter = isTargetEncodingNode ? getParameter('output_suffix') : null;
  const targetEncodingDropOriginalParameter = isTargetEncodingNode ? getParameter('drop_original') : null;
  const targetEncodingSmoothingParameter = isTargetEncodingNode ? getParameter('smoothing') : null;
  const targetEncodingEncodeMissingParameter = isTargetEncodingNode ? getParameter('encode_missing') : null;
  const targetEncodingHandleUnknownParameter = isTargetEncodingNode ? getParameter('handle_unknown') : null;

  const ordinalEncodingColumnsParameter = isOrdinalEncodingNode ? getParameter('columns') : null;
  const ordinalEncodingAutoDetectParameter = isOrdinalEncodingNode ? getParameter('auto_detect') : null;
  const ordinalEncodingMaxCategoriesParameter = isOrdinalEncodingNode ? getParameter('max_categories') : null;
  const ordinalEncodingOutputSuffixParameter = isOrdinalEncodingNode ? getParameter('output_suffix') : null;
  const ordinalEncodingDropOriginalParameter = isOrdinalEncodingNode ? getParameter('drop_original') : null;
  const ordinalEncodingEncodeMissingParameter = isOrdinalEncodingNode ? getParameter('encode_missing') : null;
  const ordinalEncodingHandleUnknownParameter = isOrdinalEncodingNode ? getParameter('handle_unknown') : null;
  const ordinalEncodingUnknownValueParameter = isOrdinalEncodingNode ? getParameter('unknown_value') : null;

  const dummyEncodingColumnsParameter = isDummyEncodingNode ? getParameter('columns') : null;
  const dummyEncodingAutoDetectParameter = isDummyEncodingNode ? getParameter('auto_detect') : null;
  const dummyEncodingMaxCategoriesParameter = isDummyEncodingNode ? getParameter('max_categories') : null;
  const dummyEncodingDropFirstParameter = isDummyEncodingNode ? getParameter('drop_first') : null;
  const dummyEncodingIncludeMissingParameter = isDummyEncodingNode ? getParameter('include_missing') : null;
  const dummyEncodingDropOriginalParameter = isDummyEncodingNode ? getParameter('drop_original') : null;
  const dummyEncodingPrefixSeparatorParameter = isDummyEncodingNode ? getParameter('prefix_separator') : null;

  const oneHotEncodingColumnsParameter = isOneHotEncodingNode ? getParameter('columns') : null;
  const oneHotEncodingAutoDetectParameter = isOneHotEncodingNode ? getParameter('auto_detect') : null;
  const oneHotEncodingMaxCategoriesParameter = isOneHotEncodingNode ? getParameter('max_categories') : null;
  const oneHotEncodingDropFirstParameter = isOneHotEncodingNode ? getParameter('drop_first') : null;
  const oneHotEncodingIncludeMissingParameter = isOneHotEncodingNode ? getParameter('include_missing') : null;
  const oneHotEncodingDropOriginalParameter = isOneHotEncodingNode ? getParameter('drop_original') : null;
  const oneHotEncodingPrefixSeparatorParameter = isOneHotEncodingNode ? getParameter('prefix_separator') : null;

  const missingIndicatorColumnsParameter = getParameterIf(isMissingIndicatorNode, 'columns');
  const missingIndicatorSuffixParameter = getParameterIf(isMissingIndicatorNode, 'flag_suffix');
  const scalingDefaultMethodParameter = getParameterIf(isScalingNode, 'default_method');
  const scalingAutoDetectParameter = getParameterIf(isScalingNode, 'auto_detect');

  const requiresColumnCatalog = useMemo(
    () =>
      parameters.some(
        (parameter) =>
          parameter?.type === 'multi_select' && parameter?.source?.type !== 'drop_column_recommendations'
      ),
    [parameters]
  );

  const dropColumnParameter = useMemo(
    () => parameters.find((parameter) => parameter?.source?.type === 'drop_column_recommendations') ?? null,
    [parameters],
  );

  const dropRowsAnyParameter = getParameterIf(isDropMissingRowsNode, 'drop_if_any_missing');
  const filteredParameters = useFilteredParameters(parameters, {
    isBinningNode,
    isCastNode,
    isDropMissingColumnsNode,
    isDropMissingRowsNode,
    isImputerNode,
    isMissingIndicatorNode,
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isReplaceInvalidValuesNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isStandardizeDatesNode,
    isScalingNode,
    isClassUndersamplingNode,
    isClassOversamplingNode,
    isTrainModelDraftNode,
    isLabelEncodingNode,
    isOrdinalEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
    isRemoveDuplicatesNode,
    isSkewnessNode,
    isFeatureTargetSplitNode,
  });

  const dataConsistencyParameters = useMemo(
    () => (isDataConsistencyNode ? filteredParameters : []),
    [filteredParameters, isDataConsistencyNode],
  );

  const shouldLoadSkewnessInsights = isSkewnessNode || isSkewnessDistributionNode;

  const {
    configState,
    setConfigState,
    stableInitialConfig,
    stableCurrentConfig,
    nodeChangeVersion,
  } = useNodeConfigState({ node });

  const {
    handleParameterChange,
    handleNumberChange,
    handlePercentileChange,
    handleBooleanChange,
    handleTextChange,
  } = useParameterHandlers({ setConfigState });

  const modelRegistryConfig = useMemo(() => {
    if (!isModelRegistryNode) {
      return null;
    }
    return configState ? { ...configState } : {};
  }, [configState, isModelRegistryNode]);

  const imputerStrategies = useMemo(
    () => normalizeImputationStrategies(configState?.strategies, imputationMethodValues),
    [configState?.strategies, imputationMethodValues]
  );
  const imputerStrategyCount = imputerStrategies.length;

  const skewnessTransformations = useMemo(
    () => dedupeSkewnessTransformations(normalizeSkewnessTransformations(configState?.transformations)),
    [configState?.transformations],
  );

  const graphContext = useMemo(() => {
    if (!graphSnapshot) {
      return null;
    }
    const nodes = Array.isArray(graphSnapshot.nodes) ? graphSnapshot.nodes : [];
    const edges = Array.isArray(graphSnapshot.edges) ? graphSnapshot.edges : [];
    return { nodes, edges };
  }, [graphSnapshot]);

  const graphNodes = useMemo(() => (graphSnapshot && Array.isArray(graphSnapshot.nodes) ? graphSnapshot.nodes : []), [graphSnapshot]);
  const graphEdges = useMemo(() => (graphSnapshot && Array.isArray(graphSnapshot.edges) ? graphSnapshot.edges : []), [graphSnapshot]);
  const graphNodeCount = graphNodes.length;

  const upstreamNodeIds = useMemo(() => {
    if (!nodeId || !graphEdges.length) {
      return [] as string[];
    }
    const visited = new Set<string>();
    const stack: string[] = [nodeId];
    while (stack.length) {
      const current = stack.pop();
      if (!current) {
        continue;
      }
      graphEdges.forEach((edge: any) => {
        const sourceRaw = edge && typeof edge.source === 'string' ? edge.source.trim() : '';
        const targetRaw = edge && typeof edge.target === 'string' ? edge.target.trim() : '';
        if (!sourceRaw || !targetRaw) {
          return;
        }
        if (targetRaw === current && !visited.has(sourceRaw)) {
          visited.add(sourceRaw);
          stack.push(sourceRaw);
        }
      });
    }
    const ordered = Array.from(visited);
    ordered.sort();
    return ordered;
  }, [graphEdges, nodeId]);

  const upstreamTargetColumn = useMemo(() => {
    if (!upstreamNodeIds.length) {
      return '';
    }

    // Prioritize feature_target_split over train_test_split
    let featureTargetSplitColumn = '';
    let trainTestSplitColumn = '';

    for (const upstreamId of upstreamNodeIds) {
      const upstreamNode = graphNodes.find((n: any) => n?.id === upstreamId);
      if (!upstreamNode) {
        continue;
      }

      const catalogType = String(upstreamNode?.data?.catalogType ?? '').toLowerCase().trim();
      
      if (catalogType === 'feature_target_split') {
        const targetCol = upstreamNode?.data?.config?.target_column;
        if (typeof targetCol === 'string' && targetCol.trim()) {
          featureTargetSplitColumn = targetCol.trim();
        }
      } else if (catalogType === 'train_test_split') {
        const targetCol = upstreamNode?.data?.config?.target_column;
        if (typeof targetCol === 'string' && targetCol.trim()) {
          trainTestSplitColumn = targetCol.trim();
        }
      }
    }

    // Return feature_target_split column if available, otherwise train_test_split
    return featureTargetSplitColumn || trainTestSplitColumn || '';
  }, [graphNodes, upstreamNodeIds]);

  const {
    featureTargetSplitConfig,
    trainTestSplitConfig,
    resamplingConfig,
    trainModelDraftConfig,
    trainModelRuntimeConfig,
    trainModelCVConfig,
    filteredModelTypeOptions,
  } = useModelingConfiguration({
    configState,
    setConfigState,
    isFeatureTargetSplitNode,
    isTrainTestSplitNode,
    isTrainModelDraftNode,
  isHyperparameterTuningNode,
    isClassResamplingNode,
    isClassOversamplingNode,
    isClassUndersamplingNode,
    upstreamTargetColumn,
    nodeId,
    modelTypeOptions: trainModelModelTypeParameter?.options ?? null,
  });

  const featureMathOperations = useMemo<FeatureMathOperationDraft[]>(() => {
    if (!isFeatureMathNode) {
      return [];
    }
    return normalizeFeatureMathOperations(configState?.operations ?? []);
  }, [configState?.operations, isFeatureMathNode]);

  const upstreamConfigFingerprints = useMemo(() => {
    if (!upstreamNodeIds.length) {
      return {} as Record<string, any>;
    }
    const map: Record<string, any> = {};
    upstreamNodeIds.forEach((identifier) => {
      const match = graphNodes.find((entry: any) => entry && typeof entry.id === 'string' && entry.id === identifier);
      if (!match) {
        return;
      }
      const configPayload = match?.data?.config ?? null;
      map[identifier] = configPayload;
    });
    return map;
  }, [graphNodes, upstreamNodeIds]);

  const previewSignature = useMemo(() => {
    return stableStringify({
      sourceId: sourceId ?? null,
      nodeId: nodeId || null,
      config: configState,
      upstreamIds: upstreamNodeIds,
      upstreamConfig: upstreamConfigFingerprints,
    });
  }, [configState, nodeId, sourceId, upstreamConfigFingerprints, upstreamNodeIds]);

  const [cachedPreviewSchema, setCachedPreviewSchema] = useState<PipelinePreviewSchema | null>(null);

  const cachedSchemaColumns = useMemo(() => cachedPreviewSchema?.columns ?? [], [cachedPreviewSchema]);

  const datasetNodeIds = useMemo(() => {
    const result = new Set<string>();
    graphNodes.forEach((entry: any) => {
      if (!entry) {
        return;
      }
      const entryId = typeof entry.id === 'string' ? entry.id.trim() : String(entry?.id ?? '').trim();
      if (!entryId) {
        return;
      }
      const entryData = entry?.data ?? {};
      if (entryId === 'dataset-source' || entryData?.isDataset === true || entryData?.catalogType === 'dataset') {
        result.add(entryId);
      }
    });
    if (!result.size) {
      result.add('dataset-source');
    }
    return Array.from(result);
  }, [graphNodes]);

  const hasReachableSource = useMemo(() => {
    if (!nodeId) {
      return false;
    }

    if (isDataset) {
      return true;
    }

    if (!graphEdges.length) {
      return datasetNodeIds.includes(nodeId);
    }

    const adjacency = new Map<string, string[]>();
    graphEdges.forEach((edge: any) => {
      const rawSource = edge?.source;
      const rawTarget = edge?.target;
      const source = typeof rawSource === 'string' ? rawSource.trim() : String(rawSource ?? '').trim();
      const target = typeof rawTarget === 'string' ? rawTarget.trim() : String(rawTarget ?? '').trim();
      if (!source || !target) {
        return;
      }
      const list = adjacency.get(source);
      if (list) {
        list.push(target);
      } else {
        adjacency.set(source, [target]);
      }
    });

    const visited = new Set<string>();
    const stack = [...datasetNodeIds];

    while (stack.length) {
      const current = stack.pop();
      if (!current || visited.has(current)) {
        continue;
      }
      visited.add(current);
      const neighbors = adjacency.get(current);
      if (neighbors) {
        neighbors.forEach((neighbor) => {
          if (neighbor && !visited.has(neighbor)) {
            stack.push(neighbor);
          }
        });
      }
    }

    return visited.has(nodeId);
  }, [datasetNodeIds, graphEdges, isDataset, nodeId]);

  const oversamplingSchemaGuard = useMemo<ResamplingSchemaGuard | null>(() => {
    if (!isClassOversamplingNode) {
      return null;
    }
    if (!cachedSchemaColumns.length) {
      return null;
    }
    const targetColumn = resamplingConfig?.targetColumn;
    if (!targetColumn) {
      return null;
    }

    const allowedFamilies = new Set(['numeric', 'integer', 'boolean']);
    const blockedDetails = cachedSchemaColumns
      .filter((column) => column && column.name !== targetColumn)
      .filter((column) => !allowedFamilies.has(String(column.logical_family ?? 'unknown')))
      .map((column) => ({
        name: column.name,
        logical_family: String(column.logical_family ?? 'unknown'),
      }));

    if (!blockedDetails.length) {
      return null;
    }

    return {
      blocked: true,
      message:
        'Class oversampling requires numeric feature columns. Encode or cast the listed fields before refreshing the preview.',
      columns: blockedDetails.map((detail) => detail.name),
      details: blockedDetails,
    };
  }, [cachedSchemaColumns, isClassOversamplingNode, resamplingConfig?.targetColumn]);

  const imputationSchemaDiagnostics = useMemo<ImputationSchemaDiagnostics | null>(() => {
    if (!isImputerNode) {
      return null;
    }
    if (!cachedSchemaColumns.length) {
      return null;
    }
    if (!imputerStrategies.length) {
      return null;
    }

    const numericFamilies = new Set(['numeric', 'integer']);
    const columnLookup = new Map(cachedSchemaColumns.map((column) => [column.name, column]));
    const entries: ImputationSchemaDiagnostics['entries'] = [];

    imputerStrategies.forEach((strategy, index) => {
      if (!isNumericImputationMethod(strategy.method)) {
        return;
      }
      if (!Array.isArray(strategy.columns) || !strategy.columns.length) {
        return;
      }

      const invalidDetails = strategy.columns
        .map((columnName) => columnLookup.get(columnName))
        .filter((column): column is NonNullable<(typeof cachedSchemaColumns)[number]> => Boolean(column))
        .filter((column) => !numericFamilies.has(String(column.logical_family ?? 'unknown')))
        .map((column) => ({
          name: column.name,
          logical_family: String(column.logical_family ?? 'unknown'),
        }));

      if (invalidDetails.length) {
        entries.push({
          index,
          columns: invalidDetails.map((detail) => detail.name),
          details: invalidDetails,
        });
      }
    });

    if (!entries.length) {
      return null;
    }

    return {
      blocked: true,
      message:
        'Numeric imputation methods can only target numeric columns. Recast or adjust the highlighted fields before running the preview.',
      entries,
    };
  }, [cachedSchemaColumns, imputerStrategies, isImputerNode]);

  const skipPreview = Boolean(oversamplingSchemaGuard?.blocked || imputationSchemaDiagnostics?.blocked);

  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [columnSearch, setColumnSearch] = useState('');
  const [columnMissingMap, setColumnMissingMap] = useState<Record<string, number>>({});
  const [columnTypeMap, setColumnTypeMap] = useState<Record<string, string>>({});
  const [columnSuggestions, setColumnSuggestions] = useState<Record<string, string[]>>({});
  const [imputerMissingFilter, setImputerMissingFilter] = useState(0);

  const hasDropColumnParameter = Boolean(dropColumnParameter);

  const {
    hasDropColumnSource,
    availableFilters,
    activeFilterId,
    setActiveFilterId,
    recommendations,
    filteredRecommendations,
    formatSignalName,
    isFetchingRecommendations,
    recommendationsError,
    recommendationsGeneratedAt,
    suggestedThreshold,
    refreshRecommendations,
  } = useDropMissingColumns({
    hasDropColumnSource: hasDropColumnParameter,
    sourceId,
    nodeId,
    graphContext,
    hasReachableSource,
    columnTypeMap,
    setAvailableColumns,
    setColumnSearch,
    setColumnMissingMap,
    setColumnTypeMap,
    setColumnSuggestions,
  });

  useDropColumnRecommendations({
    shouldLoad: isImputerNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
    setAvailableColumns,
    setColumnMissingMap,
  });

  const {
    isFetching: isFetchingLabelEncoding,
    error: labelEncodingError,
    suggestions: labelEncodingSuggestions,
    metadata: labelEncodingMetadata,
  } = useLabelEncodingRecommendations({
    isLabelEncodingNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  const {
    isFetching: isFetchingTargetEncoding,
    error: targetEncodingError,
    suggestions: targetEncodingSuggestions,
    metadata: targetEncodingMetadata,
  } = useTargetEncodingRecommendations({
    isTargetEncodingNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  useTargetEncodingDefaults({
    isTargetEncodingNode,
    enableGlobalFallbackDefault: targetEncodingMetadata.enableGlobalFallbackDefault,
    encodeMissing: configState?.encode_missing,
    handleUnknown: configState?.handle_unknown,
    setConfigState,
    nodeChangeVersion,
  });

  const {
    isFetching: isFetchingHashEncoding,
    error: hashEncodingError,
    suggestions: hashEncodingSuggestions,
    metadata: hashEncodingMetadata,
  } = useHashEncodingRecommendations({
    isHashEncodingNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  const {
    isFetching: isFetchingOrdinalEncoding,
    error: ordinalEncodingError,
    suggestions: ordinalEncodingSuggestions,
    metadata: ordinalEncodingMetadata,
  } = useOrdinalEncodingRecommendations({
    isOrdinalEncodingNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  const {
    isFetching: isFetchingDummyEncoding,
    error: dummyEncodingError,
    suggestions: dummyEncodingSuggestions,
    metadata: dummyEncodingMetadata,
  } = useDummyEncodingRecommendations({
    isDummyEncodingNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  const {
    isFetching: isFetchingOneHotEncoding,
    error: oneHotEncodingError,
    suggestions: oneHotEncodingSuggestions,
    metadata: oneHotEncodingMetadata,
  } = useOneHotEncodingRecommendations({
    isOneHotEncodingNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  useEffect(() => {
    if (!isLabelEncodingNode) {
      return;
    }
    if (!labelEncodingMetadata.autoDetectDefault) {
      return;
    }
    if (configState?.auto_detect === true) {
      return;
    }
    if (configState?.auto_detect === false) {
      return;
    }
    setConfigState((previous) => {
      if (previous?.auto_detect !== undefined && previous.auto_detect !== null) {
        return previous;
      }
      return {
        ...previous,
        auto_detect: true,
      };
    });
  }, [configState?.auto_detect, isLabelEncodingNode, labelEncodingMetadata.autoDetectDefault, setConfigState]);

  useEffect(() => {
    if (!isTargetEncodingNode) {
      return;
    }
    if (!targetEncodingMetadata.autoDetectDefault) {
      return;
    }
    if (configState?.auto_detect === true) {
      return;
    }
    if (configState?.auto_detect === false) {
      return;
    }
    setConfigState((previous) => {
      if (previous?.auto_detect !== undefined && previous.auto_detect !== null) {
        return previous;
      }
      return {
        ...previous,
        auto_detect: true,
      };
    });
  }, [configState?.auto_detect, isTargetEncodingNode, setConfigState, targetEncodingMetadata.autoDetectDefault]);

  useEffect(() => {
    if (!isHashEncodingNode) {
      return;
    }
    if (!hashEncodingMetadata.autoDetectDefault) {
      return;
    }
    if (configState?.auto_detect === true) {
      return;
    }
    if (configState?.auto_detect === false) {
      return;
    }
    setConfigState((previous) => {
      if (previous?.auto_detect !== undefined && previous.auto_detect !== null) {
        return previous;
      }
      return {
        ...previous,
        auto_detect: true,
      };
    });
  }, [configState?.auto_detect, hashEncodingMetadata.autoDetectDefault, isHashEncodingNode, setConfigState]);

  useEffect(() => {
    if (!isHashEncodingNode) {
      return;
    }
    const suggestedBuckets = hashEncodingMetadata.suggestedBucketDefault;
    if (!suggestedBuckets || !Number.isFinite(suggestedBuckets)) {
      return;
    }
    const numericBuckets = Math.max(2, Math.round(suggestedBuckets));
    setConfigState((previous) => {
      if (!previous || typeof previous !== 'object' || Array.isArray(previous)) {
        return previous;
      }
      const current = previous.n_buckets;
      if (current !== undefined && current !== null) {
        const numericCurrent = Number(current);
        if (Number.isFinite(numericCurrent) && Math.round(numericCurrent) > 0) {
          return previous;
        }
      }
      return {
        ...previous,
        n_buckets: numericBuckets,
      };
    });
  }, [hashEncodingMetadata.suggestedBucketDefault, isHashEncodingNode, setConfigState]);

  useEffect(() => {
    if (!isTargetEncodingNode) {
      return;
    }
    if (!targetEncodingMetadata.enableGlobalFallbackDefault) {
      return;
    }

    const encodeMissingActive = typeof configState?.encode_missing === 'boolean' ? configState.encode_missing : null;
    const currentHandleUnknown = typeof configState?.handle_unknown === 'string'
      ? configState.handle_unknown.trim().toLowerCase()
      : '';

    if (encodeMissingActive === true && currentHandleUnknown === 'global_mean') {
      return;
    }

    setConfigState((previous) => {
      const previousEncodeMissing = typeof previous?.encode_missing === 'boolean' ? previous.encode_missing : null;
      const previousHandleUnknown = typeof previous?.handle_unknown === 'string'
        ? previous.handle_unknown.trim().toLowerCase()
        : '';

      if (previousEncodeMissing === true && previousHandleUnknown === 'global_mean') {
        return previous;
      }

      const next: Record<string, any> = { ...previous };

      if (previousEncodeMissing !== true) {
        next.encode_missing = true;
      }

      if (previousHandleUnknown !== 'global_mean') {
        next.handle_unknown = 'global_mean';
      }

      return next;
    });
  }, [
    configState?.encode_missing,
    configState?.handle_unknown,
    isTargetEncodingNode,
    setConfigState,
    targetEncodingMetadata.enableGlobalFallbackDefault,
  ]);

  useEffect(() => {
    if (!isOrdinalEncodingNode) {
      return;
    }
    if (!ordinalEncodingMetadata.autoDetectDefault) {
      return;
    }
    if (configState?.auto_detect === true) {
      return;
    }
    if (configState?.auto_detect === false) {
      return;
    }
    setConfigState((previous) => {
      if (previous?.auto_detect !== undefined && previous.auto_detect !== null) {
        return previous;
      }
      return {
        ...previous,
        auto_detect: true,
      };
    });
  }, [configState?.auto_detect, isOrdinalEncodingNode, ordinalEncodingMetadata.autoDetectDefault, setConfigState]);

  useEffect(() => {
    if (!isOrdinalEncodingNode) {
      return;
    }
    if (!ordinalEncodingMetadata.enableUnknownDefault) {
      return;
    }
    const raw = typeof configState?.handle_unknown === 'string' ? configState.handle_unknown.trim().toLowerCase() : '';
    if (raw) {
      return;
    }
    setConfigState((previous) => {
      const current = typeof previous?.handle_unknown === 'string' ? previous.handle_unknown.trim().toLowerCase() : '';
      if (current) {
        return previous;
      }
      return {
        ...previous,
        handle_unknown: 'use_encoded_value',
      };
    });
  }, [configState?.handle_unknown, isOrdinalEncodingNode, ordinalEncodingMetadata.enableUnknownDefault, setConfigState]);

  useEffect(() => {
    if (!isDummyEncodingNode) {
      return;
    }
    if (!dummyEncodingMetadata.autoDetectDefault) {
      return;
    }
    if (configState?.auto_detect === true) {
      return;
    }
    if (configState?.auto_detect === false) {
      return;
    }
    setConfigState((previous) => {
      if (previous?.auto_detect !== undefined && previous.auto_detect !== null) {
        return previous;
      }
      return {
        ...previous,
        auto_detect: true,
      };
    });
  }, [configState?.auto_detect, dummyEncodingMetadata.autoDetectDefault, isDummyEncodingNode, setConfigState]);

  useEffect(() => {
    if (!requiresColumnCatalog && !isImputerNode) {
      return;
    }

    if (sourceId && hasReachableSource) {
      return;
    }

    setAvailableColumns([]);
    setColumnMissingMap({});
    setColumnTypeMap({});
    setColumnSuggestions({});
    setColumnSearch('');

  }, [hasReachableSource, isImputerNode, requiresColumnCatalog, sourceId]);

  useEffect(() => {
    setImputerMissingFilter(0);
  }, [isImputerNode, node?.id]);

  const [binnedSamplePreset, setBinnedSamplePreset] = useState<BinnedSamplePresetValue>('500');

  const [collapsedStrategies, setCollapsedStrategies] = useState<Set<number>>(() => new Set());
  const [collapsedFeatureMath, setCollapsedFeatureMath] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    setCollapsedStrategies(new Set());
  }, [stableInitialConfig]);


  const isPreviewNode = useMemo(() => node?.data?.catalogType === 'data_preview', [node?.data?.catalogType]);
  const isDatasetProfileNode = useMemo(
    () => node?.data?.catalogType === 'dataset_profile',
    [node?.data?.catalogType]
  );
  const profilingGraphSignature = useMemo(() => (graphContext ? stableStringify(graphContext) : ''), [graphContext]);

  const datasetProfileController = useDatasetProfileController({
    node,
    isDatasetProfileNode,
    sourceId,
    hasReachableSource,
    graphContext,
    profilingGraphSignature,
    formatRelativeTime,
  });
  const { profileState } = datasetProfileController;

  const shouldFetchPreview = useMemo(() => {
    if (!graphSnapshot) {
      return false;
    }
    if (!graphNodeCount) {
      return false;
    }
    if (!hasReachableSource) {
      return false;
    }
    return true;
  }, [graphNodeCount, graphSnapshot, hasReachableSource, sourceId]);

  const canTriggerPreview = useMemo(() => {
    if (!graphSnapshot) {
      return false;
    }
    if (!sourceId) {
      return false;
    }
    if (!hasReachableSource) {
      return false;
    }
    return graphNodeCount > 0;
  }, [graphNodeCount, graphSnapshot, hasReachableSource, sourceId]);

  const shouldIncludeSignals =
    isPreviewNode ||
    isFeatureMathNode ||
    isPolynomialFeaturesNode ||
    isFeatureSelectionNode ||
    isTrainTestSplitNode ||
    isOutlierNode ||
    isTransformerAuditNode;

  const { previewState, refreshPreview } = usePipelinePreview({
    shouldFetchPreview,
    sourceId,
    canTriggerPreview,
    graphSnapshot: graphSnapshot ?? null,
    isPreviewNode,
    targetNodeId: node?.id ?? null,
    previewSignature,
    skipPreview,
    requestPreviewRows: isPreviewNode,
    includeSignals: shouldIncludeSignals,
  });

  const featureMathSignals = useMemo<FeatureMathNodeSignal[]>(() => {
    if (!isFeatureMathNode) {
      return [];
    }
    const rawSignals = previewState.data?.signals?.feature_math;
    return Array.isArray(rawSignals) ? rawSignals : [];
  }, [isFeatureMathNode, previewState.data?.signals?.feature_math]);

  const polynomialSignal = useMemo<PolynomialFeaturesNodeSignal | null>(() => {
    if (!isPolynomialFeaturesNode) {
      return null;
    }
    const rawSignals = previewState.data?.signals?.polynomial_features;
    if (!Array.isArray(rawSignals) || rawSignals.length === 0) {
      return null;
    }
    const matching = nodeId
      ? rawSignals.find((entry) => entry && typeof entry.node_id === 'string' && entry.node_id === nodeId)
      : null;
    return matching ?? rawSignals[0] ?? null;
  }, [isPolynomialFeaturesNode, nodeId, previewState.data?.signals?.polynomial_features]);

  const featureSelectionSignal = useMemo<FeatureSelectionNodeSignal | null>(() => {
    if (!isFeatureSelectionNode) {
      return null;
    }
    const rawSignals = previewState.data?.signals?.feature_selection;
    if (!Array.isArray(rawSignals) || rawSignals.length === 0) {
      return null;
    }
    const matching = nodeId
      ? rawSignals.find((entry) => entry && typeof entry.node_id === 'string' && entry.node_id === nodeId)
      : null;
    return matching ?? rawSignals[0] ?? null;
  }, [isFeatureSelectionNode, nodeId, previewState.data?.signals?.feature_selection]);

  useEffect(() => {
    if (!isFeatureSelectionNode) {
      return;
    }

    const normalizedSignalTarget =
      typeof featureSelectionSignal?.target_column === 'string'
        ? featureSelectionSignal.target_column.trim()
        : '';
    const fallbackTarget = typeof upstreamTargetColumn === 'string' ? upstreamTargetColumn.trim() : '';
    const resolvedTarget = normalizedSignalTarget || fallbackTarget;

    if (!resolvedTarget) {
      return;
    }

    setConfigState((previous) => {
      const currentTarget =
        typeof previous?.target_column === 'string' ? previous.target_column.trim() : '';
      if (currentTarget) {
        return previous;
      }
      return {
        ...previous,
        target_column: resolvedTarget,
      };
    });
  }, [featureSelectionSignal?.target_column, isFeatureSelectionNode, setConfigState, upstreamTargetColumn]);

  useEffect(() => {
    if (!isFeatureSelectionNode) {
      return;
    }

    const backendK =
      typeof featureSelectionSignal?.k === 'number' && Number.isFinite(featureSelectionSignal.k)
        ? Math.max(0, Math.trunc(featureSelectionSignal.k))
        : null;
    const selectedCount = Array.isArray(featureSelectionSignal?.selected_columns)
      ? featureSelectionSignal.selected_columns.length
      : null;

    const candidate = (backendK ?? selectedCount) ?? null;
    if (candidate === null || candidate <= 0) {
      return;
    }

    setConfigState((previous) => {
      const rawValue = previous?.k;
      let normalizedCurrent: number | null = null;

      if (typeof rawValue === 'number' && Number.isFinite(rawValue)) {
        normalizedCurrent = Math.trunc(rawValue);
      } else if (typeof rawValue === 'string') {
        const parsed = Number(rawValue);
        if (Number.isFinite(parsed)) {
          normalizedCurrent = Math.trunc(parsed);
        }
      }

      if (normalizedCurrent !== null && normalizedCurrent <= candidate) {
        return previous;
      }

      return {
        ...previous,
        k: candidate,
      };
    });
  }, [
    featureSelectionSignal?.k,
    featureSelectionSignal?.selected_columns,
    isFeatureSelectionNode,
    setConfigState,
  ]);

  const featureMathSummaries = useMemo(
    () => (isFeatureMathNode ? buildFeatureMathSummaries(featureMathOperations, featureMathSignals) : []),
    [featureMathOperations, featureMathSignals, isFeatureMathNode],
  );

  useEffect(() => {
    if (!isFeatureMathNode) {
      setCollapsedFeatureMath(() => new Set());
      return;
    }
    setCollapsedFeatureMath((previous) => {
      if (!previous.size) {
        return previous;
      }
      const validIds = new Set(featureMathOperations.map((operation) => operation.id));
      const next = new Set<string>();
      previous.forEach((id) => {
        if (validIds.has(id)) {
          next.add(id);
        }
      });
      return next.size === previous.size ? previous : next;
    });
  }, [featureMathOperations, isFeatureMathNode]);

  const {
    outlierData,
    outlierError,
    isFetchingOutliers,
    refreshOutliers,
  } = useOutlierRecommendations({
    isOutlierNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: nodeId || null,
  });

  const {
    scalingData,
    scalingError,
    isFetchingScaling,
    refreshScaling,
  } = useScalingInsights({
    isScalingNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  const {
    binningData,
    binningError,
    isFetchingBinning,
    refreshBinning,
  } = useBinningInsights({
    isBinningNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  const handleRefreshPreview = useCallback(() => {
    setCachedPreviewSchema(null);
    refreshPreview();
    if (isScalingNode) {
      refreshScaling();
    }
    if (isBinningNode) {
      refreshBinning();
    }
    if (isOutlierNode) {
      refreshOutliers();
    }
  }, [
    isBinningNode,
    isOutlierNode,
    isScalingNode,
    refreshBinning,
    refreshOutliers,
    refreshPreview,
    refreshScaling,
  ]);

  useEffect(() => {
    const nextSchema = previewState.data?.schema ?? null;
    if (!nextSchema) {
      return;
    }
    setCachedPreviewSchema((previous) => {
      if (previous?.signature && nextSchema.signature && previous.signature === nextSchema.signature) {
        return previous;
      }
      return nextSchema;
    });
  }, [previewState.data?.schema]);

  useEffect(() => {
    setCachedPreviewSchema(null);
  }, [sourceId]);

  useEffect(() => {
    setCachedPreviewSchema(null);
  }, [previewSignature]);

  const {
    binnedDistributionData,
    binnedDistributionError,
    isFetchingBinnedDistribution,
    refreshBinnedDistribution,
  } = useBinnedDistribution({
    isBinnedDistributionNode,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
    samplePreset: binnedSamplePreset,
  });

  const {
    skewnessData,
    skewnessError,
    isFetchingSkewness,
    refreshSkewness,
  } = useSkewnessInsights({
    shouldLoad: shouldLoadSkewnessInsights,
    sourceId,
    graphContext,
    targetNodeId: node?.id ?? null,
    transformations: skewnessTransformations,
  });

  const previewColumns = useMemo(() => {
    const rawColumns = previewState.data?.columns;
    if (!Array.isArray(rawColumns)) {
      return [] as string[];
    }
    return rawColumns.filter((column): column is string => typeof column === 'string');
  }, [previewState.data?.columns]);

  const previewColumnStats = useMemo(() => {
    const stats = previewState.data?.column_stats;
    if (!Array.isArray(stats)) {
      return [] as any[];
    }
    return stats;
  }, [previewState.data?.column_stats]);

  const activeFlagSuffix = useMemo(() => {
    if (!isMissingIndicatorNode) {
      return '';
    }
    return resolveMissingIndicatorSuffix(configState?.flag_suffix, node?.data?.config?.flag_suffix);
  }, [configState?.flag_suffix, isMissingIndicatorNode, node?.data?.config?.flag_suffix]);

  const missingIndicatorColumns = useMemo(() => {
    if (!isMissingIndicatorNode) {
      return [] as string[];
    }
    return ensureArrayOfString(configState?.columns);
  }, [configState?.columns, isMissingIndicatorNode]);

  const missingIndicatorInsights = useMemo<MissingIndicatorInsights>(() => {
    if (!isMissingIndicatorNode) {
      return { rows: [], flaggedColumnsInDataset: [], conflictCount: 0 };
    }
    return buildMissingIndicatorInsights({
      selectedColumns: missingIndicatorColumns,
      availableColumns,
      columnMissingMap,
      suffix: activeFlagSuffix,
    });
  }, [activeFlagSuffix, availableColumns, columnMissingMap, isMissingIndicatorNode, missingIndicatorColumns]);

  const previewSampleRows = useMemo(() => {
    if (!Array.isArray(previewState.data?.sample_rows)) {
      return [] as Record<string, any>[];
    }
    return previewState.data.sample_rows;
  }, [previewState.data?.sample_rows]);

  const {
    aliasAutoDetectMeta,
    aliasStrategies,
    aliasStrategyCount,
    aliasSelectedColumns,
    aliasAutoDetectEnabled,
    replaceAliasesCustomPairsValue,
    aliasColumnSummary,
    aliasCustomPairSummary,
    aliasSampleMap,
    aliasColumnOptions,
  } = useAliasConfiguration({
    isReplaceAliasesNode,
    configState,
    nodeConfig: node?.data?.config ?? null,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const {
    trimWhitespace: {
      columnSummary: trimWhitespaceColumnSummary,
      sampleMap: trimWhitespaceSampleMap,
      modeDetails: trimWhitespaceModeDetails,
    },
    removeSpecial: {
      columnSummary: removeSpecialColumnSummary,
      sampleMap: removeSpecialSampleMap,
      modeDetails: removeSpecialModeDetails,
      selectedMode: removeSpecialMode,
    },
    regexCleanup: {
      columnSummary: regexCleanupColumnSummary,
      sampleMap: regexCleanupSampleMap,
      modeDetails: regexCleanupModeDetails,
      selectedMode: regexCleanupMode,
      replacementValue: regexCleanupReplacementValue,
    },
    normalizeCase: {
      columnSummary: normalizeCaseColumnSummary,
      sampleMap: normalizeCaseSampleMap,
      modeDetails: normalizeCaseModeDetails,
      selectedMode: normalizeCaseMode,
    },
  } = useTextCleanupConfiguration({
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    configState,
    nodeConfig: node?.data?.config ?? null,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const {
    selectedMode: replaceInvalidMode,
    modeDetails: replaceInvalidModeDetails,
    sampleMap: replaceInvalidSampleMap,
    columnSummary: replaceInvalidColumnSummary,
    minValue: replaceInvalidMinValue,
    maxValue: replaceInvalidMaxValue,
  } = useReplaceInvalidConfiguration({
    isReplaceInvalidValuesNode,
    configState,
    nodeConfig: node?.data?.config ?? null,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const {
    selectedMode: standardizeDatesMode,
    strategies: dateStrategies,
    sampleMap: standardizeDatesSampleMap,
    columnSummary: standardizeDatesColumnSummary,
    columnOptions: dateColumnOptions,
  } = useStandardizeDatesConfiguration({
    isStandardizeDatesNode,
    configState,
    nodeConfig: node?.data?.config ?? null,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const shouldAnalyzeNumericColumns = isBinningNode || isScalingNode || isOutlierNode;
  const { numericExcludedColumns } = useNumericColumnAnalysis({
    shouldAnalyze: shouldAnalyzeNumericColumns,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });
  
  const {
    scalingConfig,
    scalingExcludedColumns,
    scalingMethodDetailMap,
    scalingMethodOptions,
    scalingRecommendations,
    scalingRecommendationRows,
    scalingSelectedCount,
    scalingOverrideCount,
    scalingDefaultDetail,
    scalingDefaultLabel,
    scalingAutoDetectEnabled,
    scalingHasRecommendations,
    scalingStatusMessage,
    scalingOverrideExampleSummary,
  } = useScalingConfiguration({
    isScalingNode,
    configState,
    numericExcludedColumns,
    scalingData,
  });

  const {
    outlierConfig,
    outlierExcludedColumns,
    outlierMethodLabelMap,
    outlierMethodDetailMap,
    outlierMethodOptions,
    outlierRecommendations,
    outlierRecommendationRows,
    outlierSelectedCount,
    outlierOverrideCount,
    outlierParameterOverrideCount,
    outlierDefaultDetail,
    outlierDefaultLabel,
    outlierAutoDetectEnabled,
    outlierHasRecommendations,
    outlierStatusMessage,
    outlierOverrideExampleSummary,
  } = useOutlierConfiguration({
    isOutlierNode,
    configState,
    numericExcludedColumns,
    outlierData,
  });
  const outlierSampleSize = typeof outlierData?.sample_size === 'number' ? outlierData.sample_size : null;
  const relativeOutlierGeneratedAt = null;
  const outlierHasOverrides = outlierOverrideCount > 0 || outlierParameterOverrideCount > 0;
  const outlierOverrideSummaryDisplay = useMemo(() => {
    if (outlierOverrideCount > 0) {
      return outlierOverrideExampleSummary;
    }
    if (outlierParameterOverrideCount > 0) {
      return `${outlierParameterOverrideCount} column${outlierParameterOverrideCount === 1 ? '' : 's'} with parameter overrides`;
    }
    return null;
  }, [outlierOverrideCount, outlierParameterOverrideCount, outlierOverrideExampleSummary]);

  const {
    binningConfig,
    binningSelectedCount,
    binningDefaultLabel,
    binningOverrideColumns,
    binningOverrideCount,
    binningOverrideSummary,
    fieldIds: binningFieldIds,
    customEdgeDrafts: binningCustomEdgeDrafts,
    setCustomEdgeDrafts: setBinningCustomEdgeDrafts,
    customLabelDrafts: binningCustomLabelDrafts,
    setCustomLabelDrafts: setBinningCustomLabelDrafts,
  } = useBinningConfiguration({
    configState,
    nodeId,
  });
  const removeDuplicatesKeepSelectId = `${nodeId || 'node'}-remove-duplicates-keep`;

  const selectedColumns = useMemo(() => {
    if (isScalingNode) {
      return scalingConfig.columns;
    }
    if (isBinningNode) {
      return binningConfig.columns;
    }
    if (isOutlierNode) {
      return outlierConfig.columns;
    }
    return ensureArrayOfString(configState?.columns);
  }, [binningConfig.columns, configState, isBinningNode, isOutlierNode, isScalingNode, outlierConfig.columns, scalingConfig.columns]);

  const binningInsightsRecommendations = useMemo<BinningColumnRecommendation[]>(() => {
    const rows = binningData?.columns;
    if (!Array.isArray(rows)) {
      return [];
    }
    return rows.filter((entry): entry is BinningColumnRecommendation => {
      if (!entry || typeof entry.column !== 'string') {
        return false;
      }
      return Boolean(entry.column.trim());
    });
  }, [binningData?.columns]);

  const binningRecommendedColumnSet = useMemo(() => {
    if (!binningInsightsRecommendations.length) {
      return new Set<string>();
    }
    const normalized = new Set<string>();
    binningInsightsRecommendations.forEach((entry) => {
      const columnName = typeof entry.column === 'string' ? entry.column.trim() : '';
      if (!columnName) {
        return;
      }
      normalized.add(columnName);
    });
    return normalized;
  }, [binningInsightsRecommendations]);

  const binningAllNumericColumns = useMemo(() => {
    if (!isBinningNode) {
      return [] as string[];
    }
    const normalized = new Set<string>();
    availableColumns.forEach((column) => {
      if (numericExcludedColumns.has(column)) {
        return;
      }
      const trimmed = column.trim();
      if (trimmed) {
        normalized.add(trimmed);
      }
    });
    binningInsightsRecommendations.forEach((entry) => {
      const columnName = typeof entry.column === 'string' ? entry.column.trim() : '';
      if (!columnName) {
        return;
      }
      if (!numericExcludedColumns.has(columnName) || binningRecommendedColumnSet.has(columnName)) {
        normalized.add(columnName);
      }
    });
    selectedColumns.forEach((column) => {
      const columnName = typeof column === 'string' ? column.trim() : '';
      if (!columnName) {
        return;
      }
      if (!numericExcludedColumns.has(columnName) || binningRecommendedColumnSet.has(columnName)) {
        normalized.add(columnName);
      }
    });
    return Array.from(normalized).sort((a, b) => a.localeCompare(b));
  }, [availableColumns, binningInsightsRecommendations, binningRecommendedColumnSet, isBinningNode, numericExcludedColumns, selectedColumns]);

  const binningNumericColumnsNotSelected = useMemo(() => {
    if (!isBinningNode || !binningAllNumericColumns.length) {
      return [] as string[];
    }
    return binningAllNumericColumns.filter((column) => !selectedColumns.includes(column));
  }, [binningAllNumericColumns, isBinningNode, selectedColumns]);

  const canApplyAllBinningNumeric = binningNumericColumnsNotSelected.length > 0;

  const removeDuplicatesKeep = useMemo<KeepStrategy>(() => {
    const raw = typeof configState?.keep === 'string' ? configState.keep.trim().toLowerCase() : '';
    if (raw === 'last' || raw === 'none') {
      return raw;
    }
    return DEFAULT_KEEP_STRATEGY;
  }, [configState?.keep]);

  const {
    binningExcludedColumns,
    binningColumnPreviewMap,
    manualBoundColumns,
    manualRangeFallbackMap,
  } = useNumericRangeSummaries({
    isBinningNode,
    numericExcludedColumns,
    selectedColumns,
    previewSampleRows,
    availableColumns,
    recommendedColumns: binningRecommendedColumnSet,
  });

  const binningInsightsExcludedColumns = useMemo<BinningExcludedColumn[]>(() => {
    const rows = binningData?.excluded_columns;
    if (!Array.isArray(rows)) {
      return [];
    }
    return rows.filter((entry): entry is BinningExcludedColumn => {
      if (!entry || typeof entry.column !== 'string') {
        return false;
      }
      return Boolean(entry.column.trim());
    });
  }, [binningData?.excluded_columns]);

  usePruneColumnSelections({
    isScalingNode,
    isBinningNode,
    isOutlierNode,
    scalingExcludedColumns,
    binningExcludedColumns,
    outlierExcludedColumns,
    setConfigState,
  });

  const outlierPreviewSignal = useMemo<OutlierNodeSignal | null>(() => {
    if (!isOutlierNode) {
      return null;
    }
    const signals = previewState.data?.signals?.outlier_removal;
    if (!Array.isArray(signals) || !signals.length) {
      return null;
    }
    if (!nodeId) {
      return (signals[signals.length - 1] as OutlierNodeSignal | undefined) ?? null;
    }
    const match = signals.find((signal: any) => signal && signal.node_id === nodeId);
    return (match as OutlierNodeSignal | undefined) ?? ((signals[signals.length - 1] as OutlierNodeSignal | undefined) ?? null);
  }, [isOutlierNode, nodeId, previewState.data?.signals?.outlier_removal]);

  const transformerAuditSignal = useMemo<TransformerAuditNodeSignal | null>(() => {
    if (!isTransformerAuditNode) {
      return null;
    }
    const signals = previewState.data?.signals?.transformer_audit;
    if (!Array.isArray(signals) || !signals.length) {
      return null;
    }
    if (!nodeId) {
      return (signals[signals.length - 1] as TransformerAuditNodeSignal | undefined) ?? null;
    }
    const match = signals.find((signal: any) => signal && signal.node_id === nodeId);
    return (
      (match as TransformerAuditNodeSignal | undefined) ??
      ((signals[signals.length - 1] as TransformerAuditNodeSignal | undefined) ?? null)
    );
  }, [isTransformerAuditNode, nodeId, previewState.data?.signals?.transformer_audit]);

  useEffect(() => {
    const stats = Array.isArray(previewState.data?.column_stats) ? previewState.data.column_stats : [];

    if (!stats.length) {
      if (previewState.status === 'loading') {
        return;
      }
      setColumnTypeMap((previous) => (Object.keys(previous).length ? {} : previous));
      setColumnSuggestions((previous) => (Object.keys(previous).length ? {} : previous));
      return;
    }

    const nextTypeMap: Record<string, string> = {};
    const nextSuggestionMap: Record<string, string[]> = {};

    stats.forEach((stat) => {
      if (!stat || !stat.name) {
        return;
      }
      const name = String(stat.name).trim();
      if (!name) {
        return;
      }
      if (activeFlagSuffix && name.endsWith(activeFlagSuffix)) {
        return;
      }
      const dtype = typeof stat.dtype === 'string' ? stat.dtype : null;
      if (dtype) {
        nextTypeMap[name] = dtype;
      }
      const sampledValues = previewSampleRows.map((row) => (row && Object.prototype.hasOwnProperty.call(row, name) ? row[name] : undefined));
      const suggestions = inferColumnSuggestions(dtype, sampledValues);
      if (suggestions.length) {
        nextSuggestionMap[name] = suggestions;
      }
    });

    setColumnTypeMap((previous) => {
      const previousEntries = Object.entries(previous);
      const nextEntries = Object.entries(nextTypeMap);
      if (
        previousEntries.length === nextEntries.length &&
        previousEntries.every(([key, value]) => Object.prototype.hasOwnProperty.call(nextTypeMap, key) && nextTypeMap[key] === value)
      ) {
        return previous;
      }
      return nextTypeMap;
    });

    setColumnSuggestions((previous) => {
      const previousKeys = Object.keys(previous);
      const nextKeys = Object.keys(nextSuggestionMap);
      if (
        previousKeys.length === nextKeys.length &&
        previousKeys.every((key) => {
          if (!Object.prototype.hasOwnProperty.call(nextSuggestionMap, key)) {
            return false;
          }
          const existing = previous[key] ?? [];
          const next = nextSuggestionMap[key] ?? [];
          return arraysAreEqual(existing, next);
        })
      ) {
        return previous;
      }
      return nextSuggestionMap;
    });
  }, [activeFlagSuffix, previewSampleRows, previewState.data?.column_stats, previewState.status]);

  useEffect(() => {
    if (!requiresColumnCatalog) {
      return;
    }

    if (!hasReachableSource) {
      return;
    }

    const nextColumns = new Set<string>();

    const previewColumns = Array.isArray(previewState.data?.columns) ? previewState.data.columns : [];
    previewColumns.forEach((column) => {
      const normalized = typeof column === 'string' ? column.trim() : '';
      if (!normalized) {
        return;
      }
      if (activeFlagSuffix && normalized.endsWith(activeFlagSuffix)) {
        return;
      }
      nextColumns.add(normalized);
    });

    const previewStats = Array.isArray(previewState.data?.column_stats) ? previewState.data.column_stats : [];
    const nextMissingMap: Record<string, number> = {};

    previewStats.forEach((stat) => {
      if (!stat || !stat.name) {
        return;
      }
      const name = String(stat.name).trim();
      if (!name) {
        return;
      }
      if (activeFlagSuffix && name.endsWith(activeFlagSuffix)) {
        return;
      }
      nextColumns.add(name);
      const numeric = Number(stat.missing_percentage);
      if (!Number.isNaN(numeric) && numeric >= 0) {
        nextMissingMap[name] = numeric;
      }
    });

    ensureArrayOfString(node?.data?.columns).forEach((column) => {
      if (column && !(activeFlagSuffix && column.endsWith(activeFlagSuffix))) {
        nextColumns.add(column);
      }
    });
    selectedColumns.forEach((column) => {
      if (column && !(activeFlagSuffix && column.endsWith(activeFlagSuffix))) {
        nextColumns.add(column);
      }
    });

    if (nextColumns.size === 0) {
      return;
    }

    const orderedColumns = Array.from(nextColumns).sort((a, b) => a.localeCompare(b));
    setAvailableColumns((previous) => (arraysAreEqual(previous, orderedColumns) ? previous : orderedColumns));

    if (Object.keys(nextMissingMap).length > 0) {
      setColumnMissingMap((previous) => ({ ...previous, ...nextMissingMap }));
    }
  }, [
    activeFlagSuffix,
    hasReachableSource,
    node?.data?.columns,
    previewState.data?.column_stats,
    previewState.data?.columns,
    requiresColumnCatalog,
    selectedColumns,
  ]);
  const normalizedColumnSearch = useMemo(() => columnSearch.trim().toLowerCase(), [columnSearch]);

  const filteredColumnOptions = useMemo(() => {
    if (!normalizedColumnSearch) {
      return availableColumns;
    }
    return availableColumns.filter((column) => column.toLowerCase().includes(normalizedColumnSearch));
  }, [availableColumns, normalizedColumnSearch]);

  const updateSkewnessTransformations = useCallback(
    (updater: (current: SkewnessTransformationConfig[]) => SkewnessTransformationConfig[]) => {
      setConfigState((previous) => {
        const currentTransformations = dedupeSkewnessTransformations(
          normalizeSkewnessTransformations(previous?.transformations),
        );
        const nextTransformations = dedupeSkewnessTransformations(updater(currentTransformations));
        return {
          ...previous,
          transformations: nextTransformations.map((entry) => ({ ...entry })),
        };
      });
    },
    []
  );


  const {
    skewnessThreshold,
    skewnessViewMode,
    setSkewnessViewMode,
    skewnessGroupByMethod,
    setSkewnessGroupByMethod,
    skewnessDistributionView,
    setSkewnessDistributionView,
    skewnessRecommendedCount,
    skewnessNumericCount,
    skewnessTransformationsCount,
    hasSkewnessAutoRecommendations,
    skewnessRows,
    skewnessTableGroups,
    skewnessDistributionCards,
    getSkewnessMethodLabel,
    getSkewnessMethodStatus,
    applySkewnessRecommendations: handleApplySkewnessRecommendations,
    handleSkewnessOverrideChange,
    clearSkewnessTransformations,
  } = useSkewnessConfiguration({
    skewnessData,
    shouldLoadInsights: shouldLoadSkewnessInsights,
    skewnessTransformations,
    availableColumns,
    previewColumns,
    columnTypeMap,
    updateSkewnessTransformations,
  });

  const {
    imputerColumnOptions,
    imputerMissingSliderMax,
    imputerFilteredOptionCount,
    imputerMissingFilterActive,
  } = useImputationConfiguration({
    isImputerNode,
    imputerStrategies,
    availableColumns,
    columnMissingMap,
    previewColumns,
    previewColumnStats,
    nodeColumns: node?.data?.columns,
    imputerMissingFilter,
  });

  const thresholdParameter = getParameter('missing_threshold');
  const thresholdParameterName = thresholdParameter?.name ?? null;

  const normalizedSuggestedThreshold = useMemo(() => {
    if (suggestedThreshold === null || Number.isNaN(Number(suggestedThreshold))) {
      return null;
    }
    const numeric = Number(suggestedThreshold);
    return Math.round(numeric * 10) / 10;
  }, [suggestedThreshold]);

  const thresholdMatchesSuggestion = useMemo(() => {
    if (thresholdParameterName === null || normalizedSuggestedThreshold === null) {
      return false;
    }
    const currentValue = Number(configState?.[thresholdParameterName]);
    return !Number.isNaN(currentValue) && currentValue === normalizedSuggestedThreshold;
  }, [configState, normalizedSuggestedThreshold, thresholdParameterName]);

  const showDropMissingRowsSection =
    isDropMissingRowsNode && Boolean(thresholdParameter || dropRowsAnyParameter);

  const handleManualBoundChange = useCallback(
    (column: string, bound: 'lower' | 'upper', rawValue: string) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous) => {
        const source =
          previous.manual_bounds &&
          typeof previous.manual_bounds === 'object' &&
          !Array.isArray(previous.manual_bounds)
            ? { ...previous.manual_bounds }
            : {};
        const existing = source[normalizedColumn] && typeof source[normalizedColumn] === 'object'
          ? { ...source[normalizedColumn] }
          : { lower: null, upper: null };

        if (rawValue === '') {
          if (bound === 'lower') {
            existing.lower = null;
          } else {
            existing.upper = null;
          }
        } else {
          const numeric = Number(rawValue);
          if (!Number.isFinite(numeric)) {
            return previous;
          }
          if (bound === 'lower') {
            existing.lower = numeric;
          } else {
            existing.upper = numeric;
          }
        }

        const normalizedLower = existing.lower ?? null;
        const normalizedUpper = existing.upper ?? null;

        if (normalizedLower === null && normalizedUpper === null) {
          delete source[normalizedColumn];
        } else {
          source[normalizedColumn] = {
            lower: normalizedLower,
            upper: normalizedUpper,
          };
        }

        return {
          ...previous,
          manual_bounds: source,
        };
      });
    },
    []
  );

  const handleClearManualBound = useCallback((column: string) => {
    const normalizedColumn = String(column ?? '').trim();
    if (!normalizedColumn) {
      return;
    }
    setConfigState((previous) => {
      const source =
        previous.manual_bounds &&
        typeof previous.manual_bounds === 'object' &&
        !Array.isArray(previous.manual_bounds)
          ? previous.manual_bounds
          : null;
      if (!source || !Object.prototype.hasOwnProperty.call(source, normalizedColumn)) {
        return previous;
      }
      const nextBounds = { ...source };
      delete nextBounds[normalizedColumn];
      return {
        ...previous,
        manual_bounds: nextBounds,
      };
    });
  }, []);

  const handleBinningIntegerChange = useCallback(
    (
      field: 'equal_width_bins' | 'equal_frequency_bins' | 'precision' | 'kbins_n_bins',
      rawValue: string,
      min: number,
      max: number
    ) => {
      setConfigState((previous) => {
        const previousValue =
          typeof previous?.[field] === 'number' && Number.isFinite(previous[field]) ? previous[field] : undefined;
        if (rawValue === '') {
          if (previousValue === undefined) {
            return previous;
          }
          const next: Record<string, any> = { ...previous };
          delete next[field];
          return next;
        }
        const numeric = Number(rawValue);
        if (!Number.isFinite(numeric)) {
          return previous;
        }
        const clamped = Math.min(Math.max(Math.round(numeric), min), max);
        if (previousValue === clamped) {
          return previous;
        }
        return {
          ...previous,
          [field]: clamped,
        };
      });
    },
    []
  );

  const handleBinningBooleanToggle = useCallback((field: 'include_lowest' | 'drop_original', value: boolean) => {
    setConfigState((previous) => {
      const current = Boolean(previous?.[field]);
      if (current === value) {
        return previous;
      }
      return {
        ...previous,
        [field]: value,
      };
    });
  }, []);

  const handleBinningSuffixChange = useCallback((value: string) => {
    const trimmed = value.trim();
    setConfigState((previous) => {
      const current = typeof previous?.output_suffix === 'string' ? previous.output_suffix : '';
      if (!trimmed) {
        if (!current) {
          return previous;
        }
        const next: Record<string, any> = { ...previous };
        delete next.output_suffix;
        return next;
      }
      if (current === trimmed) {
        return previous;
      }
      return {
        ...previous,
        output_suffix: trimmed,
      };
    });
  }, []);

  const handleBinningLabelFormatChange = useCallback((value: BinningLabelFormat) => {
    setConfigState((previous) => {
      const raw = typeof previous?.label_format === 'string' ? previous.label_format : '';
      const current = (['range', 'bin_index', 'ordinal', 'column_suffix'] as BinningLabelFormat[]).includes(
        raw as BinningLabelFormat,
      )
        ? (raw as BinningLabelFormat)
        : 'range';
      if (current === value) {
        return previous;
      }
      return {
        ...previous,
        label_format: value,
      };
    });
  }, []);

  const handleBinningMissingStrategyChange = useCallback((value: BinningMissingStrategy) => {
    setConfigState((previous) => {
      const current = previous?.missing_strategy === 'label' ? 'label' : 'keep';
      if (current === value) {
        return previous;
      }
      const next: Record<string, any> = {
        ...previous,
        missing_strategy: value,
      };
      if (value !== 'label' && Object.prototype.hasOwnProperty.call(next, 'missing_label')) {
        delete next.missing_label;
      }
      return next;
    });
  }, []);

  const handleBinningMissingLabelChange = useCallback((value: string) => {
    const trimmed = value.trim();
    setConfigState((previous) => {
      const current = typeof previous?.missing_label === 'string' ? previous.missing_label : '';
      if (!trimmed) {
        if (!current) {
          return previous;
        }
        const next: Record<string, any> = { ...previous };
        delete next.missing_label;
        return next;
      }
      if (current === trimmed) {
        return previous;
      }
      return {
        ...previous,
        missing_label: trimmed,
      };
    });
  }, []);

  const handleBinningCustomBinsChange = useCallback((column: string, rawValue: string) => {
    const normalizedColumn = String(column ?? '').trim();
    if (!normalizedColumn) {
      return;
    }
    setBinningCustomEdgeDrafts((previous) => ({
      ...previous,
      [normalizedColumn]: rawValue,
    }));
    setConfigState((previous) => {
      const existingRaw = previous?.custom_bins;
      const existing =
        existingRaw && typeof existingRaw === 'object' && !Array.isArray(existingRaw)
          ? (existingRaw as Record<string, number[]>)
          : {};
      const existingEntry = Array.isArray(existing[normalizedColumn]) ? existing[normalizedColumn] : undefined;
      const parsed = rawValue
        .split(/[,\n]/)
        .map((segment) => segment.trim())
        .filter(Boolean)
        .map((segment) => Number(segment))
        .filter((value) => Number.isFinite(value))
        .sort((a, b) => a - b);
      const unique = parsed.filter((value, index, array) => index === 0 || value !== array[index - 1]);
      const hasValidEntry = unique.length >= 2;
      if (hasValidEntry && existingEntry) {
        if (existingEntry.length === unique.length && existingEntry.every((value, index) => value === unique[index])) {
          return previous;
        }
      }
      if (!hasValidEntry && !existingEntry) {
        return previous;
      }
      const nextState: Record<string, any> = { ...previous };
      const nextBins: Record<string, number[]> = { ...existing };
      if (hasValidEntry) {
        nextBins[normalizedColumn] = unique;
      } else {
        delete nextBins[normalizedColumn];
      }
      if (Object.keys(nextBins).length > 0) {
        nextState.custom_bins = nextBins;
      } else {
        delete nextState.custom_bins;
      }
      return nextState;
    });
  }, []);

  const handleBinningCustomLabelsChange = useCallback((column: string, rawValue: string) => {
    const normalizedColumn = String(column ?? '').trim();
    if (!normalizedColumn) {
      return;
    }
    setBinningCustomLabelDrafts((previous) => ({
      ...previous,
      [normalizedColumn]: rawValue,
    }));
    setConfigState((previous) => {
      const existingRaw = previous?.custom_labels;
      const existing =
        existingRaw && typeof existingRaw === 'object' && !Array.isArray(existingRaw)
          ? (existingRaw as Record<string, string[]>)
          : {};
      const existingEntry = Array.isArray(existing[normalizedColumn]) ? existing[normalizedColumn] : undefined;
      const parsed = rawValue
        .split(/[,\n]/)
        .map((segment) => segment.trim())
        .filter(Boolean);
      if (parsed.length > 0 && existingEntry) {
        if (existingEntry.length === parsed.length && existingEntry.every((value, index) => value === parsed[index])) {
          return previous;
        }
      }
      if (parsed.length === 0 && !existingEntry) {
        return previous;
      }
      const nextState: Record<string, any> = { ...previous };
      const nextLabels: Record<string, string[]> = { ...existing };
      if (parsed.length > 0) {
        nextLabels[normalizedColumn] = parsed;
      } else {
        delete nextLabels[normalizedColumn];
      }
      if (Object.keys(nextLabels).length > 0) {
        nextState.custom_labels = nextLabels;
      } else {
        delete nextState.custom_labels;
      }
      return nextState;
    });
  }, []);

  const handleBinningClearCustomColumn = useCallback((column: string) => {
    const normalizedColumn = String(column ?? '').trim();
    if (!normalizedColumn) {
      return;
    }
    setBinningCustomEdgeDrafts((previous) => {
      if (!Object.prototype.hasOwnProperty.call(previous, normalizedColumn)) {
        return previous;
      }
      const next = { ...previous };
      delete next[normalizedColumn];
      return next;
    });
    setBinningCustomLabelDrafts((previous) => {
      if (!Object.prototype.hasOwnProperty.call(previous, normalizedColumn)) {
        return previous;
      }
      const next = { ...previous };
      delete next[normalizedColumn];
      return next;
    });
    setConfigState((previous) => {
      let changed = false;
      const nextState: Record<string, any> = { ...previous };
      if (
        previous?.custom_bins &&
        typeof previous.custom_bins === 'object' &&
        !Array.isArray(previous.custom_bins) &&
        Object.prototype.hasOwnProperty.call(previous.custom_bins, normalizedColumn)
      ) {
        const nextBins = { ...(previous.custom_bins as Record<string, number[]>) };
        delete nextBins[normalizedColumn];
        if (Object.keys(nextBins).length > 0) {
          nextState.custom_bins = nextBins;
        } else {
          delete nextState.custom_bins;
        }
        changed = true;
      }
      if (
        previous?.custom_labels &&
        typeof previous.custom_labels === 'object' &&
        !Array.isArray(previous.custom_labels) &&
        Object.prototype.hasOwnProperty.call(previous.custom_labels, normalizedColumn)
      ) {
        const nextLabels = { ...(previous.custom_labels as Record<string, string[]>) };
        delete nextLabels[normalizedColumn];
        if (Object.keys(nextLabels).length > 0) {
          nextState.custom_labels = nextLabels;
        } else {
          delete nextState.custom_labels;
        }
        changed = true;
      }
      if (!changed) {
        return previous;
      }
      return nextState;
    });
  }, []);

  const updateBinningColumnOverride = useCallback(
    (
      column: string,
      mutator: (current: Record<string, any>) => Record<string, any> | null,
    ) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous) => {
        const sourceOverrides = (() => {
          if (
            previous?.column_strategies &&
            typeof previous.column_strategies === 'object' &&
            !Array.isArray(previous.column_strategies)
          ) {
            return previous.column_strategies as Record<string, any>;
          }
          if (
            previous?.column_overrides &&
            typeof previous.column_overrides === 'object' &&
            !Array.isArray(previous.column_overrides)
          ) {
            return previous.column_overrides as Record<string, any>;
          }
          return {};
        })();

        const workingOverrides: Record<string, any> = { ...sourceOverrides };
        const rawExisting = workingOverrides[normalizedColumn];
        const existingOverride =
          rawExisting && typeof rawExisting === 'object' && !Array.isArray(rawExisting)
            ? { ...rawExisting }
            : {};

        const mutated = mutator(existingOverride);
        const hasOverride = Boolean(mutated && Object.keys(mutated).length);

        if (hasOverride) {
          workingOverrides[normalizedColumn] = mutated as Record<string, any>;
        } else {
          delete workingOverrides[normalizedColumn];
        }

        const nextState: Record<string, any> = { ...previous };

        if (hasOverride) {
          const currentColumns = new Set(ensureArrayOfString(previous?.columns));
          if (!currentColumns.has(normalizedColumn)) {
            currentColumns.add(normalizedColumn);
            nextState.columns = Array.from(currentColumns).sort((a, b) => a.localeCompare(b));
          }
        }

        if (Object.keys(workingOverrides).length) {
          nextState.column_overrides = workingOverrides;
          nextState.column_strategies = workingOverrides;
        } else {
          if (Object.prototype.hasOwnProperty.call(nextState, 'column_overrides')) {
            delete nextState.column_overrides;
          }
          if (Object.prototype.hasOwnProperty.call(nextState, 'column_strategies')) {
            delete nextState.column_strategies;
          }
        }

        return nextState;
      });
    },
    [setConfigState],
  );

  const handleBinningOverrideStrategyChange = useCallback(
    (
      column: string,
      value: BinningStrategy | '__default__',
      options?: { recommendedBins?: number | null },
    ) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }

      if (value === '__default__') {
        handleBinningClearCustomColumn(normalizedColumn);
        updateBinningColumnOverride(normalizedColumn, () => null);
        return;
      }

      const recommendedBins = Number.isFinite(options?.recommendedBins)
        ? Math.round(options!.recommendedBins as number)
        : null;

      updateBinningColumnOverride(normalizedColumn, (current) => {
        const next = { ...current } as Record<string, any>;

        next.strategy = value;

        const resolveBins = (fallback: number) => {
          const candidate = recommendedBins ?? fallback;
          return Math.min(200, Math.max(2, Math.round(candidate)));
        };

        if (value === 'equal_width') {
          next.equal_width_bins = resolveBins(binningConfig.equalWidthBins);
          delete next.equal_frequency_bins;
          delete next.kbins_n_bins;
          delete next.kbins_encode;
          delete next.kbins_strategy;
        } else if (value === 'equal_frequency') {
          next.equal_frequency_bins = resolveBins(binningConfig.equalFrequencyBins);
          delete next.equal_width_bins;
          delete next.kbins_n_bins;
          delete next.kbins_encode;
          delete next.kbins_strategy;
        } else if (value === 'kbins') {
          next.kbins_n_bins = resolveBins(binningConfig.kbinsNBins);
          next.kbins_encode = binningConfig.kbinsEncode;
          next.kbins_strategy = binningConfig.kbinsStrategy;
          delete next.equal_width_bins;
          delete next.equal_frequency_bins;
        } else if (value === 'custom') {
          delete next.equal_width_bins;
          delete next.equal_frequency_bins;
          delete next.kbins_n_bins;
          delete next.kbins_encode;
          delete next.kbins_strategy;
        }

        if (!Object.keys(next).length) {
          return null;
        }
        return next;
      });

      if (value !== 'custom') {
        handleBinningClearCustomColumn(normalizedColumn);
      }
    },
    [
      binningConfig.equalFrequencyBins,
      binningConfig.equalWidthBins,
      binningConfig.kbinsEncode,
      binningConfig.kbinsNBins,
      binningConfig.kbinsStrategy,
      handleBinningClearCustomColumn,
      updateBinningColumnOverride,
    ],
  );

  const handleBinningOverrideNumberChange = useCallback(
    (column: string, field: 'equal_width_bins' | 'equal_frequency_bins' | 'kbins_n_bins', rawValue: string) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }

      updateBinningColumnOverride(normalizedColumn, (current) => {
        const next = { ...current } as Record<string, any>;

        if (!rawValue.trim()) {
          delete next[field];
        } else {
          const numeric = Number(rawValue);
          if (!Number.isFinite(numeric)) {
            return current;
          }
          const clamped = Math.min(200, Math.max(2, Math.round(numeric)));
          next[field] = clamped;
          if (field === 'equal_width_bins') {
            next.strategy = 'equal_width';
            delete next.equal_frequency_bins;
            delete next.kbins_n_bins;
            delete next.kbins_encode;
            delete next.kbins_strategy;
          } else if (field === 'equal_frequency_bins') {
            next.strategy = 'equal_frequency';
            delete next.equal_width_bins;
            delete next.kbins_n_bins;
            delete next.kbins_encode;
            delete next.kbins_strategy;
          } else if (field === 'kbins_n_bins') {
            next.strategy = 'kbins';
            delete next.equal_width_bins;
            delete next.equal_frequency_bins;
          }
        }

        if (!Object.keys(next).length) {
          return null;
        }
        return next;
      });

      handleBinningClearCustomColumn(normalizedColumn);
    },
    [handleBinningClearCustomColumn, updateBinningColumnOverride],
  );

  const handleBinningOverrideKbinsEncodeChange = useCallback(
    (column: string, value: KBinsEncode | null) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      updateBinningColumnOverride(normalizedColumn, (current) => {
        const next = { ...current } as Record<string, any>;
        if (value) {
          next.strategy = 'kbins';
          next.kbins_encode = value;
        } else {
          delete next.kbins_encode;
        }
        if (!Object.keys(next).length) {
          return null;
        }
        return next;
      });
    },
    [updateBinningColumnOverride],
  );

  const handleBinningOverrideKbinsStrategyChange = useCallback(
    (column: string, value: KBinsStrategy | null) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      updateBinningColumnOverride(normalizedColumn, (current) => {
        const next = { ...current } as Record<string, any>;
        if (value) {
          next.strategy = 'kbins';
          next.kbins_strategy = value;
        } else {
          delete next.kbins_strategy;
        }
        if (!Object.keys(next).length) {
          return null;
        }
        return next;
      });
    },
    [updateBinningColumnOverride],
  );

  const handleBinningClearOverride = useCallback(
    (column: string) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      handleBinningClearCustomColumn(normalizedColumn);
      updateBinningColumnOverride(normalizedColumn, () => null);
    },
    [handleBinningClearCustomColumn, updateBinningColumnOverride],
  );

  const handleBinningClearOverrides = useCallback(() => {
    setConfigState((previous) => {
      const hasOverrides =
        (previous?.column_strategies &&
          typeof previous.column_strategies === 'object' &&
          !Array.isArray(previous.column_strategies) &&
          Object.keys(previous.column_strategies).length > 0) ||
        (previous?.column_overrides &&
          typeof previous.column_overrides === 'object' &&
          !Array.isArray(previous.column_overrides) &&
          Object.keys(previous.column_overrides).length > 0);

      if (!hasOverrides) {
        return previous;
      }

      const nextState = { ...previous } as Record<string, any>;
      if (Object.prototype.hasOwnProperty.call(nextState, 'column_overrides')) {
        delete nextState.column_overrides;
      }
      if (Object.prototype.hasOwnProperty.call(nextState, 'column_strategies')) {
        delete nextState.column_strategies;
      }
      return nextState;
    });
  }, [setConfigState]);

  const handleBinningApplyStrategies = useCallback(() => {
    if (!binningInsightsRecommendations.length) {
      return;
    }
    binningInsightsRecommendations.forEach((entry) => {
      const columnName = typeof entry?.column === 'string' ? entry.column.trim() : '';
      if (!columnName) {
        return;
      }
      const strategy = (entry.recommended_strategy as BinningStrategy) ?? 'equal_width';
      handleBinningOverrideStrategyChange(columnName, strategy, {
        recommendedBins: Number.isFinite(entry?.recommended_bins) ? entry.recommended_bins : null,
      });
    });
  }, [binningInsightsRecommendations, handleBinningOverrideStrategyChange]);

  const handleBinningApplyColumns = useCallback((columns: Iterable<string>) => {
    if (!columns) {
      return;
    }
    const normalizedColumns: string[] = [];
    for (const column of columns) {
      const name = typeof column === 'string' ? column.trim() : '';
      if (!name) {
        continue;
      }
      normalizedColumns.push(name);
    }
    if (!normalizedColumns.length) {
      return;
    }
    setConfigState((previous) => {
      const base = previous && typeof previous === 'object' ? previous : {};
      const currentColumns = new Set(ensureArrayOfString(base.columns));
      let changed = false;
      normalizedColumns.forEach((column) => {
        if (currentColumns.has(column)) {
          return;
        }
        currentColumns.add(column);
        changed = true;
      });
      if (!changed) {
        return previous;
      }
      const nextState: Record<string, any> = { ...base };
      nextState.columns = Array.from(currentColumns).sort((a, b) => a.localeCompare(b));
      return nextState;
    });
  }, []);

  const handleApplyAllBinningNumeric = useCallback(() => {
    if (!binningAllNumericColumns.length) {
      return;
    }
    handleBinningApplyColumns(binningAllNumericColumns);
  }, [binningAllNumericColumns, handleBinningApplyColumns]);

  const handleToggleColumn = useCallback(
    (column: string) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous) => {
        const current = ensureArrayOfString(previous.columns);
        const exists = current.includes(normalizedColumn);
        const nextColumns = exists
          ? current.filter((item) => item !== normalizedColumn)
          : [...current, normalizedColumn];

        let manualChanged = false;
        let manualBounds = previous.manual_bounds;
        if (
          exists &&
          manualBounds &&
          typeof manualBounds === 'object' &&
          !Array.isArray(manualBounds) &&
          Object.prototype.hasOwnProperty.call(manualBounds, normalizedColumn)
        ) {
          manualBounds = { ...manualBounds };
          delete manualBounds[normalizedColumn];
          manualChanged = true;
        }

        const result: Record<string, any> = {
          ...previous,
          columns: nextColumns,
        };

        if (manualChanged) {
          result.manual_bounds = manualBounds;
        }

        if (isBinningNode) {
          const selectedSet = new Set(nextColumns);

          if (
            previous.custom_bins &&
            typeof previous.custom_bins === 'object' &&
            !Array.isArray(previous.custom_bins)
          ) {
            const nextBins: Record<string, number[]> = {};
            Object.entries(previous.custom_bins).forEach(([key, value]) => {
              if (selectedSet.has(key) && Array.isArray(value) && value.length) {
                nextBins[key] = [...value];
              }
            });
            if (Object.keys(nextBins).length) {
              result.custom_bins = nextBins;
            } else if ('custom_bins' in result) {
              delete result.custom_bins;
            }
          } else if ('custom_bins' in result) {
            delete result.custom_bins;
          }

          if (
            previous.custom_labels &&
            typeof previous.custom_labels === 'object' &&
            !Array.isArray(previous.custom_labels)
          ) {
            const nextLabels: Record<string, string[]> = {};
            Object.entries(previous.custom_labels).forEach(([key, value]) => {
              if (selectedSet.has(key) && Array.isArray(value) && value.length) {
                nextLabels[key] = value.map((entry) => String(entry));
              }
            });
            if (Object.keys(nextLabels).length) {
              result.custom_labels = nextLabels;
            } else if ('custom_labels' in result) {
              delete result.custom_labels;
            }
          } else if ('custom_labels' in result) {
            delete result.custom_labels;
          }

          if (
            previous.column_overrides &&
            typeof previous.column_overrides === 'object' &&
            !Array.isArray(previous.column_overrides)
          ) {
            const nextOverrides: Record<string, any> = {};
            Object.entries(previous.column_overrides as Record<string, any>).forEach(([key, value]) => {
              if (selectedSet.has(key) && value && typeof value === 'object' && !Array.isArray(value)) {
                nextOverrides[key] = { ...(value as Record<string, any>) };
              }
            });
            if (Object.keys(nextOverrides).length) {
              result.column_overrides = nextOverrides;
              result.column_strategies = nextOverrides;
            } else {
              if ('column_overrides' in result) {
                delete result.column_overrides;
              }
              if ('column_strategies' in result) {
                delete result.column_strategies;
              }
            }
          } else {
            if ('column_overrides' in result) {
              delete result.column_overrides;
            }
            if ('column_strategies' in result) {
              delete result.column_strategies;
            }
          }
        }

        if (isScalingNode) {
          let nextMethods: Record<string, ScalingMethodName> | null = null;
          if (
            previous.column_methods &&
            typeof previous.column_methods === 'object' &&
            !Array.isArray(previous.column_methods)
          ) {
            nextMethods = {};
            Object.entries(previous.column_methods as Record<string, any>).forEach(([key, value]) => {
              const methodKey = typeof value === 'string' ? (value.trim() as ScalingMethodName) : null;
              if (!methodKey || !SCALING_METHOD_ORDER.includes(methodKey)) {
                return;
              }
              if (exists && key === normalizedColumn) {
                return;
              }
              nextMethods![key] = methodKey;
            });
          }

          if (nextMethods && Object.keys(nextMethods).length) {
            result.column_methods = nextMethods;
          } else if (exists && Object.prototype.hasOwnProperty.call(result, 'column_methods')) {
            delete result.column_methods;
          }
        }

        return result;
      });
    },
    [isBinningNode, isScalingNode]
  );

  const handleApplyAllRecommended = useCallback(() => {
    if (!recommendations.length) {
      return;
    }
    setConfigState((previous) => {
      const next = new Set(ensureArrayOfString(previous.columns));
      recommendations.forEach((candidate) => {
        if (candidate?.name) {
          next.add(String(candidate.name));
        }
      });
      return {
        ...previous,
        columns: Array.from(next),
      };
    });
  }, [recommendations]);

  const handleApplyLabelEncodingRecommended = useCallback((columns: string[]) => {
    if (!columns.length) {
      return;
    }
    setConfigState((previous) => {
      const next = new Set(ensureArrayOfString(previous.columns));
      columns.forEach((column) => {
        const normalized = String(column ?? '').trim();
        if (normalized) {
          next.add(normalized);
        }
      });
      const ordered = Array.from(next).sort((a, b) => a.localeCompare(b));
      return {
        ...previous,
        columns: ordered,
      };
    });
  }, []);

  const handleApplyTargetEncodingRecommended = useCallback(
    (columns: string[]) => {
      if (!columns.length) {
        return;
      }

      setConfigState((previous) => {
        const next = new Set(ensureArrayOfString(previous.columns));
        let shouldEnableFallback = false;

        columns.forEach((column) => {
          const normalized = String(column ?? '').trim();
          if (!normalized) {
            return;
          }
          next.add(normalized);
          if (!shouldEnableFallback) {
            const suggestion = targetEncodingSuggestions.find((entry) => entry.column === normalized);
            if (suggestion?.recommended_use_global_fallback) {
              shouldEnableFallback = true;
            }
          }
        });

        const ordered = Array.from(next).sort((a, b) => a.localeCompare(b));
        const result: Record<string, any> = {
          ...previous,
          columns: ordered,
        };

        if (shouldEnableFallback) {
          result.encode_missing = true;
          result.handle_unknown = 'global_mean';
        }

        return result;
      });
    },
    [setConfigState, targetEncodingSuggestions],
  );

  const handleApplyHashEncodingRecommended = useCallback(
    (columns: string[]) => {
      if (!columns.length) {
        return;
      }

      setConfigState((previous) => {
        const next = new Set(ensureArrayOfString(previous.columns));
        let recommendedBuckets = 0;

        columns.forEach((column) => {
          const normalized = String(column ?? '').trim();
          if (!normalized) {
            return;
          }
          next.add(normalized);
          const suggestion = hashEncodingSuggestions.find((entry) => entry.column === normalized);
          if (suggestion && suggestion.recommended_bucket_count > recommendedBuckets) {
            recommendedBuckets = suggestion.recommended_bucket_count;
          }
        });

        const ordered = Array.from(next).sort((a, b) => a.localeCompare(b));
        const result: Record<string, any> = {
          ...previous,
          columns: ordered,
        };

        if (recommendedBuckets > 0) {
          const currentBuckets = Number(previous?.n_buckets);
          if (!Number.isFinite(currentBuckets) || currentBuckets <= 0) {
            result.n_buckets = recommendedBuckets;
          }
        }

        return result;
      });
    },
    [hashEncodingSuggestions],
  );

  const handleApplyOrdinalEncodingRecommended = useCallback(
    (columns: string[]) => {
      if (!columns.length) {
        return;
      }
      setConfigState((previous) => {
        const next = new Set(ensureArrayOfString(previous.columns));
        let shouldEnableUnknown = false;
        let shouldEncodeMissing = false;

        columns.forEach((column) => {
          const normalized = String(column ?? '').trim();
          if (!normalized) {
            return;
          }
          next.add(normalized);
          if (!shouldEnableUnknown || !shouldEncodeMissing) {
            const suggestion = ordinalEncodingSuggestions.find((entry) => entry.column === normalized);
            if (suggestion?.recommended_handle_unknown) {
              shouldEnableUnknown = true;
            }
            if ((suggestion?.missing_percentage ?? 0) > 0) {
              shouldEncodeMissing = true;
            }
          }
        });

        const ordered = Array.from(next).sort((a, b) => a.localeCompare(b));
        const result: Record<string, any> = {
          ...previous,
          columns: ordered,
        };

        if (shouldEnableUnknown) {
          result.handle_unknown = 'use_encoded_value';
        }
        if (shouldEncodeMissing) {
          result.encode_missing = true;
        }

        return result;
      });
    },
    [ordinalEncodingSuggestions],
  );

  const handleApplyDummyEncodingRecommended = useCallback(
    (columns: string[]) => {
      if (!columns.length) {
        return;
      }
      setConfigState((previous) => {
        const next = new Set(ensureArrayOfString(previous.columns));
        let enforceDropFirst = previous?.drop_first === true;

        columns.forEach((column) => {
          const normalized = String(column ?? '').trim();
          if (!normalized) {
            return;
          }
          next.add(normalized);
          if (!enforceDropFirst) {
            const suggestion = dummyEncodingSuggestions.find((entry) => entry.column === normalized);
            if (suggestion?.recommended_drop_first) {
              enforceDropFirst = true;
            }
          }
        });

        const ordered = Array.from(next).sort((a, b) => a.localeCompare(b));
        const result: Record<string, any> = {
          ...previous,
          columns: ordered,
        };

        if (enforceDropFirst) {
          result.drop_first = true;
        }

        return result;
      });
    },
    [dummyEncodingSuggestions]
  );

  const handleApplyOneHotEncodingRecommended = useCallback(
    (columns: string[]) => {
      if (!columns.length) {
        return;
      }
      setConfigState((previous) => {
        const next = new Set(ensureArrayOfString(previous.columns));
        let shouldEnableDropFirst = false;

        columns.forEach((column) => {
          const normalized = String(column ?? '').trim();
          if (!normalized) {
            return;
          }
          next.add(normalized);
          if (!shouldEnableDropFirst) {
            const suggestion = oneHotEncodingSuggestions.find((entry) => entry.column === normalized);
            if (suggestion?.recommended_drop_first) {
              shouldEnableDropFirst = true;
            }
          }
        });

        const ordered = Array.from(next).sort((a, b) => a.localeCompare(b));
        const result: Record<string, any> = {
          ...previous,
          columns: ordered,
        };

        if (shouldEnableDropFirst) {
          result.drop_first = true;
        }

        return result;
      });
    },
    [oneHotEncodingSuggestions]
  );

  const handleSelectAllColumns = useCallback(() => {
    if (!availableColumns.length) {
      return;
    }
    setConfigState((previous) => {
      const aggregate = new Set(ensureArrayOfString(previous.columns));
      const eligibleColumns = availableColumns.filter((column) => {
        if (isBinningNode) {
          return !binningExcludedColumns.has(column);
        }
        if (isScalingNode) {
          return !scalingExcludedColumns.has(column);
        }
        return true;
      });
      eligibleColumns.forEach((column) => aggregate.add(column));
      return {
        ...previous,
        columns: Array.from(aggregate).sort((a, b) => a.localeCompare(b)),
      };
    });
  }, [availableColumns, binningExcludedColumns, isBinningNode, isScalingNode, scalingExcludedColumns]);

  const handleOutlierDefaultMethodChange = useCallback(
    (method: OutlierMethodName) => {
      if (!OUTLIER_METHOD_ORDER.includes(method)) {
        return;
      }
      setConfigState((previous) => {
        if (previous?.default_method === method) {
          return previous;
        }
        return {
          ...previous,
          default_method: method,
        };
      });
    },
    [setConfigState]
  );

  const handleOutlierAutoDetectToggle = useCallback(
    (value: boolean) => {
      setConfigState((previous) => {
        const current = typeof previous?.auto_detect === 'boolean' ? previous.auto_detect : true;
        if (current === value) {
          return previous;
        }
        return {
          ...previous,
          auto_detect: value,
        };
      });
    },
    [setConfigState]
  );

  const setOutlierColumnMethod = useCallback(
    (column: string, method: OutlierMethodName | null) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }

      setConfigState((previous) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const currentColumns = ensureArrayOfString(previous.columns);
        const columnSet = new Set(currentColumns);

        const existingMethods =
          previous.column_methods &&
          typeof previous.column_methods === 'object' &&
          !Array.isArray(previous.column_methods)
            ? { ...(previous.column_methods as Record<string, OutlierMethodName>) }
            : {};

        let changed = false;

        if (method) {
          if (!OUTLIER_METHOD_ORDER.includes(method)) {
            return previous;
          }
          if (existingMethods[normalized] !== method) {
            existingMethods[normalized] = method;
            changed = true;
          }
          if (!columnSet.has(normalized)) {
            columnSet.add(normalized);
            changed = true;
          }
        } else if (Object.prototype.hasOwnProperty.call(existingMethods, normalized)) {
          delete existingMethods[normalized];
          changed = true;
        }

        if (!changed) {
          return previous;
        }

        const nextState: Record<string, any> = {
          ...previous,
        };

        nextState.columns = Array.from(columnSet).sort((a, b) => a.localeCompare(b));

        if (Object.keys(existingMethods).length) {
          nextState.column_methods = existingMethods;
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
          delete nextState.column_methods;
        }

        return nextState;
      });
    },
    [setConfigState]
  );

  const handleOutlierClearOverrides = useCallback(() => {
    setConfigState((previous) => {
      if (!previous || typeof previous !== 'object') {
        return previous;
      }

      const hasColumnMethods =
        previous.column_methods &&
        typeof previous.column_methods === 'object' &&
        !Array.isArray(previous.column_methods) &&
        Object.keys(previous.column_methods).length > 0;

      const hasColumnParameters =
        previous.column_parameters &&
        typeof previous.column_parameters === 'object' &&
        !Array.isArray(previous.column_parameters) &&
        Object.keys(previous.column_parameters).length > 0;

      if (!hasColumnMethods && !hasColumnParameters) {
        return previous;
      }

      const nextState: Record<string, any> = { ...previous };

      if (hasColumnMethods) {
        delete nextState.column_methods;
      }

      if (hasColumnParameters) {
        delete nextState.column_parameters;
      }

      return nextState;
    });
  }, [setConfigState]);

  const handleOutlierApplyAllRecommendations = useCallback(() => {
    if (!outlierRecommendationRows.length) {
      return;
    }

    setConfigState((previous) => {
      if (!previous || typeof previous !== 'object') {
        return previous;
      }

      const currentColumns = new Set(ensureArrayOfString(previous.columns));
      const skippedSet = new Set(ensureArrayOfString(previous.skipped_columns));
      const nextMethods =
        previous.column_methods &&
        typeof previous.column_methods === 'object' &&
        !Array.isArray(previous.column_methods)
          ? { ...(previous.column_methods as Record<string, OutlierMethodName>) }
          : {};
      const nextParameters =
        previous.column_parameters &&
        typeof previous.column_parameters === 'object' &&
        !Array.isArray(previous.column_parameters)
          ? { ...(previous.column_parameters as Record<string, Record<string, number>>) }
          : {};

      let updated = false;

      outlierRecommendationRows.forEach((row) => {
        if (!row || row.isExcluded || !row.recommendedMethod) {
          return;
        }

        const columnKey = String(row.column ?? '').trim();
        if (!columnKey || outlierExcludedColumns.has(columnKey)) {
          return;
        }

        const recommendedMethod = row.recommendedMethod;
        if (!OUTLIER_METHOD_ORDER.includes(recommendedMethod)) {
          return;
        }

        if (skippedSet.has(columnKey)) {
          skippedSet.delete(columnKey);
          updated = true;
        }

        if (!currentColumns.has(columnKey)) {
          currentColumns.add(columnKey);
          updated = true;
        }

        if (recommendedMethod === outlierConfig.defaultMethod) {
          if (Object.prototype.hasOwnProperty.call(nextMethods, columnKey)) {
            delete nextMethods[columnKey];
            updated = true;
          }
        } else if (nextMethods[columnKey] !== recommendedMethod) {
          nextMethods[columnKey] = recommendedMethod;
          updated = true;
        }

        if (Object.prototype.hasOwnProperty.call(nextParameters, columnKey)) {
          delete nextParameters[columnKey];
          updated = true;
        }
      });

      if (!updated) {
        return previous;
      }

      const nextState: Record<string, any> = {
        ...previous,
        columns: Array.from(currentColumns).sort((a, b) => a.localeCompare(b)),
      };

      if (skippedSet.size) {
        nextState.skipped_columns = Array.from(skippedSet).sort((a, b) => a.localeCompare(b));
      } else if (Object.prototype.hasOwnProperty.call(nextState, 'skipped_columns')) {
        delete nextState.skipped_columns;
      }

      if (Object.keys(nextMethods).length) {
        nextState.column_methods = nextMethods;
      } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
        delete nextState.column_methods;
      }

      if (Object.keys(nextParameters).length) {
        nextState.column_parameters = nextParameters;
      } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_parameters')) {
        delete nextState.column_parameters;
      }

      return nextState;
    });
  }, [outlierConfig.defaultMethod, outlierExcludedColumns, outlierRecommendationRows, setConfigState]);

  const handleOutlierSkipColumn = useCallback(
    (column: string) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }

      setConfigState((previous) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const currentColumns = ensureArrayOfString(previous.columns);
        const filteredColumns = currentColumns.filter((value) => value !== normalized);

        const skippedSet = new Set(ensureArrayOfString(previous.skipped_columns));
        const skipSize = skippedSet.size;
        skippedSet.add(normalized);
        const skipChanged = skipSize !== skippedSet.size;

        const existingMethods =
          previous.column_methods &&
          typeof previous.column_methods === 'object' &&
          !Array.isArray(previous.column_methods)
            ? { ...(previous.column_methods as Record<string, OutlierMethodName>) }
            : {};
        const hadMethod = Object.prototype.hasOwnProperty.call(existingMethods, normalized);
        if (hadMethod) {
          delete existingMethods[normalized];
        }

        const existingParameters =
          previous.column_parameters &&
          typeof previous.column_parameters === 'object' &&
          !Array.isArray(previous.column_parameters)
            ? { ...(previous.column_parameters as Record<string, Record<string, number>>) }
            : {};
        const hadParameters = Object.prototype.hasOwnProperty.call(existingParameters, normalized);
        if (hadParameters) {
          delete existingParameters[normalized];
        }

        const columnsChanged = filteredColumns.length !== currentColumns.length;

        if (!columnsChanged && !skipChanged && !hadMethod && !hadParameters) {
          return previous;
        }

        const nextState: Record<string, any> = {
          ...previous,
          columns: filteredColumns.sort((a, b) => a.localeCompare(b)),
          skipped_columns: Array.from(skippedSet).sort((a, b) => a.localeCompare(b)),
        };

        if (Object.keys(existingMethods).length) {
          nextState.column_methods = existingMethods;
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
          delete nextState.column_methods;
        }

        if (Object.keys(existingParameters).length) {
          nextState.column_parameters = existingParameters;
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_parameters')) {
          delete nextState.column_parameters;
        }

        return nextState;
      });
    },
    [setConfigState]
  );

  const handleOutlierUnskipColumn = useCallback(
    (column: string) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }

      setConfigState((previous) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const skipped = ensureArrayOfString(previous.skipped_columns);
        if (!skipped.includes(normalized)) {
          return previous;
        }

        const nextSkipped = skipped.filter((value) => value !== normalized);
        const currentColumns = new Set(ensureArrayOfString(previous.columns));
        if (!currentColumns.has(normalized)) {
          currentColumns.add(normalized);
        }

        const nextState: Record<string, any> = {
          ...previous,
          columns: Array.from(currentColumns).sort((a, b) => a.localeCompare(b)),
        };

        if (nextSkipped.length) {
          nextState.skipped_columns = nextSkipped.sort((a, b) => a.localeCompare(b));
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'skipped_columns')) {
          delete nextState.skipped_columns;
        }

        return nextState;
      });
    },
    [setConfigState]
  );

  const handleOutlierOverrideSelect = useCallback(
    (column: string, value: string) => {
      if (value === '__skip__') {
        handleOutlierSkipColumn(column);
        return;
      }

      if (value === '__default__') {
        handleOutlierUnskipColumn(column);
        setOutlierColumnMethod(column, null);
        return;
      }

      if (OUTLIER_METHOD_ORDER.includes(value as OutlierMethodName)) {
        handleOutlierUnskipColumn(column);
        setOutlierColumnMethod(column, value as OutlierMethodName);
      }
    },
    [handleOutlierSkipColumn, handleOutlierUnskipColumn, setOutlierColumnMethod]
  );

  const handleOutlierMethodParameterChange = useCallback(
    (method: OutlierMethodName, parameter: string, value: number | null) => {
      const normalizedParameter = String(parameter ?? '').trim().toLowerCase();

      setConfigState((previous) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const baseParameters: Partial<Record<OutlierMethodName, Record<string, number>>> =
          previous.method_parameters &&
          typeof previous.method_parameters === 'object' &&
          !Array.isArray(previous.method_parameters)
            ? { ...(previous.method_parameters as Record<OutlierMethodName, Record<string, number>>) }
            : {};

        const methodParameters = {
          ...(baseParameters[method] ?? {}),
        };

        const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : null;
        let changed = false;

        if (numericValue === null) {
          if (Object.prototype.hasOwnProperty.call(methodParameters, normalizedParameter)) {
            delete methodParameters[normalizedParameter];
            changed = true;
          }
        } else if (methodParameters[normalizedParameter] !== numericValue) {
          methodParameters[normalizedParameter] = numericValue;
          changed = true;
        }

        if (!changed) {
          return previous;
        }

        if (Object.keys(methodParameters).length) {
          baseParameters[method] = methodParameters;
        } else {
          delete baseParameters[method];
        }

        if (!Object.keys(baseParameters).length) {
          const nextState = { ...previous };
          if (Object.prototype.hasOwnProperty.call(nextState, 'method_parameters')) {
            delete nextState.method_parameters;
          }
          return nextState;
        }

        return {
          ...previous,
          method_parameters: baseParameters,
        };
      });
    },
    [setConfigState]
  );

  const handleOutlierColumnParameterChange = useCallback(
    (column: string, parameter: string, value: number | null) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      const normalizedParameter = String(parameter ?? '').trim().toLowerCase();

      setConfigState((previous) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const baseParameters =
          previous.column_parameters &&
          typeof previous.column_parameters === 'object' &&
          !Array.isArray(previous.column_parameters)
            ? { ...(previous.column_parameters as Record<string, Record<string, number>>) }
            : {};

        const columnParameters = {
          ...(baseParameters[normalizedColumn] ?? {}),
        };

        const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : null;
        let changed = false;

        if (numericValue === null) {
          if (Object.prototype.hasOwnProperty.call(columnParameters, normalizedParameter)) {
            delete columnParameters[normalizedParameter];
            changed = true;
          }
        } else if (columnParameters[normalizedParameter] !== numericValue) {
          columnParameters[normalizedParameter] = numericValue;
          changed = true;
        }

        if (!changed) {
          return previous;
        }

        if (Object.keys(columnParameters).length) {
          baseParameters[normalizedColumn] = columnParameters;
        } else {
          delete baseParameters[normalizedColumn];
        }

        if (!Object.keys(baseParameters).length) {
          const nextState = { ...previous };
          if (Object.prototype.hasOwnProperty.call(nextState, 'column_parameters')) {
            delete nextState.column_parameters;
          }
          return nextState;
        }

        return {
          ...previous,
          column_parameters: baseParameters,
        };
      });
    },
    [setConfigState]
  );

  const handleScalingDefaultMethodChange = useCallback(
    (method: ScalingMethodName) => {
      if (!SCALING_METHOD_ORDER.includes(method)) {
        return;
      }
      setConfigState((previous) => {
        if (previous?.default_method === method) {
          return previous;
        }
        return {
          ...previous,
          default_method: method,
        };
      });
    },
    [setConfigState]
  );

  const handleScalingAutoDetectToggle = useCallback(
    (value: boolean) => {
      setConfigState((previous) => {
        const current = typeof previous?.auto_detect === 'boolean' ? previous.auto_detect : true;
        if (current === value) {
          return previous;
        }
        return {
          ...previous,
          auto_detect: value,
        };
      });
    },
    [setConfigState]
  );

  const setScalingColumnMethod = useCallback(
    (column: string, method: ScalingMethodName | null) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }

      setConfigState((previous) => {
        const currentColumns = ensureArrayOfString(previous.columns);
        const hasColumn = currentColumns.includes(normalized);
        const existingMethods =
          previous.column_methods &&
          typeof previous.column_methods === 'object' &&
          !Array.isArray(previous.column_methods)
            ? { ...(previous.column_methods as Record<string, ScalingMethodName>) }
            : {};

        if (method) {
          if (!SCALING_METHOD_ORDER.includes(method)) {
            return previous;
          }
          const currentMethod = existingMethods[normalized] ?? null;
          const needsMethodUpdate = currentMethod !== method;
          const needsColumnUpdate = !hasColumn;
          if (!needsMethodUpdate && !needsColumnUpdate) {
            return previous;
          }
          existingMethods[normalized] = method;
          const nextColumns = needsColumnUpdate
            ? [...currentColumns, normalized].sort((a, b) => a.localeCompare(b))
            : currentColumns;
          return {
            ...previous,
            columns: nextColumns,
            column_methods: existingMethods,
          };
        }

        if (!Object.prototype.hasOwnProperty.call(existingMethods, normalized)) {
          return previous;
        }

        delete existingMethods[normalized];
        if (Object.keys(existingMethods).length) {
          return {
            ...previous,
            column_methods: existingMethods,
          };
        }

        const nextState: Record<string, any> = {
          ...previous,
        };
        if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
          delete nextState.column_methods;
        }
        return nextState;
      });
    },
    [setConfigState]
  );

  const handleScalingClearOverrides = useCallback(() => {
    setConfigState((previous) => {
      if (
        !previous.column_methods ||
        typeof previous.column_methods !== 'object' ||
        Array.isArray(previous.column_methods) ||
        !Object.keys(previous.column_methods).length
      ) {
        return previous;
      }
      const nextState = { ...previous } as Record<string, any>;
      delete nextState.column_methods;
      return nextState;
    });
  }, [setConfigState]);

  const handleScalingApplyAllRecommendations = useCallback(() => {
    if (!scalingRecommendations.length) {
      return;
    }
    setConfigState((previous) => {
      const currentColumns = ensureArrayOfString(previous.columns);
      const columnSet = new Set(currentColumns);
      const skippedSet = new Set(ensureArrayOfString(previous.skipped_columns));
      const nextMethods =
        previous.column_methods &&
        typeof previous.column_methods === 'object' &&
        !Array.isArray(previous.column_methods)
          ? { ...(previous.column_methods as Record<string, ScalingMethodName>) }
          : {};

      let updated = false;

      scalingRecommendations.forEach((entry) => {
        const normalized = String(entry.column ?? '').trim();
        const method = entry.recommended_method;
        if (!normalized || !SCALING_METHOD_ORDER.includes(method)) {
          return;
        }
        if (skippedSet.has(normalized)) {
          return;
        }
        if (nextMethods[normalized] !== method) {
          nextMethods[normalized] = method;
          updated = true;
        }
        if (!columnSet.has(normalized)) {
          columnSet.add(normalized);
          updated = true;
        }
      });

      if (!updated) {
        return previous;
      }

      const nextColumns = Array.from(columnSet).sort((a, b) => a.localeCompare(b));

      return {
        ...previous,
        columns: nextColumns,
        column_methods: nextMethods,
      };
    });
  }, [scalingRecommendations, setConfigState]);

  const handleScalingSkipColumn = useCallback(
    (column: string) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }
      setConfigState((previous) => {
        const currentColumns = ensureArrayOfString(previous.columns);
        const filteredColumns = currentColumns.filter((value) => value !== normalized);

        const skippedSet = new Set(ensureArrayOfString(previous.skipped_columns));
        const sizeBefore = skippedSet.size;
        skippedSet.add(normalized);
        const skipChanged = skippedSet.size !== sizeBefore;

        const rawColumnMethods =
          previous.column_methods && typeof previous.column_methods === 'object' && !Array.isArray(previous.column_methods)
            ? { ...(previous.column_methods as Record<string, ScalingMethodName>) }
            : {};
        const hadMethod = Object.prototype.hasOwnProperty.call(rawColumnMethods, normalized);
        if (hadMethod) {
          delete rawColumnMethods[normalized];
        }

        if (!skipChanged && filteredColumns.length === currentColumns.length && !hadMethod) {
          return previous;
        }

        return {
          ...previous,
          columns: filteredColumns.sort((a, b) => a.localeCompare(b)),
          column_methods: rawColumnMethods,
          skipped_columns: Array.from(skippedSet).sort((a, b) => a.localeCompare(b)),
        };
      });
    },
    [setConfigState]
  );

  const handleScalingUnskipColumn = useCallback(
    (column: string) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }
      setConfigState((previous) => {
        const skipped = ensureArrayOfString(previous.skipped_columns);
        if (!skipped.includes(normalized)) {
          return previous;
        }

        const nextSkippedSet = new Set(skipped);
        nextSkippedSet.delete(normalized);

        const currentColumns = new Set(ensureArrayOfString(previous.columns));
        const hadColumn = currentColumns.has(normalized);
        if (!hadColumn) {
          currentColumns.add(normalized);
        }

        return {
          ...previous,
          columns: Array.from(currentColumns).sort((a, b) => a.localeCompare(b)),
          skipped_columns: Array.from(nextSkippedSet).sort((a, b) => a.localeCompare(b)),
        };
      });
    },
    [setConfigState]
  );

  const handleScalingOverrideSelect = useCallback(
    (column: string, value: string) => {
      if (value === '__skip__') {
        handleScalingSkipColumn(column);
        return;
      }

      if (value === '__default__') {
        handleScalingUnskipColumn(column);
        setScalingColumnMethod(column, null);
        return;
      }

      if (SCALING_METHOD_ORDER.includes(value as ScalingMethodName)) {
        handleScalingUnskipColumn(column);
        setScalingColumnMethod(column, value as ScalingMethodName);
      }
    },
    [handleScalingSkipColumn, handleScalingUnskipColumn, setScalingColumnMethod]
  );

  const handleClearColumns = useCallback(() => {
    setConfigState((previous) => {
      const next: Record<string, any> = {
        ...previous,
        columns: [],
      };
      if (
        previous?.manual_bounds &&
        typeof previous.manual_bounds === 'object' &&
        !Array.isArray(previous.manual_bounds) &&
        Object.keys(previous.manual_bounds).length
      ) {
        next.manual_bounds = {};
      }
      if (isBinningNode) {
        if ('custom_bins' in next) {
          delete next.custom_bins;
        }
        if ('custom_labels' in next) {
          delete next.custom_labels;
        }
        if ('column_overrides' in next) {
          delete next.column_overrides;
        }
        if ('column_strategies' in next) {
          delete next.column_strategies;
        }
      }
      if (isScalingNode && 'column_methods' in next) {
        delete next.column_methods;
      }
      return next;
    });
  }, [isBinningNode, isScalingNode]);

  const handleRemoveColumn = useCallback((column: string) => {
    const normalizedColumn = String(column ?? '').trim();
    if (!normalizedColumn) {
      return;
    }
    setConfigState((previous) => {
      const current = ensureArrayOfString(previous.columns);
      if (!current.includes(normalizedColumn)) {
        return previous;
      }
      const nextColumns = current.filter((item) => item !== normalizedColumn);
      let manualChanged = false;
      let manualBounds = previous.manual_bounds;
      if (
        manualBounds &&
        typeof manualBounds === 'object' &&
        !Array.isArray(manualBounds) &&
        Object.prototype.hasOwnProperty.call(manualBounds, normalizedColumn)
      ) {
        manualBounds = { ...manualBounds };
        delete manualBounds[normalizedColumn];
        manualChanged = true;
      }
      const result: Record<string, any> = {
        ...previous,
        columns: nextColumns,
      };

      if (manualChanged) {
        result.manual_bounds = manualBounds;
      }

      if (isBinningNode) {
        if (
          previous.custom_bins &&
          typeof previous.custom_bins === 'object' &&
          !Array.isArray(previous.custom_bins)
        ) {
          const nextBins: Record<string, number[]> = {};
          Object.entries(previous.custom_bins).forEach(([key, value]) => {
            if (key !== normalizedColumn && Array.isArray(value) && value.length) {
              nextBins[key] = [...value];
            }
          });
          if (Object.keys(nextBins).length) {
            result.custom_bins = nextBins;
          } else if ('custom_bins' in result) {
            delete result.custom_bins;
          }
        } else if ('custom_bins' in result) {
          delete result.custom_bins;
        }

        if (
          previous.custom_labels &&
          typeof previous.custom_labels === 'object' &&
          !Array.isArray(previous.custom_labels)
        ) {
          const nextLabels: Record<string, string[]> = {};
          Object.entries(previous.custom_labels).forEach(([key, value]) => {
            if (key !== normalizedColumn && Array.isArray(value) && value.length) {
              nextLabels[key] = value.map((entry) => String(entry));
            }
          });
          if (Object.keys(nextLabels).length) {
            result.custom_labels = nextLabels;
          } else if ('custom_labels' in result) {
            delete result.custom_labels;
          }
        } else if ('custom_labels' in result) {
          delete result.custom_labels;
        }

        if (
          previous.column_overrides &&
          typeof previous.column_overrides === 'object' &&
          !Array.isArray(previous.column_overrides)
        ) {
          const nextOverrides: Record<string, any> = {};
          Object.entries(previous.column_overrides as Record<string, any>).forEach(([key, value]) => {
            if (key !== normalizedColumn && value && typeof value === 'object' && !Array.isArray(value)) {
              nextOverrides[key] = { ...(value as Record<string, any>) };
            }
          });
          if (Object.keys(nextOverrides).length) {
            result.column_overrides = nextOverrides;
            result.column_strategies = nextOverrides;
          } else {
            if ('column_overrides' in result) {
              delete result.column_overrides;
            }
            if ('column_strategies' in result) {
              delete result.column_strategies;
            }
          }
        } else {
          if ('column_overrides' in result) {
            delete result.column_overrides;
          }
          if ('column_strategies' in result) {
            delete result.column_strategies;
          }
        }
      }

      if (isScalingNode) {
        if (
          previous.column_methods &&
          typeof previous.column_methods === 'object' &&
          !Array.isArray(previous.column_methods)
        ) {
          const nextMethods: Record<string, ScalingMethodName> = {};
          Object.entries(previous.column_methods as Record<string, any>).forEach(([key, value]) => {
            const methodKey = typeof value === 'string' ? (value.trim() as ScalingMethodName) : null;
            if (!methodKey || !SCALING_METHOD_ORDER.includes(methodKey)) {
              return;
            }
            if (key === normalizedColumn) {
              return;
            }
            nextMethods[key] = methodKey;
          });
          if (Object.keys(nextMethods).length) {
            result.column_methods = nextMethods;
          } else if (Object.prototype.hasOwnProperty.call(result, 'column_methods')) {
            delete result.column_methods;
          }
        } else if (Object.prototype.hasOwnProperty.call(result, 'column_methods')) {
          delete result.column_methods;
        }
      }

      return result;
    });
  }, [isBinningNode, isScalingNode]);

  const handleApplySuggestedThreshold = useCallback(() => {
    if (normalizedSuggestedThreshold === null || !thresholdParameterName) {
      return;
    }
    handleParameterChange(thresholdParameterName, normalizedSuggestedThreshold);
  }, [handleParameterChange, normalizedSuggestedThreshold, thresholdParameterName]);

  const handleResetNode = useCallback(() => {
    if (!canResetNode || !nodeId) {
      return;
    }
    const template = cloneConfig(defaultConfigTemplate ?? {});
    setConfigState(template);
    onResetConfig?.(nodeId, template);
  }, [canResetNode, defaultConfigTemplate, nodeId, onResetConfig, setConfigState]);

  const handleSave = useCallback((options?: { closeModal?: boolean }) => {
    let payload = cloneConfig(configState);
    
    if (isBinningNode) {
      const normalized = normalizeBinningConfigValue(payload);
      payload = {
        ...payload,
        strategy: normalized.strategy,
        columns: normalized.columns,
        equal_width_bins: normalized.equalWidthBins,
        equal_frequency_bins: normalized.equalFrequencyBins,
        include_lowest: normalized.includeLowest,
        precision: normalized.precision,
        duplicates: normalized.duplicates,
        output_suffix: normalized.outputSuffix,
        drop_original: normalized.dropOriginal,
        label_format: normalized.labelFormat,
        missing_strategy: normalized.missingStrategy,
      };

      if (normalized.missingStrategy === 'label') {
        payload.missing_label = normalized.missingLabel;
      } else if (Object.prototype.hasOwnProperty.call(payload, 'missing_label')) {
        delete payload.missing_label;
      }

      if (Object.keys(normalized.customBins).length) {
        payload.custom_bins = normalized.customBins;
      } else if (Object.prototype.hasOwnProperty.call(payload, 'custom_bins')) {
        delete payload.custom_bins;
      }

      if (Object.keys(normalized.customLabels).length) {
        payload.custom_labels = normalized.customLabels;
      } else if (Object.prototype.hasOwnProperty.call(payload, 'custom_labels')) {
        delete payload.custom_labels;
      }
    }

    if (isScalingNode) {
      const normalized = normalizeScalingConfigValue(payload);
      payload = {
        ...payload,
        columns: normalized.columns,
        default_method: normalized.defaultMethod,
        auto_detect: normalized.autoDetect,
        skipped_columns: normalized.skippedColumns,
      };
      
      // Only include column_methods if there are overrides
      if (Object.keys(normalized.columnMethods).length > 0) {
        payload.column_methods = normalized.columnMethods;
      } else if (Object.prototype.hasOwnProperty.call(payload, 'column_methods')) {
        delete payload.column_methods;
      }
    }

    onUpdateConfig(node.id, payload);
    if (options?.closeModal !== false) {
      onClose();
    }
    
    // Trigger full dataset execution in background after saving
    // This pre-loads the full dataset for this node's transformations
    // Skip for inspection nodes (they only view data, don't transform it)
    if (sourceId && graphSnapshot && !isInspectionNode) {
      // Set loading status immediately
      if (onUpdateNodeData) {
        onUpdateNodeData(node.id, { backgroundExecutionStatus: 'loading' });
      }
      
      triggerFullDatasetExecution({
        dataset_source_id: sourceId,
        graph: {
          nodes: graphSnapshot.nodes || [],
          edges: graphSnapshot.edges || [],
        },
        target_node_id: node.id,
      })
        .then(() => {
          // Update to success when complete
          if (onUpdateNodeData) {
            onUpdateNodeData(node.id, { backgroundExecutionStatus: 'success' });
          }
        })
        .catch((error) => {
          // Update to error on failure
          if (onUpdateNodeData) {
            onUpdateNodeData(node.id, { backgroundExecutionStatus: 'error' });
          }
          // Silent fail - this is a background optimization, not critical
          console.warn('Background full dataset execution failed:', error);
        });
    }
  }, [
    configState,
    isBinningNode,
    node.id,
    onClose,
    onUpdateConfig,
    onUpdateNodeData,
    sourceId,
    graphSnapshot,
    isInspectionNode,
  ]);

  const canApplySuggestedThreshold =
    normalizedSuggestedThreshold !== null && !thresholdMatchesSuggestion;
  const selectionCount = selectedColumns.length;
  const relativeGeneratedAt = formatRelativeTime(recommendationsGeneratedAt);
  const relativeScalingGeneratedAt = formatRelativeTime(scalingData?.generated_at ?? null);
  const relativeBinningGeneratedAt = formatRelativeTime(binningData?.generated_at ?? null);
  const relativeBinnedGeneratedAt = formatRelativeTime(binnedDistributionData?.generated_at ?? null);
  const scalingSampleSizeNumeric = Number(scalingData?.sample_size);
  const scalingSampleSize = Number.isFinite(scalingSampleSizeNumeric)
    ? Math.max(0, Math.round(scalingSampleSizeNumeric))
    : null;
  const binningSampleSizeNumeric = Number(binningData?.sample_size);
  const binningSampleSize = Number.isFinite(binningSampleSizeNumeric)
    ? Math.max(0, Math.round(binningSampleSizeNumeric))
    : null;
  const binnedSampleSizeNumeric = Number(binnedDistributionData?.sample_size);
  const binnedSampleSize = Number.isFinite(binnedSampleSizeNumeric)
    ? Math.max(0, Math.round(binnedSampleSizeNumeric))
    : null;
  const showRecommendations = hasDropColumnSource && Boolean(sourceId) && hasReachableSource;
  const showSaveButton = !datasetBadge && !isInspectionNode;
  const canSave = showSaveButton && stableInitialConfig !== stableCurrentConfig;
  const isProfileLoading = profileState.status === 'loading';
  const isPreviewLoading = previewState.status === 'loading';
  const hasActiveAsyncWork =
    isProfileLoading ||
    isPreviewLoading ||
    isFetchingScaling ||
    isFetchingBinning ||
    isFetchingHashEncoding ||
    isFetchingBinnedDistribution ||
    isFetchingRecommendations;

  const busyLabel = useMemo(() => {
    if (isProfileLoading) {
      return 'Generating dataset profile…';
    }
    if (isFetchingScaling) {
      return 'Loading scaling insights…';
    }
    if (isFetchingBinning) {
      return 'Loading binning insights…';
    }
    if (isFetchingHashEncoding) {
      return 'Loading hash encoding insights…';
    }
    if (isFetchingBinnedDistribution) {
      return 'Computing binned distributions…';
    }
    if (isFetchingRecommendations) {
      return 'Fetching column recommendations…';
    }
    if (isPreviewLoading) {
      return isPreviewNode ? 'Loading dataset preview…' : 'Analyzing node outputs…';
    }
    return null;
  }, [
    isFetchingBinnedDistribution,
    isFetchingRecommendations,
    isFetchingScaling,
    isFetchingBinning,
    isFetchingHashEncoding,
    isFetchingSkewness,
    isPreviewLoading,
    isProfileLoading,
    isPreviewNode,
  ]);
  const footerBusyLabel = busyLabel ?? (hasActiveAsyncWork ? 'Processing…' : null);

  const updateImputerStrategies = useCallback(
    (updater: (current: ImputationStrategyConfig[]) => ImputationStrategyConfig[]) => {
      setConfigState((previous) => {
        const currentStrategies = normalizeImputationStrategies(previous?.strategies, imputationMethodValues);
        const nextStrategies = updater(currentStrategies).map((strategy) => ({
          ...strategy,
          options: sanitizeOptionsForMethod(strategy.method, strategy.options),
        }));
        return {
          ...previous,
          strategies: serializeImputationStrategies(nextStrategies),
        };
      });
    },
    [imputationMethodValues]
  );

  const updateAliasStrategies = useCallback(
    (updater: (current: AliasStrategyConfig[]) => AliasStrategyConfig[]) => {
      if (!isReplaceAliasesNode) {
        return;
      }

      setConfigState((previous) => {
        if (!isReplaceAliasesNode) {
          return previous;
        }

        const fallbackColumns = ensureArrayOfString(previous?.columns ?? node?.data?.config?.columns);
        const fallbackMode = resolveAliasMode(previous?.mode, node?.data?.config?.mode);

        const fallbackAutoDetect = (() => {
          const localValue = pickAutoDetectValue(previous as Record<string, unknown>);
          if (localValue !== undefined) {
            const normalized = normalizeConfigBoolean(localValue);
            if (normalized !== null) {
              return normalized;
            }
          }
          const nodeValue = pickAutoDetectValue(node?.data?.config as Record<string, unknown> | undefined);
          if (nodeValue !== undefined) {
            const normalized = normalizeConfigBoolean(nodeValue);
            if (normalized !== null) {
              return normalized;
            }
          }
          return fallbackColumns.length === 0;
        })();

        const rawStrategies =
          previous?.alias_strategies ??
          previous?.strategies ??
          node?.data?.config?.alias_strategies ??
          node?.data?.config?.strategies;

        const currentStrategies = normalizeAliasStrategies(rawStrategies, {
          mode: fallbackMode,
          columns: fallbackColumns,
          autoDetect: fallbackAutoDetect,
        });

        const nextStrategies = updater(currentStrategies).map((strategy) => ({
          mode: resolveAliasMode(strategy.mode, fallbackMode),
          columns: Array.from(
            new Set(strategy.columns.map((column) => column.trim()).filter(Boolean)),
          ).sort((a, b) => a.localeCompare(b)),
          autoDetect: Boolean(strategy.autoDetect),
        }));

        const serialized = serializeAliasStrategies(nextStrategies);
        const unionColumns = Array.from(
          new Set(
            nextStrategies.flatMap((strategy) => strategy.columns),
          ),
        ).sort((a, b) => a.localeCompare(b));

        const primaryMode = nextStrategies[0]?.mode ?? fallbackMode;
        const autoDetectAny = nextStrategies.length
          ? nextStrategies.some((strategy) => strategy.autoDetect)
          : fallbackAutoDetect;

        return {
          ...previous,
          alias_strategies: serialized,
          columns: unionColumns,
          mode: primaryMode,
          auto_detect: autoDetectAny,
        };
      });
    },
    [isReplaceAliasesNode, node?.data?.config]
  );

  const updateDateStrategies = useCallback(
    (updater: (current: DateFormatStrategyConfig[]) => DateFormatStrategyConfig[]) => {
      setConfigState((previous) => {
        if (!isStandardizeDatesNode) {
          return previous;
        }

        const fallbackColumns = ensureArrayOfString(previous?.columns ?? node?.data?.config?.columns);
        const fallbackMode = resolveDateMode(previous?.mode, node?.data?.config?.mode);
        const fallbackAutoDetect = fallbackColumns.length === 0;

        const currentStrategies = normalizeDateFormatStrategies(
          previous?.format_strategies ?? node?.data?.config?.format_strategies,
          {
            columns: fallbackColumns,
            mode: fallbackMode,
            autoDetect: fallbackAutoDetect,
          },
        );

        const nextStrategies = updater(currentStrategies);
        const serialized = serializeDateFormatStrategies(nextStrategies);
        const unionColumns = Array.from(
          new Set(
            nextStrategies.flatMap((strategy) =>
              strategy.columns.map((column) => column.trim()).filter(Boolean),
            ),
          ),
        ).sort((a, b) => a.localeCompare(b));
        const primaryMode = nextStrategies[0]?.mode ?? fallbackMode;

        return {
          ...previous,
          format_strategies: serialized,
          columns: unionColumns,
          mode: primaryMode,
        };
      });
    },
    [isStandardizeDatesNode, node?.data?.config?.columns, node?.data?.config?.format_strategies, node?.data?.config?.mode],
  );
  const handleRefreshBinnedDistribution = useCallback(() => {
    if (!sourceId) {
      return;
    }
    refreshBinnedDistribution();
  }, [refreshBinnedDistribution, sourceId]);

  const binnedDistributionCards = useMemo<BinnedDistributionCard[]>(() => {
    if (!isBinnedDistributionNode) {
      return [];
    }

    const rawColumns = Array.isArray(binnedDistributionData?.columns) ? binnedDistributionData?.columns : [];

    const cards = rawColumns
      .map((entry) => {
        if (!entry || !entry.column) {
          return null;
        }

        const totalRowsNumeric = Number(entry.total_rows);
        const totalRows = Number.isFinite(totalRowsNumeric) ? Math.max(0, Math.round(totalRowsNumeric)) : 0;
        if (totalRows <= 0) {
          return null;
        }

        const rawBins = Array.isArray(entry.bins) ? entry.bins : [];
        const sanitizedBins = rawBins
          .map((bin): BinnedDistributionBin | null => {
            if (!bin) {
              return null;
            }

            const labelRaw = typeof bin.label === 'string' ? bin.label.trim() : String(bin.label ?? '');
            const label = labelRaw || (bin.is_missing ? 'Missing' : 'Unlabeled bin');

            const numericCount = Number(bin.count);
            if (!Number.isFinite(numericCount)) {
              return null;
            }

            const safeCount = Math.max(0, Math.round(numericCount));

            const numericPercentage = Number(bin.percentage);
            const clampedPercentage = Number.isFinite(numericPercentage)
              ? Math.min(100, Math.max(0, numericPercentage))
              : 0;
            const roundedPercentage = Number(clampedPercentage.toFixed(2));

            return {
              label,
              count: safeCount,
              percentage: roundedPercentage,
              isMissing: Boolean(bin.is_missing),
            };
          })
          .filter((bin): bin is BinnedDistributionBin => Boolean(bin));

        if (!sanitizedBins.length) {
          return null;
        }

        sanitizedBins.sort((a, b) => {
          if (a.isMissing !== b.isMissing) {
            return a.isMissing ? 1 : -1;
          }
          if (a.count !== b.count) {
            return b.count - a.count;
          }
          return a.label.localeCompare(b.label);
        });

        const totalBinCount = sanitizedBins.length;
        const MAX_BINS = 12;
        const hasMoreBins = totalBinCount > MAX_BINS;
        const bins = hasMoreBins ? sanitizedBins.slice(0, MAX_BINS) : sanitizedBins;

        const missingRowsNumeric = Number(entry.missing_rows);
        const missingRows = Number.isFinite(missingRowsNumeric) ? Math.max(0, Math.round(missingRowsNumeric)) : 0;

        const distinctBinsNumeric = Number(entry.distinct_bins);
        const distinctBins = Number.isFinite(distinctBinsNumeric)
          ? Math.max(0, Math.round(distinctBinsNumeric))
          : sanitizedBins.length;

        const rawTopLabel = entry.top_label;
        let topLabel = typeof rawTopLabel === 'string' ? rawTopLabel.trim() || null : rawTopLabel ?? null;
        const topCountNumeric = Number(entry.top_count);
        let topCount = Number.isFinite(topCountNumeric) ? Math.max(0, Math.round(topCountNumeric)) : null;
        const topPercentageNumeric = Number(entry.top_percentage);
        let topPercentage = Number.isFinite(topPercentageNumeric)
          ? Number(Math.min(100, Math.max(0, topPercentageNumeric)).toFixed(2))
          : null;

        if ((topLabel === null || topCount === null || topPercentage === null) && bins.length) {
          topLabel = bins[0].label;
          topCount = bins[0].count;
          topPercentage = bins[0].percentage;
        }

        const sourceColumn = typeof entry.source_column === 'string' ? entry.source_column.trim() || null : entry.source_column ?? null;

        return {
          column: entry.column,
          sourceColumn,
          totalRows,
          missingRows,
          distinctBins,
          topLabel,
          topCount,
          topPercentage,
          bins,
          hasMoreBins,
          totalBinCount,
        } as BinnedDistributionCard;
      })
      .filter((card): card is BinnedDistributionCard => Boolean(card));

    cards.sort((a, b) => {
      const diff = (b.topCount ?? 0) - (a.topCount ?? 0);
      if (diff !== 0) {
        return diff;
      }
      return a.column.localeCompare(b.column);
    });

    return cards;
  }, [binnedDistributionData, isBinnedDistributionNode]);

  const handleAddImputerStrategy = useCallback(() => {
    const nextIndex = imputerStrategyCount;
    const defaultMethod = imputationMethodOptions[0]?.value ?? 'mean';
    updateImputerStrategies((current) => [
      ...current,
      {
        method: defaultMethod,
        columns: [],
        options: sanitizeOptionsForMethod(defaultMethod, buildDefaultOptionsForMethod(defaultMethod)),
      },
    ]);
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      next.delete(nextIndex);
      return next;
    });
  }, [imputationMethodOptions, imputerStrategyCount, updateImputerStrategies]);

  const handleRemoveImputerStrategy = useCallback(
    (index: number) => {
      updateImputerStrategies((current) => current.filter((_, idx) => idx !== index));
      setCollapsedStrategies((previous) => {
        const next = new Set<number>();
        previous.forEach((value) => {
          if (value === index) {
            return;
          }
          next.add(value > index ? value - 1 : value);
        });
        return next;
      });
    },
    [updateImputerStrategies]
  );

  const toggleImputerStrategySection = useCallback((index: number) => {
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, []);

  const handleImputerMethodChange = useCallback(
    (index: number, method: ImputationStrategyMethod) => {
      updateImputerStrategies((current) =>
        current.map((strategy, idx) =>
          idx === index
            ? {
                ...strategy,
                method,
                options: sanitizeOptionsForMethod(method, strategy.options),
              }
            : strategy
        )
      );
    },
    [updateImputerStrategies]
  );

  const handleImputerOptionNumberChange = useCallback(
    (index: number, key: 'neighbors' | 'max_iter', rawValue: string) => {
      updateImputerStrategies((current) =>
        current.map((strategy, idx) => {
          if (idx !== index) {
            return strategy;
          }
          const nextOptions: ImputationStrategyOptions = {
            ...(strategy.options ?? {}),
          };
          if (rawValue === '') {
            delete nextOptions[key];
          } else {
            const parsed = Number(rawValue);
            if (!Number.isFinite(parsed)) {
              return strategy;
            }
            nextOptions[key] = parsed;
          }
          return {
            ...strategy,
            options: sanitizeOptionsForMethod(strategy.method, nextOptions),
          };
        })
      );
    },
    [updateImputerStrategies]
  );

  const handleImputerColumnsChange = useCallback(
    (index: number, value: string) => {
      const normalized = value
        .split(',')
        .map((column) => column.trim())
        .filter(Boolean);
      updateImputerStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, columns: normalized } : strategy))
      );
    },
    [updateImputerStrategies]
  );

  const handleImputerColumnToggle = useCallback(
    (index: number, column: string) => {
      const normalized = column.trim();
      if (!normalized) {
        return;
      }

      updateImputerStrategies((current) =>
        current.map((strategy, idx) => {
          if (idx !== index) {
            return strategy;
          }
          const hasColumn = strategy.columns.includes(normalized);
          const nextColumns = hasColumn
            ? strategy.columns.filter((item) => item !== normalized)
            : [...strategy.columns, normalized];
          nextColumns.sort((a, b) => a.localeCompare(b));
          return {
            ...strategy,
            columns: nextColumns,
          };
        })
      );
    },
    [updateImputerStrategies]
  );

  const handleImputerMissingFilterChange = useCallback(
    (value: number) => {
      setImputerMissingFilter(value);
    },
    [setImputerMissingFilter]
  );

  const handleAddDateStrategy = useCallback(() => {
    const nextIndex = dateStrategies.length;
    updateDateStrategies((current) => {
      const assigned = new Set<string>();
      current.forEach((strategy) => {
        strategy.columns.forEach((column) => {
          const normalized = column.trim();
          if (normalized) {
            assigned.add(normalized);
          }
        });
      });
      const suggested = standardizeDatesColumnSummary.recommendedColumns.find((column) => !assigned.has(column));
      const defaultMode = current.length
        ? current[current.length - 1].mode
        : DATE_MODE_OPTIONS[0]?.value ?? standardizeDatesMode;
      const nextStrategy: DateFormatStrategyConfig = {
        mode: defaultMode,
        columns: suggested ? [suggested] : [],
        autoDetect: !suggested,
      };
      return [...current, nextStrategy];
    });
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      next.delete(nextIndex);
      return next;
    });
  }, [dateStrategies.length, setCollapsedStrategies, standardizeDatesColumnSummary.recommendedColumns, standardizeDatesMode, updateDateStrategies]);

  const handleRemoveDateStrategy = useCallback(
    (index: number) => {
      updateDateStrategies((current) => current.filter((_, idx) => idx !== index));
      setCollapsedStrategies((previous) => {
        const next = new Set<number>();
        previous.forEach((value) => {
          if (value === index) {
            return;
          }
          next.add(value > index ? value - 1 : value);
        });
        return next;
      });
    },
    [setCollapsedStrategies, updateDateStrategies],
  );

  const toggleDateStrategySection = useCallback((index: number) => {
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, []);

  const handleDateStrategyModeChange = useCallback(
    (index: number, mode: DateMode) => {
      updateDateStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, mode } : strategy)),
      );
    },
    [updateDateStrategies],
  );

  const handleDateStrategyColumnsChange = useCallback(
    (index: number, value: string) => {
      const normalized = value
        .split(',')
        .map((column) => column.trim())
        .filter(Boolean)
        .sort((a, b) => a.localeCompare(b));
      updateDateStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, columns: normalized } : strategy)),
      );
    },
    [updateDateStrategies],
  );

  const handleDateStrategyColumnToggle = useCallback(
    (index: number, column: string) => {
      const normalized = column.trim();
      if (!normalized) {
        return;
      }
      updateDateStrategies((current) =>
        current.map((strategy, idx) => {
          if (idx !== index) {
            return strategy;
          }
          const hasColumn = strategy.columns.includes(normalized);
          const nextColumns = hasColumn
            ? strategy.columns.filter((entry) => entry !== normalized)
            : [...strategy.columns, normalized];
          nextColumns.sort((a, b) => a.localeCompare(b));
          return {
            ...strategy,
            columns: nextColumns,
          };
        }),
      );
    },
    [updateDateStrategies],
  );

  const handleDateStrategyAutoDetectToggle = useCallback(
    (index: number, enabled: boolean) => {
      updateDateStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, autoDetect: enabled } : strategy)),
      );
    },
    [updateDateStrategies],
  );

  const updateFeatureMathOperations = useCallback(
    (updater: (operations: FeatureMathOperationDraft[]) => FeatureMathOperationDraft[]) => {
      setConfigState((previous) => {
        if (!isFeatureMathNode) {
          return previous;
        }
        const baseConfig =
          previous && typeof previous === 'object' && !Array.isArray(previous) ? { ...previous } : {};
        const existingOperations = normalizeFeatureMathOperations(
          Array.isArray((previous as any)?.operations) ? (previous as any).operations : [],
        );
        const nextOperations = updater(existingOperations);
        baseConfig.operations = serializeFeatureMathOperations(nextOperations);
        return baseConfig;
      });
    },
    [isFeatureMathNode, setConfigState],
  );

  const handleAddFeatureMathOperation = useCallback(
    (operationType: FeatureMathOperationType) => {
      if (!isFeatureMathNode) {
        return;
      }
      let createdOperationId = '';
      updateFeatureMathOperations((current) => {
        const existingIds = new Set(current.map((operation) => operation.id));
        const draft = createFeatureMathOperation(operationType, existingIds);
        createdOperationId = draft.id;
        return [...current, draft];
      });
      if (createdOperationId) {
        setCollapsedFeatureMath((previous) => {
          const next = new Set(previous);
          next.delete(createdOperationId);
          return next;
        });
      }
    },
    [isFeatureMathNode, setCollapsedFeatureMath, updateFeatureMathOperations],
  );

  const handleDuplicateFeatureMathOperation = useCallback(
    (operationId: string) => {
      if (!isFeatureMathNode) {
        return;
      }
      let createdOperationId = '';
      updateFeatureMathOperations((current) => {
        const index = current.findIndex((operation) => operation.id === operationId);
        if (index === -1) {
          return current;
        }
        const source = current[index];
        const existingIds = new Set(current.map((operation) => operation.id));
        const seed = createFeatureMathOperation(source.type, existingIds);
        createdOperationId = seed.id;
        const duplicate: FeatureMathOperationDraft = {
          ...source,
          id: seed.id,
          inputColumns: [...source.inputColumns],
          secondaryColumns: [...source.secondaryColumns],
          constants: [...source.constants],
          datetimeFeatures: [...source.datetimeFeatures],
          outputColumn: source.outputColumn ? `${source.outputColumn}_copy` : '',
        };
        const next = [...current];
        next.splice(index + 1, 0, duplicate);
        return next;
      });
      if (createdOperationId) {
        setCollapsedFeatureMath((previous) => {
          const next = new Set(previous);
          next.delete(createdOperationId);
          return next;
        });
      }
    },
    [isFeatureMathNode, setCollapsedFeatureMath, updateFeatureMathOperations],
  );

  const handleRemoveFeatureMathOperation = useCallback(
    (operationId: string) => {
      if (!isFeatureMathNode) {
        return;
      }
      updateFeatureMathOperations((current) => current.filter((operation) => operation.id !== operationId));
      setCollapsedFeatureMath((previous) => {
        const next = new Set(previous);
        next.delete(operationId);
        return next;
      });
    },
    [isFeatureMathNode, setCollapsedFeatureMath, updateFeatureMathOperations],
  );

  const handleReorderFeatureMathOperation = useCallback(
    (operationId: string, direction: 'up' | 'down') => {
      if (!isFeatureMathNode) {
        return;
      }
      updateFeatureMathOperations((current) => {
        const index = current.findIndex((operation) => operation.id === operationId);
        if (index === -1) {
          return current;
        }
        const targetIndex = direction === 'up' ? index - 1 : index + 1;
        if (targetIndex < 0 || targetIndex >= current.length) {
          return current;
        }
        const next = [...current];
        const [moved] = next.splice(index, 1);
        next.splice(targetIndex, 0, moved);
        return next;
      });
    },
    [isFeatureMathNode, updateFeatureMathOperations],
  );

  const handleToggleFeatureMathOperation = useCallback((operationId: string) => {
    setCollapsedFeatureMath((previous) => {
      const next = new Set(previous);
      if (next.has(operationId)) {
        next.delete(operationId);
      } else {
        next.add(operationId);
      }
      return next;
    });
  }, []);

  const handleFeatureMathOperationChange = useCallback(
    (operationId: string, updates: Partial<FeatureMathOperationDraft>) => {
      if (!isFeatureMathNode) {
        return;
      }
      updateFeatureMathOperations((current) =>
        current.map((operation) => {
          if (operation.id !== operationId) {
            return operation;
          }

          let next: FeatureMathOperationDraft = { ...operation };

          if (updates.type) {
            next.type = updates.type;
          }

          if (updates.method !== undefined) {
            next.method = updates.method;
          }

          if (updates.inputColumns !== undefined) {
            next.inputColumns = sanitizeStringList(updates.inputColumns);
          }

          if (updates.secondaryColumns !== undefined) {
            next.secondaryColumns = sanitizeStringList(updates.secondaryColumns);
          }

          if (updates.constants !== undefined) {
            next.constants = sanitizeConstantsList(updates.constants);
          }

          if (updates.outputColumn !== undefined) {
            next.outputColumn = updates.outputColumn.trim();
          }

          if (updates.outputPrefix !== undefined) {
            next.outputPrefix = updates.outputPrefix.trim();
          }

          if (updates.datetimeFeatures !== undefined) {
            next.datetimeFeatures = sanitizeDatetimeFeaturesList(updates.datetimeFeatures);
          }

          if (updates.timezone !== undefined) {
            next.timezone = sanitizeTimezoneValue(updates.timezone);
          }

          if (updates.fillna !== undefined) {
            next.fillna = sanitizeNumberValue(updates.fillna);
          }

          if (updates.roundDigits !== undefined) {
            next.roundDigits = sanitizeIntegerValue(updates.roundDigits);
          }

          if (updates.normalize !== undefined) {
            next.normalize = updates.normalize;
          }

          if (updates.epsilon !== undefined) {
            next.epsilon = sanitizeNumberValue(updates.epsilon);
          }

          if (updates.allowOverwrite !== undefined) {
            next.allowOverwrite = updates.allowOverwrite;
          }

          if (updates.description !== undefined) {
            const trimmed = typeof updates.description === 'string' ? updates.description.trim() : '';
            next.description = trimmed ? trimmed : undefined;
          }

          const resolvedType = next.type;
          const methodChoices = getMethodOptions(resolvedType);
          if (methodChoices.length) {
            const allowed = new Set(methodChoices.map((option) => option.value));
            if (!allowed.has(next.method)) {
              next.method = methodChoices[0].value;
            }
          } else if (resolvedType === 'ratio') {
            next.method = 'ratio';
          } else if (resolvedType === 'datetime_extract') {
            next.method = 'datetime_extract';
          }

          next.inputColumns = sanitizeStringList(next.inputColumns);
          if (resolvedType === 'ratio' || resolvedType === 'similarity') {
            next.secondaryColumns = sanitizeStringList(next.secondaryColumns);
          } else {
            next.secondaryColumns = [];
          }

          next.constants = sanitizeConstantsList(next.constants);
          next.fillna = sanitizeNumberValue(next.fillna);
          next.roundDigits = sanitizeIntegerValue(next.roundDigits);
          next.epsilon = sanitizeNumberValue(next.epsilon);

          if (resolvedType === 'datetime_extract') {
            next.datetimeFeatures = sanitizeDatetimeFeaturesList(next.datetimeFeatures);
            next.timezone = sanitizeTimezoneValue(next.timezone);
          } else {
            next.datetimeFeatures = [];
            next.timezone = 'UTC';
          }

          if (resolvedType !== 'similarity') {
            next.normalize = false;
          } else {
            next.normalize = Boolean(next.normalize);
          }

          if (typeof next.allowOverwrite !== 'boolean') {
            next.allowOverwrite = null;
          }

          next.outputColumn = next.outputColumn.trim();
          next.outputPrefix = next.outputPrefix.trim();

          return next;
        }),
      );
    },
    [isFeatureMathNode, updateFeatureMathOperations],
  );

  const handleAddAliasStrategy = useCallback(() => {
    if (!isReplaceAliasesNode) {
      return;
    }
    const nextIndex = aliasStrategyCount;
    updateAliasStrategies((current) => {
      const assigned = new Set<string>();
      current.forEach((strategy) => {
        strategy.columns.forEach((column) => {
          const normalized = column.trim();
          if (normalized) {
            assigned.add(normalized);
          }
        });
      });
      const suggested = aliasColumnSummary.recommendedColumns.find((column) => !assigned.has(column));
      const defaultMode = current.length
        ? current[current.length - 1].mode
        : ALIAS_MODE_OPTIONS[0]?.value ?? DEFAULT_ALIAS_MODE;
      const nextStrategy: AliasStrategyConfig = {
        mode: defaultMode,
        columns: suggested ? [suggested] : [],
        autoDetect: !suggested,
      };
      return [...current, nextStrategy];
    });
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      next.delete(nextIndex);
      return next;
    });
  }, [aliasColumnSummary.recommendedColumns, aliasStrategyCount, isReplaceAliasesNode, setCollapsedStrategies, updateAliasStrategies]);

  const handleRemoveAliasStrategy = useCallback(
    (index: number) => {
      updateAliasStrategies((current) => current.filter((_, idx) => idx !== index));
      setCollapsedStrategies((previous) => {
        const next = new Set<number>();
        previous.forEach((value) => {
          if (value === index) {
            return;
          }
          next.add(value > index ? value - 1 : value);
        });
        return next;
      });
    },
    [setCollapsedStrategies, updateAliasStrategies],
  );

  const toggleAliasStrategySection = useCallback((index: number) => {
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, []);

  const handleAliasModeChange = useCallback(
    (index: number, mode: AliasMode) => {
      updateAliasStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, mode } : strategy)),
      );
    },
    [updateAliasStrategies],
  );

  const handleAliasColumnToggle = useCallback(
    (index: number, column: string) => {
      const normalized = column.trim();
      if (!normalized) {
        return;
      }
      updateAliasStrategies((current) =>
        current.map((strategy, idx) => {
          if (idx !== index) {
            return strategy;
          }
          const hasColumn = strategy.columns.includes(normalized);
          const nextColumns = hasColumn
            ? strategy.columns.filter((entry) => entry !== normalized)
            : [...strategy.columns, normalized];
          nextColumns.sort((a, b) => a.localeCompare(b));
          return {
            ...strategy,
            columns: nextColumns,
          };
        }),
      );
    },
    [updateAliasStrategies],
  );

  const handleAliasColumnsChange = useCallback(
    (index: number, value: string) => {
      const entries = value
        .split(',')
        .map((column) => column.trim())
        .filter(Boolean);
      const nextColumns = Array.from(new Set(entries)).sort((a, b) => a.localeCompare(b));
      updateAliasStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, columns: nextColumns } : strategy)),
      );
    },
    [updateAliasStrategies],
  );

  const handleAliasAutoDetectToggle = useCallback(
    (index: number, enabled: boolean) => {
      updateAliasStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, autoDetect: enabled } : strategy)),
      );
    },
    [updateAliasStrategies],
  );

  const renderMultiSelectField = useCallback(
    (parameter: FeatureNodeParameter) => {
      const requiresRecommendations = parameter?.source?.type === 'drop_column_recommendations';
      const isCatalogOnly = !requiresRecommendations;
      const isCatalogLoading = isCatalogOnly && previewState.status === 'loading';
      const isBinningColumnsParameter = isBinningNode && parameter.name === 'columns';
      const isScalingColumnsParameter = isScalingNode && parameter.name === 'columns';

      const binningCandidateColumns = (() => {
        if (!isBinningColumnsParameter) {
          return [] as string[];
        }
        const merged = [...binningAllNumericColumns];
        if (binningRecommendedColumnSet.size > 0) {
          binningRecommendedColumnSet.forEach((column) => {
            if (!merged.includes(column)) {
              merged.push(column);
            }
          });
        }
        selectedColumns.forEach((column) => {
          if (column && !merged.includes(column)) {
            merged.push(column);
          }
        });
        return merged;
      })();

      const displayedColumns = (() => {
        if (isBinningColumnsParameter) {
          return binningCandidateColumns;
        }
        if (isScalingColumnsParameter) {
          return availableColumns.filter((column) => !scalingExcludedColumns.has(column));
        }
        return availableColumns;
      })();

      const renderedColumnOptions = (() => {
        if (isBinningColumnsParameter) {
          if (!normalizedColumnSearch) {
            return binningCandidateColumns;
          }
          return binningCandidateColumns.filter((column) =>
            column.toLowerCase().includes(normalizedColumnSearch)
          );
        }
        if (isScalingColumnsParameter) {
          return filteredColumnOptions.filter((column) => !scalingExcludedColumns.has(column));
        }
        return filteredColumnOptions;
      })();

      const selectionDisplayCount = isBinningColumnsParameter
        ? selectedColumns.filter((column) => !binningExcludedColumns.has(column)).length
        : isScalingColumnsParameter
          ? selectedColumns.filter((column) => !scalingExcludedColumns.has(column)).length
          : selectionCount;

      const allColumnsSelected =
        isCatalogOnly && displayedColumns.length > 0 && selectionDisplayCount >= displayedColumns.length;
      const showMissingMetric = requiresRecommendations || isImputerNode;
      const availableColumnSet = new Set(displayedColumns);
      const suggestionSummaries =
        isCastNode && isCatalogOnly
          ? Object.entries(columnSuggestions)
              .filter(([name, suggestions]) => availableColumnSet.has(name) && suggestions.length > 0)
              .map(([name, suggestions]) => `${name}: ${suggestions.join(', ')}`)
              .slice(0, 4)
          : [];
      const binningExcludedPreview = isBinningColumnsParameter ? Array.from(binningExcludedColumns).slice(0, 4) : [];
      const scalingExcludedPreview = isScalingColumnsParameter ? Array.from(scalingExcludedColumns).slice(0, 4) : [];
      const hasBackendRecommendations = isBinningColumnsParameter && binningRecommendedColumnSet.size > 0;
      const canAddRecommendedColumns = hasBackendRecommendations
        ? Array.from(binningRecommendedColumnSet).some((column) => !selectedColumns.includes(column))
        : false;


      return (
        <div
          key={parameter.name}
          className="canvas-modal__parameter-field canvas-modal__parameter-field--multiselect"
        >
          <div className="canvas-modal__parameter-label">
            <span>{parameter.label}</span>
            {requiresRecommendations && sourceId && (
              <div className="canvas-modal__parameter-actions">
                <button
                  type="button"
                  className="btn btn-outline-secondary"
                  onClick={refreshRecommendations}
                  disabled={isFetchingRecommendations || !hasReachableSource}
                >
                  {isFetchingRecommendations ? 'Refreshing…' : 'Refresh suggestions'}
                </button>
              </div>
            )}
          </div>
          {parameter.description && (
            <p className="canvas-modal__parameter-description">{parameter.description}</p>
          )}

          {suggestionSummaries.length > 0 && (
            <p className="canvas-modal__note">
              Smart suggestions: {suggestionSummaries.join('; ')}
            </p>
          )}

          {isBinningColumnsParameter && binningExcludedColumns.size > 0 && (
            <p className="canvas-modal__note">
              Skipping {binningExcludedColumns.size} non-numeric column
              {binningExcludedColumns.size === 1 ? '' : 's'}
              {binningExcludedPreview.length
                ? ` (examples: ${binningExcludedPreview.join(', ')}${binningExcludedColumns.size > binningExcludedPreview.length ? ', …' : ''})`
                : ''}
              — binning only supports numeric inputs.
            </p>
          )}

          {isScalingColumnsParameter && scalingExcludedColumns.size > 0 && (
            <p className="canvas-modal__note">
              Skipping {scalingExcludedColumns.size} non-numeric column
              {scalingExcludedColumns.size === 1 ? '' : 's'}
              {scalingExcludedPreview.length
                ? ` (examples: ${scalingExcludedPreview.join(', ')}${scalingExcludedColumns.size > scalingExcludedPreview.length ? ', …' : ''})`
                : ''}
              — scaling only supports numeric inputs.
            </p>
          )}

          {requiresRecommendations && (
            <DropMissingColumnsSection
              sourceId={sourceId}
              availableFilters={availableFilters}
              activeFilterId={activeFilterId}
              setActiveFilterId={setActiveFilterId}
              recommendations={recommendations}
              filteredRecommendations={filteredRecommendations}
              isFetchingRecommendations={isFetchingRecommendations}
              recommendationsError={recommendationsError}
              relativeGeneratedAt={relativeGeneratedAt}
              formatSignalName={formatSignalName}
              formatMissingPercentage={formatMissingPercentage}
              getPriorityClass={getPriorityClass}
              getPriorityLabel={getPriorityLabel}
              handleToggleColumn={handleToggleColumn}
              selectedColumns={selectedColumns}
            />
          )}

          <div className="canvas-modal__note">
            Selected columns: <strong>{selectionDisplayCount}</strong>
          </div>

          {selectionDisplayCount > 0 && (
            <div className="canvas-modal__selection-summary">
              <h4>Selected columns</h4>
              <div className="canvas-modal__selection-chips">
                {selectedColumns.map((column) => (
                  <span key={column} className="canvas-modal__selection-chip">
                    {column}
                    <button
                      type="button"
                      onClick={() => handleRemoveColumn(column)}
                      aria-label={`Remove ${column}`}
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="canvas-modal__multi-select-actions">
            {requiresRecommendations && (
              <button
                type="button"
                className="btn btn-outline-secondary"
                onClick={handleApplyAllRecommended}
                disabled={!recommendations.length}
              >
                Use all recommendations
              </button>
            )}
            {hasBackendRecommendations && (
              <button
                type="button"
                className="btn btn-outline-secondary"
                onClick={() => handleBinningApplyColumns(binningRecommendedColumnSet)}
                disabled={!canAddRecommendedColumns}
              >
                Add recommended columns
              </button>
            )}
            {isCatalogOnly && (
              <button
                type="button"
                className="btn btn-outline-secondary"
                onClick={handleSelectAllColumns}
                disabled={!displayedColumns.length || allColumnsSelected}
              >
                Select all columns
              </button>
            )}
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={handleClearColumns}
              disabled={!selectionCount}
            >
              Clear selection
            </button>
          </div>

          <div className="canvas-modal__all-columns">
            <h4>All columns</h4>
            <div className="canvas-modal__all-columns-search">
              <input
                type="text"
                className="canvas-modal__custom-input"
                value={columnSearch}
                onChange={(event) => setColumnSearch(event.target.value)}
                placeholder="Search columns"
                aria-label="Search columns"
              />
            </div>
            <div className="canvas-modal__all-columns-list" role="group" aria-label="All columns">
              {renderedColumnOptions.length ? (
                renderedColumnOptions.map((column) => {
                  const isSelected = selectedColumns.includes(column);
                  const missingValue = Object.prototype.hasOwnProperty.call(columnMissingMap, column)
                    ? columnMissingMap[column]
                    : undefined;
                  const missingLabel = formatMissingPercentage(
                    typeof missingValue === 'number' ? missingValue : null
                  );
                  const columnType = columnTypeMap[column] ?? null;
                  const columnSuggestionList = columnSuggestions[column] ?? [];
                  const rangeMeta: { min: number | null; max: number | null; distinct?: number | null } | undefined =
                    isBinningColumnsParameter
                      ? binningColumnPreviewMap[column]
                      : undefined;
                  const hasRangeMeta = Boolean(
                    rangeMeta &&
                      ((rangeMeta.min !== null && rangeMeta.min !== undefined) ||
                        (rangeMeta.max !== null && rangeMeta.max !== undefined))
                  );
                  const distinctCount = isBinningColumnsParameter ? rangeMeta?.distinct ?? null : null;
                  const hasDistinctMeta =
                    isBinningColumnsParameter && distinctCount !== null && Number.isFinite(distinctCount);
                  const distinctDisplayValue = hasDistinctMeta ? (distinctCount as number) : null;
                  const showTypeMeta = !isBinningColumnsParameter;
                  const shouldShowMetaRow = showTypeMeta || showMissingMetric || hasRangeMeta || hasDistinctMeta;
                  const hasSuggestionHints = showTypeMeta && columnSuggestionList.length > 0;
                  return (
                    <label
                      key={column}
                      className={`canvas-modal__checkbox-item canvas-modal__checkbox-item--compact${
                        isSelected ? ' canvas-modal__checkbox-item--selected' : ''
                      }`}
                    >
                      <input type="checkbox" checked={isSelected} onChange={() => handleToggleColumn(column)} />
                      <div className="canvas-modal__column-option">
                        <span className="canvas-modal__column-option-name">{column}</span>
                        {shouldShowMetaRow && (
                          <div className="canvas-modal__column-option-meta">
                            {showTypeMeta && (
                              <span className="canvas-modal__column-option-metric">
                                Type: {formatColumnType(columnType)}
                              </span>
                            )}
                            {showMissingMetric && (
                              <span className="canvas-modal__column-option-metric">Missing: {missingLabel}</span>
                            )}
                            {hasRangeMeta && (
                              <span className="canvas-modal__column-option-metric">
                                {isBinningColumnsParameter ? 'Sample range' : 'Range'}:{' '}
                                {rangeMeta && rangeMeta.min !== null && rangeMeta.min !== undefined
                                  ? formatNumericStat(rangeMeta.min)
                                  : '—'}{' '}
                                –{' '}
                                {rangeMeta && rangeMeta.max !== null && rangeMeta.max !== undefined
                                  ? formatNumericStat(rangeMeta.max)
                                  : '—'}
                              </span>
                            )}
                            {hasDistinctMeta && distinctDisplayValue !== null && (
                              <span className="canvas-modal__column-option-metric">
                                Distinct sample values: {formatMetricValue(distinctDisplayValue)}
                              </span>
                            )}
                          </div>
                        )}
                        {hasSuggestionHints && (
                          <span className="canvas-modal__column-option-hint">
                            Suggested: {columnSuggestionList.join(', ')}
                          </span>
                        )}
                      </div>
                    </label>
                  );
                })
              ) : (
                <p className="canvas-modal__note">
                  {availableColumns.length
                    ? isBinningColumnsParameter && displayedColumns.length === 0
                      ? 'No eligible columns for binning (non-numeric fields are skipped).'
                      : 'No columns match your search.'
                    : isCatalogLoading
                      ? 'Loading column catalog…'
                      : !sourceId
                        ? 'Select a dataset to load column catalog.'
                        : !hasReachableSource
                          ? 'Connect this step to an upstream output to load column catalog.'
                          : 'Column catalog unavailable for this dataset.'}
                </p>
              )}
            </div>
          </div>
        </div>
      );
    },
    [
      activeFilterId,
      availableColumns,
      availableFilters,
      columnMissingMap,
      columnSearch,
      columnSuggestions,
      columnTypeMap,
  binningAllNumericColumns,
  numericExcludedColumns,
      filteredColumnOptions,
      filteredRecommendations,
      formatSignalName,
      handleApplyAllRecommended,
  handleBinningApplyColumns,
      handleClearColumns,
      handleRemoveColumn,
      handleSelectAllColumns,
      handleToggleColumn,
      hasReachableSource,
      isCastNode,
      isFetchingRecommendations,
      isBinningNode,
  binningRecommendedColumnSet,
      binningColumnPreviewMap,
      isImputerNode,
      isScalingNode,
      binningExcludedColumns,
      previewState.status,
      recommendations,
      recommendationsError,
      relativeGeneratedAt,
      selectedColumns,
      selectionCount,
      setActiveFilterId,
      setColumnSearch,
      scalingExcludedColumns,
      showRecommendations,
      sourceId,
      refreshRecommendations,
      normalizedColumnSearch,
    ]
  );

  const renderParameterField = useCallback(
    (parameter: FeatureNodeParameter) => {
      if (!parameter?.name) {
        return null;
      }

      if (parameter.type === 'multi_select') {
        return renderMultiSelectField(parameter);
      }

      if (parameter.type === 'number') {
        const inputId = `node-${node.id}-${parameter.name}`;
        const value = configState?.[parameter.name];
        const numericValue = typeof value === 'number' ? value : value ?? '';

        return (
          <div key={parameter.name} className="canvas-modal__parameter-field">
            <label htmlFor={inputId} className="canvas-modal__parameter-label">
              <span>{parameter.label}</span>
              {parameter.unit && <span className="canvas-modal__parameter-unit">{parameter.unit}</span>}
            </label>
            {parameter.description && (
              <p className="canvas-modal__parameter-description">{parameter.description}</p>
            )}
            <div className="canvas-modal__parameter-control">
              <input
                id={inputId}
                type="number"
                className="canvas-modal__input"
                value={numericValue}
                min={parameter.min !== undefined ? parameter.min : undefined}
                max={parameter.max !== undefined ? parameter.max : undefined}
                step={parameter.step !== undefined ? parameter.step : 'any'}
                onChange={(event) => handleNumberChange(parameter.name, event.target.value)}
              />
              {parameter.unit && (
                <span className="canvas-modal__parameter-unit">{parameter.unit}</span>
              )}
            </div>
            {thresholdParameterName === parameter.name &&
              normalizedSuggestedThreshold !== null &&
              showRecommendations && (
                <div className="canvas-modal__note">
                  Suggested threshold from EDA:{' '}
                  <strong>{formatMissingPercentage(normalizedSuggestedThreshold)}</strong>
                  <button
                    type="button"
                    className="btn btn-outline-secondary"
                    onClick={handleApplySuggestedThreshold}
                    disabled={!canApplySuggestedThreshold}
                  >
                    {thresholdMatchesSuggestion ? 'Applied' : 'Apply suggestion'}
                  </button>
                </div>
              )}
          </div>
        );
      }

      if (parameter.type === 'boolean') {
        const inputId = `node-${node.id}-${parameter.name}`;
        const checked = Boolean(configState?.[parameter.name]);

        return (
          <div key={parameter.name} className="canvas-modal__parameter-field">
            <div className="canvas-modal__parameter-label">
              <label htmlFor={inputId}>{parameter.label}</label>
            </div>
            {parameter.description && (
              <p className="canvas-modal__parameter-description">{parameter.description}</p>
            )}
            <label className="canvas-modal__boolean-control">
              <input
                id={inputId}
                type="checkbox"
                checked={checked}
                onChange={(event) => handleBooleanChange(parameter.name, event.target.checked)}
              />
              <span>{checked ? 'Enabled' : 'Disabled'}</span>
            </label>
          </div>
        );
      }

      if (parameter.type === 'select') {
        const inputId = `node-${node.id}-${parameter.name}`;
        const options = Array.isArray(parameter.options) ? parameter.options : [];
        const currentValue = configState?.[parameter.name];
        const defaultValue =
          typeof parameter.default === 'string'
            ? parameter.default
            : options.find((option) => option && typeof option.value === 'string')?.value ?? '';
        const value =
          typeof currentValue === 'string' && currentValue.trim().length
            ? currentValue
            : defaultValue;

        return (
          <div key={parameter.name} className="canvas-modal__parameter-field">
            <label htmlFor={inputId} className="canvas-modal__parameter-label">
              {parameter.label}
            </label>
            {parameter.description && (
              <p className="canvas-modal__parameter-description">{parameter.description}</p>
            )}
            <select
              id={inputId}
              className="canvas-modal__input"
              value={value}
              onChange={(event) => handleTextChange(parameter.name, event.target.value)}
            >
              {options.map((option) => (
                <option key={option.value} value={option.value} title={option.description ?? undefined}>
                  {option.label ?? option.value}
                </option>
              ))}
            </select>
          </div>
        );
      }

      if (parameter.type === 'textarea') {
        const inputId = `node-${node.id}-${parameter.name}`;
        
        // Special handling for hyperparameters - convert JSON to simple format
        if (parameter.name === 'hyperparameters') {
          let displayValue = '';
          let internalValue = configState?.[parameter.name];
          
          // Convert JSON to simple key: value format for display
          if (typeof internalValue === 'string' && internalValue.trim()) {
            try {
              const parsed = JSON.parse(internalValue);
              if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                displayValue = Object.entries(parsed)
                  .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
                  .join('\n');
              } else {
                displayValue = internalValue;
              }
            } catch {
              displayValue = internalValue;
            }
          } else if (internalValue && typeof internalValue === 'object' && !Array.isArray(internalValue)) {
            displayValue = Object.entries(internalValue)
              .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
              .join('\n');
          }

          return (
            <div key={parameter.name} className="canvas-modal__parameter-field">
              <label htmlFor={inputId} className="canvas-modal__parameter-label">
                {parameter.label}
              </label>
              <p className="canvas-modal__parameter-description">
                {parameter.description || 'Enter parameters as key: value pairs (one per line)'}
              </p>
              <textarea
                id={inputId}
                className="canvas-modal__input canvas-modal__input--wide"
                value={displayValue}
                placeholder={'n_estimators: 100\nmax_depth: 10\nlearning_rate: 0.01'}
                onChange={(event) => {
                  const text = event.target.value;
                  // Convert simple format to JSON
                  const lines = text.split('\n').filter(line => line.trim());
                  const params: Record<string, any> = {};
                  
                  for (const line of lines) {
                    const colonIndex = line.indexOf(':');
                    if (colonIndex === -1) continue;
                    
                    const key = line.substring(0, colonIndex).trim();
                    const valueStr = line.substring(colonIndex + 1).trim();
                    
                    if (!key) continue;
                    
                    // Try to parse the value
                    try {
                      params[key] = JSON.parse(valueStr);
                    } catch {
                      // If parse fails, treat as string
                      params[key] = valueStr;
                    }
                  }
                  
                  // Store as JSON string
                  handleTextChange(parameter.name, JSON.stringify(params, null, 2));
                }}
                rows={6}
              />
            </div>
          );
        }

        // Default textarea handling for non-hyperparameters
        const value = typeof configState?.[parameter.name] === 'string' ? configState?.[parameter.name] : '';

        return (
          <div key={parameter.name} className="canvas-modal__parameter-field">
            <label htmlFor={inputId} className="canvas-modal__parameter-label">
              {parameter.label}
            </label>
            {parameter.description && (
              <p className="canvas-modal__parameter-description">{parameter.description}</p>
            )}
            <textarea
              id={inputId}
              className="canvas-modal__input canvas-modal__input--wide"
              value={value}
              placeholder={parameter.placeholder ?? ''}
              onChange={(event) => handleTextChange(parameter.name, event.target.value)}
              rows={4}
            />
          </div>
        );
      }

      const inputId = `node-${node.id}-${parameter.name}`;
      const value = configState?.[parameter.name] ?? '';

      return (
        <div key={parameter.name} className="canvas-modal__parameter-field">
          <label htmlFor={inputId} className="canvas-modal__parameter-label">
            {parameter.label}
          </label>
          {parameter.description && (
            <p className="canvas-modal__parameter-description">{parameter.description}</p>
          )}
          <input
            id={inputId}
            type="text"
            className="canvas-modal__input canvas-modal__input--wide"
            value={value}
            placeholder={parameter.placeholder ?? ''}
            onChange={(event) => handleTextChange(parameter.name, event.target.value)}
          />
        </div>
      );
    },
    [
      canApplySuggestedThreshold,
      configState,
      handleApplySuggestedThreshold,
      handleBooleanChange,
      handleNumberChange,
      handleTextChange,
      node.id,
      normalizedSuggestedThreshold,
      renderMultiSelectField,
      showRecommendations,
      thresholdMatchesSuggestion,
      thresholdParameterName,
    ]
  );

  const handleRemoveDuplicatesKeepChange = useCallback(
    (value: KeepStrategy) => {
      setConfigState((previous) => ({
        ...previous,
        keep: value,
      }));
    },
    [setConfigState],
  );

  return (
    <div className="canvas-modal" role="dialog" aria-modal="true" aria-labelledby="node-settings-title">
      <div className="canvas-modal__backdrop" onClick={onClose} />
      <div className="canvas-modal__panel">
        <NodeSettingsHeader
          title={title}
          isDataset={datasetBadge}
          onClose={onClose}
          canResetNode={canResetNode}
          onResetNode={handleResetNode}
        />

        <div className="canvas-modal__body">
          {connectionInfo && (
            <ConnectionRequirementsSection
              connectionInfo={connectionInfo}
              connectedInputHandles={connectedHandleKeys}
              connectedOutputHandles={connectedOutputHandleKeys}
              connectionReady={connectionReady}
            />
          )}
          {showDropMissingRowsSection && (
            <DropMissingRowsSection
              thresholdParameter={thresholdParameter ?? null}
              dropIfAnyParameter={dropRowsAnyParameter}
              renderParameterField={renderParameterField}
            />
          )}
          {dropColumnParameter && (
            <section className="canvas-modal__section">
              <div className="canvas-modal__section-header">
                <h3>Missingness recommendations</h3>
              </div>
              {thresholdParameter && renderParameterField(thresholdParameter)}
              {renderMultiSelectField(dropColumnParameter)}
            </section>
          )}
          {isMissingIndicatorNode && (
            <>
              <MissingIndicatorInsightsSection
                suffix={activeFlagSuffix}
                insights={missingIndicatorInsights}
                formatMissingPercentage={formatMissingPercentage}
              />
              {(missingIndicatorColumnsParameter || missingIndicatorSuffixParameter) && (
                <section className="canvas-modal__section">
                  <div className="canvas-modal__section-header">
                    <h3>Missing indicator settings</h3>
                  </div>
                  <div className="canvas-modal__parameter-list">
                    {missingIndicatorColumnsParameter ? renderParameterField(missingIndicatorColumnsParameter) : null}
                    {missingIndicatorSuffixParameter ? renderParameterField(missingIndicatorSuffixParameter) : null}
                  </div>
                </section>
              )}
            </>
          )}
          {isPreviewNode && (
            <DataSnapshotSection
              previewState={previewState}
              datasetSourceId={sourceId ?? null}
              canTriggerPreview={canTriggerPreview}
              onRefresh={handleRefreshPreview}
              formatCellValue={formatCellValue}
              formatMetricValue={formatMetricValue}
              formatMissingPercentage={formatMissingPercentage}
              formatNumericStat={formatNumericStat}
              formatModeStat={formatModeStat}
            />
          )}
          <DatasetProfileSection
            isDatasetProfileNode={isDatasetProfileNode}
            controller={datasetProfileController}
            formatCellValue={formatCellValue}
            formatNumericStat={formatNumericStat}
            formatMissingPercentage={formatMissingPercentage}
          />
          {dataConsistencyHint && (
            <section className="canvas-modal__section">
              <p className="canvas-modal__note">
                <strong>{dataConsistencyHint.title}.</strong> {dataConsistencyHint.body}
              </p>
            </section>
          )}
          {isTrimWhitespaceNode && (
            <TrimWhitespaceSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={trimWhitespaceColumnsParameter}
              modeParameter={trimWhitespaceModeParameter}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={trimWhitespaceColumnSummary}
              modeDetails={trimWhitespaceModeDetails}
              sampleMap={trimWhitespaceSampleMap}
            />
          )}
          {isRemoveSpecialCharsNode && (
            <RemoveSpecialCharactersSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={removeSpecialColumnsParameter}
              modeParameter={removeSpecialModeParameter}
              replacementParameter={removeSpecialReplacementParameter}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={removeSpecialColumnSummary}
              modeDetails={removeSpecialModeDetails}
              sampleMap={removeSpecialSampleMap}
            />
          )}
          {isReplaceInvalidValuesNode && (
            <ReplaceInvalidValuesSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={replaceInvalidColumnsParameter}
              modeParameter={replaceInvalidModeParameter}
              minValueParameter={replaceInvalidMinValueParameter}
              maxValueParameter={replaceInvalidMaxValueParameter}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={replaceInvalidColumnSummary}
              modeDetails={replaceInvalidModeDetails}
              sampleMap={replaceInvalidSampleMap}
              selectedMode={replaceInvalidMode}
              minValue={replaceInvalidMinValue}
              maxValue={replaceInvalidMaxValue}
            />
          )}
          {isRegexCleanupNode && (
            <RegexCleanupSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={regexCleanupColumnsParameter}
              modeParameter={regexCleanupModeParameter}
              patternParameter={regexCleanupPatternParameter}
              replacementParameter={regexCleanupReplacementParameter}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={regexCleanupColumnSummary}
              modeDetails={regexCleanupModeDetails}
              sampleMap={regexCleanupSampleMap}
              selectedMode={regexCleanupMode}
              replacementValue={regexCleanupReplacementValue}
            />
          )}
          {isNormalizeTextCaseNode && (
            <NormalizeTextCaseSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={normalizeCaseColumnsParameter}
              modeParameter={normalizeCaseModeParameter}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={normalizeCaseColumnSummary}
              modeDetails={normalizeCaseModeDetails}
              sampleMap={normalizeCaseSampleMap}
              selectedMode={normalizeCaseMode}
            />
          )}
          {isFeatureMathNode && (
            <FeatureMathSection
              operations={featureMathOperations}
              summaries={featureMathSummaries}
              collapsed={collapsedFeatureMath}
              onToggleCollapsed={handleToggleFeatureMathOperation}
              onAddOperation={handleAddFeatureMathOperation}
              onDuplicateOperation={handleDuplicateFeatureMathOperation}
              onRemoveOperation={handleRemoveFeatureMathOperation}
              onReorderOperation={handleReorderFeatureMathOperation}
              onOperationChange={handleFeatureMathOperationChange}
              availableColumns={availableColumns}
              signals={featureMathSignals}
              previewStatus={previewState.status}
              errorHandlingParameter={featureMathErrorHandlingParameter}
              allowOverwriteParameter={featureMathAllowOverwriteParameter}
              defaultTimezoneParameter={featureMathDefaultTimezoneParameter}
              epsilonParameter={featureMathEpsilonParameter}
              renderParameterField={renderParameterField}
              canResetNode={canResetNode}
              onResetNode={handleResetNode}
            />
          )}
          {isOrdinalEncodingNode && (
            <OrdinalEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={isFetchingOrdinalEncoding}
              error={ordinalEncodingError}
              suggestions={ordinalEncodingSuggestions}
              metadata={ordinalEncodingMetadata}
              columnsParameter={ordinalEncodingColumnsParameter}
              autoDetectParameter={ordinalEncodingAutoDetectParameter}
              maxCategoriesParameter={ordinalEncodingMaxCategoriesParameter}
              outputSuffixParameter={ordinalEncodingOutputSuffixParameter}
              dropOriginalParameter={ordinalEncodingDropOriginalParameter}
              encodeMissingParameter={ordinalEncodingEncodeMissingParameter}
              handleUnknownParameter={ordinalEncodingHandleUnknownParameter}
              unknownValueParameter={ordinalEncodingUnknownValueParameter}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={handleApplyOrdinalEncodingRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isTargetEncodingNode && (
            <TargetEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={isFetchingTargetEncoding}
              error={targetEncodingError}
              suggestions={targetEncodingSuggestions}
              metadata={targetEncodingMetadata}
              columnsParameter={targetEncodingColumnsParameter}
              targetColumnParameter={targetEncodingTargetColumnParameter}
              autoDetectParameter={targetEncodingAutoDetectParameter}
              maxCategoriesParameter={targetEncodingMaxCategoriesParameter}
              outputSuffixParameter={targetEncodingOutputSuffixParameter}
              dropOriginalParameter={targetEncodingDropOriginalParameter}
              smoothingParameter={targetEncodingSmoothingParameter}
              encodeMissingParameter={targetEncodingEncodeMissingParameter}
              handleUnknownParameter={targetEncodingHandleUnknownParameter}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={handleApplyTargetEncodingRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isDummyEncodingNode && (
            <DummyEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={isFetchingDummyEncoding}
              error={dummyEncodingError}
              suggestions={dummyEncodingSuggestions}
              metadata={dummyEncodingMetadata}
              columnsParameter={dummyEncodingColumnsParameter}
              autoDetectParameter={dummyEncodingAutoDetectParameter}
              maxCategoriesParameter={dummyEncodingMaxCategoriesParameter}
              dropFirstParameter={dummyEncodingDropFirstParameter}
              includeMissingParameter={dummyEncodingIncludeMissingParameter}
              dropOriginalParameter={dummyEncodingDropOriginalParameter}
              prefixSeparatorParameter={dummyEncodingPrefixSeparatorParameter}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={handleApplyDummyEncodingRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isOneHotEncodingNode && (
            <OneHotEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={isFetchingOneHotEncoding}
              error={oneHotEncodingError}
              suggestions={oneHotEncodingSuggestions}
              metadata={oneHotEncodingMetadata}
              columnsParameter={oneHotEncodingColumnsParameter}
              autoDetectParameter={oneHotEncodingAutoDetectParameter}
              maxCategoriesParameter={oneHotEncodingMaxCategoriesParameter}
              dropFirstParameter={oneHotEncodingDropFirstParameter}
              includeMissingParameter={oneHotEncodingIncludeMissingParameter}
              dropOriginalParameter={oneHotEncodingDropOriginalParameter}
              prefixSeparatorParameter={oneHotEncodingPrefixSeparatorParameter}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={handleApplyOneHotEncodingRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isFeatureTargetSplitNode && (
            <FeatureTargetSplitSection
              nodeId={nodeId}
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              previewState={previewState}
              onRefreshPreview={handleRefreshPreview}
              config={featureTargetSplitConfig}
              targetColumnParameter={featureTargetSplitTargetColumnParameter}
              renderParameterField={renderParameterField}
              formatMetricValue={formatMetricValue}
              formatMissingPercentage={formatMissingPercentage}
              canResetNode={canResetNode}
              onResetNode={handleResetNode}
            />
          )}
          {isTrainTestSplitNode && (
            <TrainTestSplitSection
              nodeId={nodeId}
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              previewState={previewState}
              onRefreshPreview={handleRefreshPreview}
              config={trainTestSplitConfig}
              parameters={parameters}
              renderParameterField={renderParameterField}
              formatMetricValue={formatMetricValue}
              canResetNode={canResetNode}
              onResetNode={handleResetNode}
            />
          )}
          {isClassResamplingNode && (
            <ClassResamplingSection
              mode={isClassOversamplingNode ? 'oversampling' : 'undersampling'}
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              previewState={previewState}
              onRefreshPreview={handleRefreshPreview}
              config={resamplingConfig}
              methodParameter={resamplingMethodParameter}
              targetColumnParameter={resamplingTargetColumnParameter}
              samplingStrategyParameter={resamplingSamplingStrategyParameter}
              randomStateParameter={resamplingRandomStateParameter}
              kNeighborsParameter={isClassOversamplingNode ? resamplingKNeighborsParameter : null}
              replacementParameter={isClassUndersamplingNode ? resamplingReplacementParameter : null}
              renderParameterField={renderParameterField}
              formatMetricValue={formatMetricValue}
              formatMissingPercentage={formatMissingPercentage}
              schemaGuard={oversamplingSchemaGuard}
            />
          )}
          {(isTrainModelDraftNode || isHyperparameterTuningNode) && (
            <TrainModelDraftSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              previewState={previewState}
              onRefreshPreview={handleRefreshPreview}
              config={trainModelDraftConfig}
              targetColumnParameter={trainModelTargetColumnParameter}
              problemTypeParameter={trainModelProblemTypeParameter}
              renderParameterField={renderParameterField}
              formatMetricValue={formatMetricValue}
              formatMissingPercentage={formatMissingPercentage}
              schemaColumns={cachedSchemaColumns}
            />
          )}
          {isTrainModelDraftNode && (
            <div style={gatedSectionStyle} aria-disabled={gatedAriaDisabled}>
              <ModelTrainingSection
                nodeId={nodeId}
                sourceId={sourceId}
                graph={graphContext}
                config={trainModelDraftConfig}
                runtimeConfig={trainModelRuntimeConfig}
                cvConfig={trainModelCVConfig}
                draftConfigState={configState}
                renderParameterField={renderParameterField}
                modelTypeParameter={trainModelModelTypeParameter}
                modelTypeOptions={filteredModelTypeOptions}
                hyperparametersParameter={trainModelHyperparametersParameter}
                cvEnabledParameter={trainModelCvEnabledParameter}
                cvStrategyParameter={trainModelCvStrategyParameter}
                cvFoldsParameter={trainModelCvFoldsParameter}
                cvShuffleParameter={trainModelCvShuffleParameter}
                cvRandomStateParameter={trainModelCvRandomStateParameter}
                cvRefitStrategyParameter={trainModelCvRefitStrategyParameter}
                onSaveDraftConfig={handleSave}
              />
            </div>
          )}
          {isModelEvaluationNode && (
            <div style={gatedSectionStyle} aria-disabled={gatedAriaDisabled}>
              <EvaluationPackSection
                nodeId={nodeId}
                sourceId={sourceId}
                graph={graphContext}
                config={configState}
                setConfigState={setConfigState}
                renderParameterField={renderParameterField}
                formatMetricValue={formatMetricValue}
                connectedSplitHandles={evaluationSplitConnectivity}
                canResetNode={canResetNode}
                onResetNode={handleResetNode}
              />
            </div>
          )}
          {isModelRegistryNode && (
            <div style={gatedSectionStyle} aria-disabled={gatedAriaDisabled}>
              <ModelRegistrySection
                nodeId={nodeId}
                sourceId={sourceId}
                graph={graphContext}
                parameters={parameters}
                config={modelRegistryConfig}
                renderParameterField={renderParameterField}
              />
            </div>
          )}
          {isHyperparameterTuningNode && (
            <div style={gatedSectionStyle} aria-disabled={gatedAriaDisabled}>
              <HyperparameterTuningSection
                nodeId={nodeId}
                sourceId={sourceId}
                graph={graphContext}
                config={trainModelDraftConfig}
                runtimeConfig={trainModelRuntimeConfig}
                cvConfig={trainModelCVConfig}
                draftConfigState={configState}
                renderParameterField={renderParameterField}
                modelTypeParameter={trainModelModelTypeParameter}
                modelTypeOptions={filteredModelTypeOptions}
                searchStrategyParameter={hyperparameterTuningSearchStrategyParameter}
                searchIterationsParameter={hyperparameterTuningSearchIterationsParameter}
                searchRandomStateParameter={hyperparameterTuningSearchRandomStateParameter}
                scoringMetricParameter={hyperparameterTuningScoringMetricParameter}
                setDraftConfigState={setConfigState}
                onSaveDraftConfig={handleSave}
              />
            </div>
          )}
          {isLabelEncodingNode && (
            <LabelEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={isFetchingLabelEncoding}
              error={labelEncodingError}
              suggestions={labelEncodingSuggestions}
              metadata={labelEncodingMetadata}
              columnsParameter={labelEncodingColumnsParameter}
              autoDetectParameter={labelEncodingAutoDetectParameter}
              maxUniqueParameter={labelEncodingMaxUniqueParameter}
              outputSuffixParameter={labelEncodingOutputSuffixParameter}
              dropOriginalParameter={labelEncodingDropOriginalParameter}
              missingStrategyParameter={labelEncodingMissingStrategyParameter}
              missingCodeParameter={labelEncodingMissingCodeParameter}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={handleApplyLabelEncodingRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isHashEncodingNode && (
            <HashEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={isFetchingHashEncoding}
              error={hashEncodingError}
              suggestions={hashEncodingSuggestions}
              metadata={hashEncodingMetadata}
              columnsParameter={hashEncodingColumnsParameter}
              autoDetectParameter={hashEncodingAutoDetectParameter}
              maxCategoriesParameter={hashEncodingMaxCategoriesParameter}
              bucketsParameter={hashEncodingBucketsParameter}
              outputSuffixParameter={hashEncodingOutputSuffixParameter}
              dropOriginalParameter={hashEncodingDropOriginalParameter}
              encodeMissingParameter={hashEncodingEncodeMissingParameter}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={handleApplyHashEncodingRecommended}
              canResetNode={canResetNode}
              onResetNode={handleResetNode}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isPolynomialFeaturesNode && (
            <PolynomialFeaturesSection
              columnsParameter={polynomialColumnsParameter}
              autoDetectParameter={polynomialAutoDetectParameter}
              degreeParameter={polynomialDegreeParameter}
              includeBiasParameter={polynomialIncludeBiasParameter}
              interactionOnlyParameter={polynomialInteractionOnlyParameter}
              includeInputFeaturesParameter={polynomialIncludeInputFeaturesParameter}
              outputPrefixParameter={polynomialOutputPrefixParameter}
              renderParameterField={renderParameterField}
              signal={polynomialSignal}
              canResetNode={canResetNode}
              onResetNode={handleResetNode}
            />
          )}
          {isFeatureSelectionNode && (
            <FeatureSelectionSection
              columnsParameter={featureSelectionColumnsParameter}
              autoDetectParameter={featureSelectionAutoDetectParameter}
              targetColumnParameter={featureSelectionTargetColumnParameter}
              methodParameter={featureSelectionMethodParameter}
              scoreFuncParameter={featureSelectionScoreFuncParameter}
              problemTypeParameter={featureSelectionProblemTypeParameter}
              kParameter={featureSelectionKParameter}
              percentileParameter={featureSelectionPercentileParameter}
              alphaParameter={featureSelectionAlphaParameter}
              thresholdParameter={featureSelectionThresholdParameter}
              modeParameter={featureSelectionModeParameter}
              estimatorParameter={featureSelectionEstimatorParameter}
              stepParameter={featureSelectionStepParameter}
              minFeaturesParameter={featureSelectionMinFeaturesParameter}
              maxFeaturesParameter={featureSelectionMaxFeaturesParameter}
              dropUnselectedParameter={featureSelectionDropUnselectedParameter}
              renderParameterField={renderParameterField}
              signal={featureSelectionSignal}
              canResetNode={canResetNode}
              onResetNode={handleResetNode}
            />
          )}
          {isStandardizeDatesNode && (
            <StandardizeDatesSection
              hasSource={Boolean(sourceId)}
              hasReachableSource={hasReachableSource}
              strategies={dateStrategies}
              columnSummary={standardizeDatesColumnSummary}
              columnOptions={dateColumnOptions}
              sampleMap={standardizeDatesSampleMap}
              collapsedStrategies={collapsedStrategies}
              onToggleStrategy={toggleDateStrategySection}
              onRemoveStrategy={handleRemoveDateStrategy}
              onAddStrategy={handleAddDateStrategy}
              onModeChange={handleDateStrategyModeChange}
              onColumnToggle={handleDateStrategyColumnToggle}
              onColumnsChange={handleDateStrategyColumnsChange}
              onAutoDetectToggle={handleDateStrategyAutoDetectToggle}
              modeOptions={DATE_MODE_OPTIONS}
            />
          )}
          {isReplaceAliasesNode && (
            <ReplaceAliasesSection
              hasSource={Boolean(sourceId)}
              hasReachableSource={hasReachableSource}
              strategies={aliasStrategies}
              columnSummary={aliasColumnSummary}
              columnOptions={aliasColumnOptions}
              sampleMap={aliasSampleMap}
              customPairSummary={aliasCustomPairSummary}
              customPairsParameter={replaceAliasesCustomPairsParameter}
              collapsedStrategies={collapsedStrategies}
              onToggleStrategy={toggleAliasStrategySection}
              onRemoveStrategy={handleRemoveAliasStrategy}
              onAddStrategy={handleAddAliasStrategy}
              onModeChange={handleAliasModeChange}
              onAutoDetectToggle={handleAliasAutoDetectToggle}
              onColumnToggle={handleAliasColumnToggle}
              onColumnsChange={handleAliasColumnsChange}
              renderParameterField={renderParameterField}
            />
          )}
          {isImputerNode && (
            <ImputationStrategiesSection
              hasSource={Boolean(sourceId)}
              hasReachableSource={hasReachableSource}
              strategies={imputerStrategies}
              columnOptions={imputerColumnOptions}
              methodOptions={imputationMethodOptions}
              missingFilter={imputerMissingFilter}
              missingFilterMax={imputerMissingSliderMax}
              missingFilterActive={imputerMissingFilterActive}
              filteredOptionCount={imputerFilteredOptionCount}
              collapsedStrategies={collapsedStrategies}
              onToggleStrategy={toggleImputerStrategySection}
              onRemoveStrategy={handleRemoveImputerStrategy}
              onAddStrategy={handleAddImputerStrategy}
              onMethodChange={handleImputerMethodChange}
              onOptionNumberChange={handleImputerOptionNumberChange}
              onColumnsChange={handleImputerColumnsChange}
              onColumnToggle={handleImputerColumnToggle}
              onMissingFilterChange={handleImputerMissingFilterChange}
              formatMissingPercentage={formatMissingPercentage}
              formatNumericStat={formatNumericStat}
              formatModeStat={formatModeStat}
              schemaDiagnostics={imputationSchemaDiagnostics}
            />
          )}
          {isCastNode && (
            <CastColumnTypesSection
              configState={configState}
              setConfigState={setConfigState}
              availableColumns={availableColumns}
              columnTypeMap={columnTypeMap}
              columnSuggestions={columnSuggestions}
              selectedColumns={selectedColumns}
              previewColumns={previewColumns}
              previewStatus={previewState.status}
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
            />
          )}
          {isOutlierNode && (
            <OutlierInsightsSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetchingOutliers={isFetchingOutliers}
              outlierConfig={outlierConfig}
              outlierMethodOptions={outlierMethodOptions}
              outlierMethodDetailMap={outlierMethodDetailMap}
              outlierDefaultDetail={outlierDefaultDetail}
              outlierAutoDetectEnabled={outlierAutoDetectEnabled}
              outlierSelectedCount={outlierSelectedCount}
              outlierDefaultLabel={outlierDefaultLabel}
              outlierOverrideCount={outlierOverrideCount}
              outlierParameterOverrideCount={outlierParameterOverrideCount}
              outlierOverrideExampleSummary={outlierOverrideSummaryDisplay}
              outlierHasOverrides={outlierHasOverrides}
              outlierRecommendationRows={outlierRecommendationRows}
              outlierHasRecommendations={outlierHasRecommendations}
              outlierStatusMessage={outlierStatusMessage}
              outlierError={outlierError}
              outlierSampleSize={outlierSampleSize}
              relativeOutlierGeneratedAt={relativeOutlierGeneratedAt}
              outlierPreviewSignal={outlierPreviewSignal}
              formatMetricValue={formatMetricValue}
              formatNumericStat={formatNumericStat}
              onApplyAllRecommendations={handleOutlierApplyAllRecommendations}
              onClearOverrides={handleOutlierClearOverrides}
              onDefaultMethodChange={handleOutlierDefaultMethodChange}
              onAutoDetectToggle={handleOutlierAutoDetectToggle}
              onOverrideSelect={handleOutlierOverrideSelect}
              onMethodParameterChange={handleOutlierMethodParameterChange}
              onColumnParameterChange={handleOutlierColumnParameterChange}
            />
          )}
          {isScalingNode && (
            <ScalingInsightsSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetchingScaling={isFetchingScaling}
              scalingConfig={scalingConfig}
              scalingMethodOptions={scalingMethodOptions}
              scalingMethodDetailMap={scalingMethodDetailMap}
              scalingDefaultDetail={scalingDefaultDetail}
              scalingAutoDetectEnabled={scalingAutoDetectEnabled}
              scalingSelectedCount={scalingSelectedCount}
              scalingDefaultLabel={scalingDefaultLabel}
              scalingOverrideCount={scalingOverrideCount}
              scalingOverrideExampleSummary={scalingOverrideExampleSummary}
              scalingRecommendationRows={scalingRecommendationRows}
              scalingHasRecommendations={scalingHasRecommendations}
              scalingStatusMessage={scalingStatusMessage}
              scalingError={scalingError}
              scalingSampleSize={scalingSampleSize}
              relativeScalingGeneratedAt={relativeScalingGeneratedAt}
              formatMetricValue={formatMetricValue}
              formatNumericStat={formatNumericStat}
              formatMissingPercentage={formatMissingPercentage}
              onApplyAllRecommendations={handleScalingApplyAllRecommendations}
              onClearOverrides={handleScalingClearOverrides}
              onDefaultMethodChange={handleScalingDefaultMethodChange}
              onAutoDetectToggle={handleScalingAutoDetectToggle}
              onOverrideSelect={handleScalingOverrideSelect}
            />
          )}
          {isBinningNode && (
            <>
              <BinningInsightsSection
                hasSource={Boolean(sourceId)}
                hasReachableSource={hasReachableSource}
                isFetching={isFetchingBinning}
                error={binningError}
                relativeGeneratedAt={relativeBinningGeneratedAt}
                sampleSize={binningSampleSize}
                binningConfig={binningConfig}
                binningDefaultLabel={binningDefaultLabel}
                binningOverrideCount={binningOverrideCount}
                binningOverrideSummary={binningOverrideSummary}
                recommendations={binningInsightsRecommendations}
                excludedColumns={binningInsightsExcludedColumns}
                numericColumnCount={binningAllNumericColumns.length}
                canApplyAllNumeric={canApplyAllBinningNumeric}
                onApplyAllNumeric={handleApplyAllBinningNumeric}
                onApplyStrategies={handleBinningApplyStrategies}
                onClearOverrides={handleBinningClearOverrides}
                customEdgeDrafts={binningCustomEdgeDrafts}
                customLabelDrafts={binningCustomLabelDrafts}
                onOverrideStrategyChange={handleBinningOverrideStrategyChange}
                onOverrideNumberChange={handleBinningOverrideNumberChange}
                onOverrideKbinsEncodeChange={handleBinningOverrideKbinsEncodeChange}
                onOverrideKbinsStrategyChange={handleBinningOverrideKbinsStrategyChange}
                onOverrideClear={handleBinningClearOverride}
                onCustomBinsChange={handleBinningCustomBinsChange}
                onCustomLabelsChange={handleBinningCustomLabelsChange}
                onClearCustomColumn={handleBinningClearCustomColumn}
                formatMetricValue={formatMetricValue}
                formatNumericStat={formatNumericStat}
              />
              <BinNumericColumnsSection
                config={binningConfig}
                fieldIds={binningFieldIds}
                onIntegerChange={handleBinningIntegerChange}
                onBooleanToggle={handleBinningBooleanToggle}
                onSuffixChange={handleBinningSuffixChange}
                onLabelFormatChange={handleBinningLabelFormatChange}
                onMissingStrategyChange={handleBinningMissingStrategyChange}
                onMissingLabelChange={handleBinningMissingLabelChange}
              />
            </>
          )}
          {isSkewnessNode && (
            <SkewnessInsightsSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetchingSkewness={isFetchingSkewness}
              skewnessThreshold={skewnessThreshold}
              skewnessError={skewnessError}
              skewnessViewMode={skewnessViewMode}
              skewnessRecommendedCount={skewnessRecommendedCount}
              skewnessNumericCount={skewnessNumericCount}
              skewnessGroupByMethod={skewnessGroupByMethod}
              skewnessRows={skewnessRows}
              skewnessTableGroups={skewnessTableGroups}
              hasSkewnessAutoRecommendations={hasSkewnessAutoRecommendations}
              skewnessTransformationsCount={skewnessTransformationsCount}
              isFetchingRecommendations={isFetchingRecommendations}
              getSkewnessMethodLabel={getSkewnessMethodLabel}
              getSkewnessMethodStatus={getSkewnessMethodStatus}
              onApplyRecommendations={handleApplySkewnessRecommendations}
              onViewModeChange={setSkewnessViewMode}
              onGroupByToggle={setSkewnessGroupByMethod}
              onOverrideChange={handleSkewnessOverrideChange}
              onClearSelections={clearSkewnessTransformations}
            />
          )}
          {isSkewnessDistributionNode && (
            <section className="canvas-modal__section">
              <div className="canvas-modal__section-header">
                <h3>Skewness distributions</h3>
                <div className="canvas-modal__section-actions">
                  <div className="canvas-skewness__segmented" role="group" aria-label="Distribution view">
                    <button
                      type="button"
                      className="canvas-skewness__segmented-button"
                      data-active={skewnessDistributionView === 'before'}
                      onClick={() => setSkewnessDistributionView('before')}
                      disabled={!skewnessDistributionCards.length}
                    >
                      Before
                    </button>
                    <button
                      type="button"
                      className="canvas-skewness__segmented-button"
                      data-active={skewnessDistributionView === 'after'}
                      onClick={() => setSkewnessDistributionView('after')}
                      disabled={!skewnessDistributionCards.length}
                    >
                      After
                    </button>
                  </div>
                </div>
              </div>
              {skewnessThreshold !== null && (
                <p className="canvas-modal__note">
                  Columns with |skewness| ≥ {skewnessThreshold.toFixed(2)} are visualized below.
                </p>
              )}
              {isFetchingSkewness && skewnessDistributionCards.length === 0 && <p className="canvas-modal__note">Loading skewness distributions…</p>}
              {!isFetchingSkewness && skewnessError && (
                <p className="canvas-modal__note canvas-modal__note--error">{skewnessError}</p>
              )}
              {(!isFetchingSkewness || skewnessDistributionCards.length > 0) && !skewnessError && (
                skewnessDistributionCards.length ? (
                  <div className="canvas-skewness__distribution-grid">
                    {skewnessDistributionCards.map((card) => {
                      const renderDistributionSection = (
                        distribution: SkewnessColumnDistribution,
                        sectionLabel: string,
                        methodLabel?: string | null,
                      ) => {
                        const validSamples = Math.max(0, distribution.sample_size || 0);
                        const missingSamples = Math.max(0, distribution.missing_count || 0);
                        return (
                          <div className="canvas-skewness__distribution-block" key={sectionLabel}>
                            <div className="canvas-skewness__distribution-block-title">
                              <span>{sectionLabel}</span>
                              {methodLabel ? (
                                <span className="canvas-skewness__distribution-block-method">{methodLabel}</span>
                              ) : null}
                            </div>
                            <HistogramSparkline
                              counts={distribution.counts}
                              binEdges={distribution.bin_edges}
                              className="canvas-skewness__histogram"
                            />
                            <div className="canvas-skewness__distribution-stats">
                              <div>
                                <span className="canvas-skewness__stat-label">Min</span>
                                <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.minimum)}</span>
                              </div>
                              <div>
                                <span className="canvas-skewness__stat-label">Median</span>
                                <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.median)}</span>
                              </div>
                              <div>
                                <span className="canvas-skewness__stat-label">Max</span>
                                <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.maximum)}</span>
                              </div>
                              <div>
                                <span className="canvas-skewness__stat-label">Mean</span>
                                <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.mean)}</span>
                              </div>
                              <div>
                                <span className="canvas-skewness__stat-label">Std dev</span>
                                <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.stddev)}</span>
                              </div>
                              <div>
                                <span className="canvas-skewness__stat-label">Valid</span>
                                <span className="canvas-skewness__stat-value">{validSamples.toLocaleString()}</span>
                              </div>
                              <div>
                                <span className="canvas-skewness__stat-label">Missing</span>
                                <span className="canvas-skewness__stat-value">{missingSamples.toLocaleString()}</span>
                              </div>
                            </div>
                          </div>
                        );
                      };

                      const afterDistribution = card.distributionAfter;
                      const showBefore = skewnessDistributionView === 'before';
                      const showAfter = skewnessDistributionView === 'after' && Boolean(afterDistribution);
                      const showAfterPlaceholder = skewnessDistributionView === 'after' && !afterDistribution;

                      const footnoteMessage = skewnessDistributionView === 'before'
                        ? 'Showing the original distribution returned by the skewness analysis.'
                        : afterDistribution
                          ? 'Showing the recomputed distribution after the applied transform.'
                          : 'A transformed distribution is not available yet for this column.';

                      return (
                        <article
                          key={`skewness-dist-${card.column}`}
                          className="canvas-skewness__distribution-card"
                          aria-label={`Distribution for ${card.column}`}
                        >
                          <header className="canvas-skewness__distribution-header">
                            <div>
                              <h4>{card.column}</h4>
                              <div className="canvas-skewness__distribution-tags">
                                {card.directionLabel && (
                                  <span className="canvas-skewness__chip canvas-skewness__chip--muted">{card.directionLabel}</span>
                                )}
                                {card.magnitudeLabel && (
                                  <span className="canvas-skewness__chip canvas-skewness__chip--muted">{card.magnitudeLabel}</span>
                                )}
                                {card.recommendedLabel && (
                                  <span className="canvas-skewness__chip canvas-skewness__chip--recommended">
                                    Suggested: {card.recommendedLabel}
                                  </span>
                                )}
                                {card.appliedLabel && (
                                  <span className="canvas-skewness__chip canvas-skewness__chip--applied">
                                    Applied: {card.appliedLabel}
                                  </span>
                                )}
                              </div>
                            </div>
                            <div className="canvas-skewness__distribution-skew">
                              <span>Skewness</span>
                              <strong>{card.skewness !== null ? formatMetricValue(card.skewness, 2) : '—'}</strong>
                            </div>
                          </header>
                          {card.summary && (
                            <p className="canvas-skewness__distribution-summary">{card.summary}</p>
                          )}
                          <div className="canvas-skewness__distribution-comparison">
                            {showBefore
                              ? renderDistributionSection(card.distributionBefore, 'Before transform')
                              : null}
                            {showAfter && afterDistribution
                              ? renderDistributionSection(afterDistribution, 'After transform', card.appliedLabel)
                              : null}
                            {showAfterPlaceholder ? (
                              <div className="canvas-skewness__distribution-empty">
                                After transform data isn’t available yet. Apply a skewness method and rerun the node to
                                generate this view.
                              </div>
                            ) : null}
                          </div>
                          <footer className="canvas-skewness__distribution-footnote">
                            Histogram bins are based on the sampled rows returned by the skewness analysis. {footnoteMessage}
                          </footer>
                        </article>
                      );
                    })}
                  </div>
                ) : (
                  <p className="canvas-modal__note">
                    No columns met the skewness threshold for visualization. Refresh with a larger sample if needed.
                  </p>
                )
              )}
            </section>
          )}
          {isBinnedDistributionNode && (
            <BinnedDistributionSection
              cards={binnedDistributionCards}
              selectedPreset={binnedSamplePreset}
              onSelectPreset={(value) => setBinnedSamplePreset(value)}
              isFetching={isFetchingBinnedDistribution}
              sampleSize={binnedSampleSize}
              relativeGeneratedAt={relativeBinnedGeneratedAt}
              error={binnedDistributionError}
              onRefresh={handleRefreshBinnedDistribution}
              canRefresh={Boolean(sourceId)}
            />
          )}
          {isRemoveDuplicatesNode && (
            <RemoveDuplicatesSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              removeDuplicatesColumnsParameter={removeDuplicatesColumnsParameter}
              removeDuplicatesKeepParameter={removeDuplicatesKeepParameter}
              renderMultiSelectField={renderMultiSelectField}
              removeDuplicatesKeepSelectId={removeDuplicatesKeepSelectId}
              removeDuplicatesKeep={removeDuplicatesKeep}
              onKeepChange={handleRemoveDuplicatesKeepChange}
            />
          )}
          {isDataConsistencyNode && dataConsistencyParameters.length > 0 && (
            <section className="canvas-modal__section">
              <div className="canvas-modal__section-header">
                <h3>Data consistency settings</h3>
                {canResetNode && (
                  <div className="canvas-modal__section-actions">
                    <button
                      type="button"
                      className="btn btn-outline-secondary"
                      onClick={handleResetNode}
                    >
                      Reset node
                    </button>
                  </div>
                )}
              </div>
              <div className="canvas-modal__parameter-list">
                {dataConsistencyParameters.map((parameter) => renderParameterField(parameter))}
              </div>
            </section>
          )}
          {isTransformerAuditNode && (
            <TransformerAuditSection
              signal={transformerAuditSignal}
              previewStatus={previewState.status}
              hasSource={Boolean(sourceId)}
              hasReachableSource={hasReachableSource}
              onRefreshPreview={handleRefreshPreview}
            />
          )}
          <section className="canvas-modal__section">
            <h3>Node details</h3>
            {metadata.length ? (
              <dl className="canvas-modal__metadata">
                {metadata.map((entry, index) => (
                  <div key={`${entry.label}-${index}`} className="canvas-modal__metadata-row">
                    <dt>{entry.label}</dt>
                    <dd>{entry.value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="canvas-modal__empty">No metadata available for this node.</p>
            )}
          </section>


        </div>

        <NodeSettingsFooter
          onClose={onClose}
          canResetNode={canResetNode}
          onResetNode={handleResetNode}
          showSaveButton={showSaveButton}
          canSave={canSave}
          onSave={handleSave}
          isBusy={hasActiveAsyncWork}
          busyLabel={footerBusyLabel}
        />
      </div>
    </div>
  );
};
