import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Node } from 'react-flow-renderer';
import {
  FeatureNodeParameter,
  OutlierMethodName,
  OutlierNodeSignal,
  TransformerAuditNodeSignal,
  PipelinePreviewSchema,
  SkewnessColumnDistribution,
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
import { DatasetProfileSection } from './node-settings/nodes/dataset/datasetProfile';
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
  normalizeImputationStrategies,
  sanitizeOptionsForMethod,
  serializeImputationStrategies,
  type ImputationMethodOption,
  type ImputationStrategyConfig,
  type ImputationStrategyMethod,
  type ImputationStrategyOptions,
} from './node-settings/nodes/imputation/imputationSettings';
import { ImputationStrategiesSection } from './node-settings/nodes/imputation/ImputationStrategiesSection';
import { SCALING_METHOD_ORDER } from './node-settings/nodes/scaling/scalingSettings';
import { ScalingInsightsSection } from './node-settings/nodes/scaling/ScalingInsightsSection';
import { SkewnessInsightsSection } from './node-settings/nodes/skewness/SkewnessInsightsSection';
import {
  dedupeSkewnessTransformations,
  normalizeSkewnessTransformations,
  type SkewnessTransformationConfig,
} from './node-settings/nodes/skewness/skewnessSettings';
import { SkewnessDistributionSection } from './node-settings/nodes/skewness/SkewnessDistributionSection';
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
import { ClassResamplingSection } from './node-settings/nodes/resampling/ClassResamplingSection';
import { useCatalogFlags } from './node-settings/hooks/useCatalogFlags';
import { useScalingInsights } from './node-settings/hooks/useScalingInsights';
import { useBinningInsights } from './node-settings/hooks/useBinningInsights';
import { useSkewnessInsights } from './node-settings/hooks/useSkewnessInsights';
import { useBinnedDistribution } from './node-settings/hooks/useBinnedDistribution';
import { useDropColumnRecommendations } from './node-settings/hooks/useDropColumnRecommendations';
import { useOutlierRecommendations } from './node-settings/hooks/useOutlierRecommendations';
import { useNumericColumnAnalysis } from './node-settings/hooks/useNumericColumnAnalysis';
import { useImputationConfiguration } from './node-settings/hooks/useImputationConfiguration';
import { useModelingConfiguration } from './node-settings/hooks/useModelingConfiguration';
import {
  useSkewnessConfiguration,
  type SkewnessDistributionView,
} from './node-settings/hooks/useSkewnessConfiguration';
import {
  arraysAreEqual,
  normalizeConfigBoolean,
  pickAutoDetectValue,
  stableStringify,
} from './node-settings/utils/configParsers';
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
import { useFilteredParameters } from './node-settings/hooks/useFilteredParameters';
import { useNodeConfigState } from './node-settings/hooks/useNodeConfigState';
import { useParameterHandlers } from './node-settings/hooks/useParameterHandlers';
import { useEncodingRecommendationsState } from './node-settings/hooks/useEncodingRecommendationsState';
import { useNumericAnalysisState } from './node-settings/hooks/useNumericAnalysisState';
import { useDataCleaningState } from './node-settings/hooks/useDataCleaningState';
import { useGraphTopology } from './node-settings/hooks/useGraphTopology';
import { useNodePreview } from './node-settings/hooks/useNodePreview';
import { useColumnCatalogState } from './node-settings/hooks/useColumnCatalogState';
import { usePreviewColumnTypes, usePreviewAvailableColumns } from './node-settings/hooks/usePreviewColumnMetadata';
import { usePreviewSignals } from './node-settings/hooks/usePreviewSignals';
import { useFeatureSelectionAutoConfig } from './node-settings/hooks/useFeatureSelectionAutoConfig';
import { useColumnSelectionHandlers } from './node-settings/hooks/useColumnSelectionHandlers';
import { useNodeSaveHandlers } from './node-settings/hooks/useNodeSaveHandlers';
import { useNodeParameters } from './node-settings/hooks/useNodeParameters';
import { useResetPermissions } from './node-settings/hooks/useResetPermissions';
import { useSchemaDiagnostics } from './node-settings/hooks/useSchemaDiagnostics';
import { usePreviewData } from './node-settings/hooks/usePreviewData';
import { useAsyncBusyLabel } from './node-settings/hooks/useAsyncBusyLabel';
import { useInsightSummaries } from './node-settings/hooks/useInsightSummaries';
import { useThresholdRecommendations } from './node-settings/hooks/useThresholdRecommendations';
import { useDatasetProfiling } from './node-settings/hooks/useDatasetProfiling';
import { NodeSettingsParameterField } from './node-settings/fields/NodeSettingsParameterField';
import { NodeSettingsMultiSelectField } from './node-settings/fields/NodeSettingsMultiSelectField';
import { useImputationStrategyHandlers } from './node-settings/hooks/useImputationStrategyHandlers';

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

  const {
    parameters,
    getParameter,
    getParameterIf,
    requiresColumnCatalog,
    dropColumnParameter,
  } = useNodeParameters(node);

  const dataConsistencyHint = useMemo(
    () => DATA_CONSISTENCY_GUIDANCE[catalogType] ?? null,
    [catalogType]
  );

  const nodeId = typeof node?.id === 'string' ? node.id : '';
  const footerResetFlags = [
    isFeatureMathNode,
    isFeatureTargetSplitNode,
    isTrainTestSplitNode,
    isModelEvaluationNode,
    isHashEncodingNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
    isDataConsistencyNode,
    isTrainModelDraftNode,
    isHyperparameterTuningNode,
    isClassResamplingNode,
    isLabelEncodingNode,
    isTargetEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
    isOrdinalEncodingNode,
    isImputerNode,
    isMissingIndicatorNode,
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isReplaceInvalidValuesNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isStandardizeDatesNode,
    isCastNode,
    isRemoveDuplicatesNode,
    isDropMissingNode,
    isOutlierNode,
  ];

  const { canResetNode, headerCanResetNode, footerCanResetNode } = useResetPermissions({
    isResetAvailable,
    isDataset,
    defaultConfigTemplate,
    footerResetFlags,
    isScalingNode,
    isBinningNode,
    isBinnedDistributionNode,
    isSkewnessDistributionNode,
  });

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

  // requiresColumnCatalog & dropColumnParameter provided by useNodeParameters

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

  const {
    graphContext,
    graphNodes,
    graphNodeCount,
    upstreamNodeIds,
    upstreamTargetColumn,
    hasReachableSource,
  } = useGraphTopology({
    graphSnapshot: graphSnapshot ?? null,
    nodeId,
    isDataset,
  });

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

  const {
    cachedSchemaColumns,
    oversamplingSchemaGuard,
    imputationSchemaDiagnostics,
    skipPreview,
  } = useSchemaDiagnostics({
    cachedPreviewSchema,
    isClassOversamplingNode,
    resamplingTargetColumn: resamplingConfig?.targetColumn ?? null,
    isImputerNode,
    imputerStrategies,
  });

  const {
    availableColumns,
    setAvailableColumns,
    columnSearch,
    setColumnSearch,
    columnMissingMap,
    setColumnMissingMap,
    columnTypeMap,
    setColumnTypeMap,
    columnSuggestions,
    setColumnSuggestions,
    imputerMissingFilter,
    setImputerMissingFilter,
  } = useColumnCatalogState({
    requiresColumnCatalog,
    isImputerNode,
    nodeId,
    sourceId,
    hasReachableSource,
  });

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
    isFetchingLabelEncoding,
    labelEncodingError,
    labelEncodingSuggestions,
    labelEncodingMetadata,
    isFetchingTargetEncoding,
    targetEncodingError,
    targetEncodingSuggestions,
    targetEncodingMetadata,
    isFetchingHashEncoding,
    hashEncodingError,
    hashEncodingSuggestions,
    hashEncodingMetadata,
    isFetchingOrdinalEncoding,
    ordinalEncodingError,
    ordinalEncodingSuggestions,
    ordinalEncodingMetadata,
    isFetchingDummyEncoding,
    dummyEncodingError,
    dummyEncodingSuggestions,
    dummyEncodingMetadata,
    isFetchingOneHotEncoding,
    oneHotEncodingError,
    oneHotEncodingSuggestions,
    oneHotEncodingMetadata,
    handleApplyLabelEncodingRecommended,
    handleApplyTargetEncodingRecommended,
    handleApplyHashEncodingRecommended,
    handleApplyOrdinalEncodingRecommended,
    handleApplyDummyEncodingRecommended,
    handleApplyOneHotEncodingRecommended,
  } = useEncodingRecommendationsState({
    isLabelEncodingNode,
    isTargetEncodingNode,
    isHashEncodingNode,
    isOrdinalEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
    sourceId,
    hasReachableSource,
    graphContext,
    node,
    configState,
    setConfigState,
    nodeChangeVersion,
  });

  const [binnedSamplePreset, setBinnedSamplePreset] = useState<BinnedSamplePresetValue>('500');

  const [collapsedStrategies, setCollapsedStrategies] = useState<Set<number>>(() => new Set());
  const [collapsedFeatureMath, setCollapsedFeatureMath] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    setCollapsedStrategies(new Set());
  }, [stableInitialConfig]);

  const {
    isPreviewNode,
    isDatasetProfileNode,
    datasetProfileController,
    profileState,
  } = useDatasetProfiling({
    node,
    graphContext,
    hasReachableSource,
    sourceId,
    formatRelativeTime,
  });

  const {
    previewState,
    refreshPreview,
    clearCachedPreviewSchema,
    canTriggerPreview,
  } = useNodePreview({
    graphSnapshot: graphSnapshot ?? null,
    graphNodeCount,
    hasReachableSource,
    sourceId,
    node,
    nodeId,
    previewSignature,
    skipPreview,
    cachedPreviewSchema,
    setCachedPreviewSchema,
    isPreviewNode,
    isFeatureMathNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
    isTrainTestSplitNode,
    isOutlierNode,
    isTransformerAuditNode,
  });

  const {
    featureMathSignals,
    polynomialSignal,
    featureSelectionSignal,
  } = usePreviewSignals({
    previewState,
    nodeId,
    isFeatureMathNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
  });

  useFeatureSelectionAutoConfig({
    isFeatureSelectionNode,
    featureSelectionSignal,
    upstreamTargetColumn,
    setConfigState,
  });

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
    clearCachedPreviewSchema();
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
    clearCachedPreviewSchema,
    refreshPreview,
    refreshScaling,
  ]);

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

  const { previewColumns, previewColumnStats, previewSampleRows } = usePreviewData(previewState);

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

  const canRefreshSkewnessDistributions = Boolean(sourceId) && hasReachableSource;

  const nodeColumns = useMemo(() => ensureArrayOfString(node?.data?.columns), [node?.data?.columns]);

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
    trimWhitespaceColumnSummary,
    trimWhitespaceSampleMap,
    trimWhitespaceModeDetails,
    removeSpecialColumnSummary,
    removeSpecialSampleMap,
    removeSpecialModeDetails,
    removeSpecialMode,
    regexCleanupColumnSummary,
    regexCleanupSampleMap,
    regexCleanupModeDetails,
    regexCleanupMode,
    regexCleanupReplacementValue,
    normalizeCaseColumnSummary,
    normalizeCaseSampleMap,
    normalizeCaseModeDetails,
    normalizeCaseMode,
    replaceInvalidMode,
    replaceInvalidModeDetails,
    replaceInvalidSampleMap,
    replaceInvalidColumnSummary,
    replaceInvalidMinValue,
    replaceInvalidMaxValue,
    standardizeDatesMode,
    dateStrategies,
    standardizeDatesSampleMap,
    standardizeDatesColumnSummary,
    dateColumnOptions,
  } = useDataCleaningState({
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isReplaceInvalidValuesNode,
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
    outlierSampleSize,
    relativeOutlierGeneratedAt,
    outlierHasOverrides,
    outlierOverrideSummaryDisplay,
    binningConfig,
    binningSelectedCount,
    binningDefaultLabel,
    binningOverrideColumns,
    binningOverrideCount,
    binningOverrideSummary,
    binningFieldIds,
    binningCustomEdgeDrafts,
    setBinningCustomEdgeDrafts,
    binningCustomLabelDrafts,
    setBinningCustomLabelDrafts,
    selectedColumns,
    binningInsightsRecommendations,
    binningRecommendedColumnSet,
    binningAllNumericColumns,
    binningNumericColumnsNotSelected,
    canApplyAllBinningNumeric,
    binningExcludedColumns,
    binningColumnPreviewMap,
    manualBoundColumns,
    manualRangeFallbackMap,
    binningInsightsExcludedColumns,
    handleScalingDefaultMethodChange,
    handleScalingAutoDetectToggle,
    setScalingColumnMethod,
    handleScalingClearOverrides,
    handleScalingApplyAllRecommendations,
    handleScalingSkipColumn,
    handleScalingUnskipColumn,
    handleScalingOverrideSelect,
    handleOutlierDefaultMethodChange,
    handleOutlierAutoDetectToggle,
    setOutlierColumnMethod,
    handleOutlierClearOverrides,
    handleOutlierApplyAllRecommendations,
    handleOutlierSkipColumn,
    handleOutlierUnskipColumn,
    handleOutlierOverrideSelect,
    handleOutlierMethodParameterChange,
    handleOutlierColumnParameterChange,
    handleBinningIntegerChange,
    handleBinningBooleanToggle,
    handleBinningSuffixChange,
    handleBinningLabelFormatChange,
    handleBinningMissingStrategyChange,
    handleBinningMissingLabelChange,
    handleBinningCustomBinsChange,
    handleBinningCustomLabelsChange,
    handleBinningClearCustomColumn,
    updateBinningColumnOverride,
    handleBinningOverrideStrategyChange,
    handleBinningOverrideNumberChange,
    handleBinningOverrideKbinsEncodeChange,
    handleBinningOverrideKbinsStrategyChange,
    handleBinningClearOverride,
    handleBinningClearOverrides,
    handleBinningApplyStrategies,
    handleBinningApplyColumns,
    handleApplyAllBinningNumeric,
  } = useNumericAnalysisState({
    isScalingNode,
    isOutlierNode,
    isBinningNode,
    configState,
    setConfigState,
    nodeId,
    numericExcludedColumns,
    scalingData,
    outlierData,
    binningData,
    availableColumns,
    previewSampleRows,
  });

  usePreviewColumnTypes({
    previewState,
    previewSampleRows,
    activeFlagSuffix: activeFlagSuffix ?? '',
    setColumnTypeMap,
    setColumnSuggestions,
  });

  usePreviewAvailableColumns({
    previewState,
    activeFlagSuffix: activeFlagSuffix ?? '',
    hasReachableSource,
    requiresColumnCatalog,
    nodeColumns,
    selectedColumns,
    setAvailableColumns,
    setColumnMissingMap,
  });

  const removeDuplicatesKeepSelectId = `${nodeId || 'node'}-remove-duplicates-keep`;

  const removeDuplicatesKeep = useMemo<KeepStrategy>(() => {
    const raw = typeof configState?.keep === 'string' ? configState.keep.trim().toLowerCase() : '';
    if (raw === 'last' || raw === 'none') {
      return raw;
    }
    return DEFAULT_KEEP_STRATEGY;
  }, [configState?.keep]);

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

  const {
    normalizedSuggestedThreshold,
    thresholdMatchesSuggestion,
    canApplySuggestedThreshold,
    handleApplySuggestedThreshold,
  } = useThresholdRecommendations({
    suggestedThreshold,
    thresholdParameterName,
    configState,
    handleParameterChange,
  });

  const showDropMissingRowsSection =
    isDropMissingRowsNode && Boolean(thresholdParameter || dropRowsAnyParameter);

  const {
    handleManualBoundChange,
    handleClearManualBound,
    handleToggleColumn,
    handleApplyAllRecommended,
    handleSelectAllColumns,
    handleClearColumns,
    handleRemoveColumn,
  } = useColumnSelectionHandlers({
    isBinningNode,
    isScalingNode,
    setConfigState,
    binningExcludedColumns,
    scalingExcludedColumns,
    availableColumns,
    recommendations,
  });

  const { handleSave, handleResetNode } = useNodeSaveHandlers({
    configState,
    isBinningNode,
    isScalingNode,
    nodeId,
    onUpdateConfig,
    onClose,
    sourceId,
    graphSnapshot: graphSnapshot ?? null,
    isInspectionNode,
    onUpdateNodeData,
    setConfigState,
    canResetNode,
    defaultConfigTemplate,
    onResetConfig,
  });

  const selectionCount = selectedColumns.length;
  const {
    relativeGeneratedAt,
    relativeScalingGeneratedAt,
    relativeBinningGeneratedAt,
    relativeBinnedGeneratedAt,
    scalingSampleSize,
    binningSampleSize,
    binnedSampleSize,
  } = useInsightSummaries({
    recommendationsGeneratedAt,
    scalingData,
    binningData,
    binnedDistributionData,
  });
  const showRecommendations = hasDropColumnSource && Boolean(sourceId) && hasReachableSource;
  const showSaveButton = !datasetBadge && !isInspectionNode;
  const canSave = showSaveButton && stableInitialConfig !== stableCurrentConfig;
  const isProfileLoading = profileState.status === 'loading';
  const isPreviewLoading = previewState.status === 'loading';
  const { hasActiveAsyncWork, busyLabel, footerBusyLabel } = useAsyncBusyLabel({
    isProfileLoading,
    isPreviewLoading,
    isFetchingScaling,
    isFetchingBinning,
    isFetchingHashEncoding,
    isFetchingBinnedDistribution,
    isFetchingRecommendations,
    isPreviewNode,
  });

  const {
    updateImputerStrategies,
    handleAddImputerStrategy,
    handleRemoveImputerStrategy,
    toggleImputerStrategySection,
    handleImputerMethodChange,
    handleImputerOptionNumberChange,
    handleImputerColumnsChange,
    handleImputerColumnToggle,
    handleImputerMissingFilterChange,
  } = useImputationStrategyHandlers({
    setConfigState,
    imputationMethodValues,
    imputationMethodOptions,
    imputerStrategyCount,
    setCollapsedStrategies,
    setImputerMissingFilter,
  });

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
      return (
        <NodeSettingsMultiSelectField
          parameter={parameter}
          previewStateStatus={previewState.status}
          isBinningNode={isBinningNode}
          isScalingNode={isScalingNode}
          binningAllNumericColumns={binningAllNumericColumns}
          binningRecommendedColumnSet={binningRecommendedColumnSet}
          selectedColumns={selectedColumns}
          availableColumns={availableColumns}
          scalingExcludedColumns={scalingExcludedColumns}
          normalizedColumnSearch={normalizedColumnSearch}
          filteredColumnOptions={filteredColumnOptions}
          binningExcludedColumns={binningExcludedColumns}
          selectionCount={selectionCount}
          isCastNode={isCastNode}
          columnSuggestions={columnSuggestions}
          sourceId={sourceId}
          isFetchingRecommendations={isFetchingRecommendations}
          hasReachableSource={hasReachableSource}
          refreshRecommendations={refreshRecommendations}
          availableFilters={availableFilters}
          activeFilterId={activeFilterId}
          setActiveFilterId={setActiveFilterId}
          recommendations={recommendations}
          filteredRecommendations={filteredRecommendations}
          recommendationsError={recommendationsError}
          relativeGeneratedAt={relativeGeneratedAt}
          formatSignalName={formatSignalName}
          handleToggleColumn={handleToggleColumn}
          handleRemoveColumn={handleRemoveColumn}
          handleApplyAllRecommended={handleApplyAllRecommended}
          handleBinningApplyColumns={handleBinningApplyColumns}
          handleSelectAllColumns={handleSelectAllColumns}
          handleClearColumns={handleClearColumns}
          columnSearch={columnSearch}
          setColumnSearch={setColumnSearch}
          columnMissingMap={columnMissingMap}
          columnTypeMap={columnTypeMap}
          binningColumnPreviewMap={binningColumnPreviewMap}
          isImputerNode={isImputerNode}
          showRecommendations={showRecommendations}
        />
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
      return (
        <NodeSettingsParameterField
          parameter={parameter}
          nodeId={node.id}
          configState={configState}
          handleNumberChange={handleNumberChange}
          handleBooleanChange={handleBooleanChange}
          handleTextChange={handleTextChange}
          thresholdParameterName={thresholdParameterName}
          normalizedSuggestedThreshold={normalizedSuggestedThreshold}
          showRecommendations={showRecommendations}
          canApplySuggestedThreshold={canApplySuggestedThreshold}
          thresholdMatchesSuggestion={thresholdMatchesSuggestion}
          handleApplySuggestedThreshold={handleApplySuggestedThreshold}
          renderMultiSelect={renderMultiSelectField}
        />
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
          canResetNode={headerCanResetNode}
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
            <SkewnessDistributionSection
              skewnessThreshold={skewnessThreshold}
              isFetchingSkewness={isFetchingSkewness}
              skewnessError={skewnessError}
              skewnessDistributionCards={skewnessDistributionCards}
              skewnessDistributionView={skewnessDistributionView}
              setSkewnessDistributionView={setSkewnessDistributionView}
              onRefresh={refreshSkewness}
              canRefresh={canRefreshSkewnessDistributions}
            />
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
          canResetNode={footerCanResetNode}
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
