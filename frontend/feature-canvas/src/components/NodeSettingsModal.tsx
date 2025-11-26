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
} from './node-settings/nodes/drop_col_rows/DropMissingColumnsSection';
import { useDropMissingColumns } from './node-settings/hooks/useDropMissingColumns';
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
  type AliasMode,
  type AliasStrategyConfig,
} from './node-settings/nodes/replace_aliases/replaceAliasesSettings';
import { StandardizeDatesSection } from './node-settings/nodes/standardize_date/StandardizeDatesSection';
import {
  DATE_MODE_OPTIONS,
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
  buildFeatureMathSummaries,
  FeatureMathOperationDraft,
  normalizeFeatureMathOperations,
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
import { useBinnedDistributionCards } from './node-settings/hooks/useBinnedDistributionCards';
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
import { useNodeSpecificParameters } from './node-settings/hooks/useNodeSpecificParameters';
import { useResetPermissions } from './node-settings/hooks/useResetPermissions';
import { useSchemaDiagnostics } from './node-settings/hooks/useSchemaDiagnostics';
import { usePreviewData } from './node-settings/hooks/usePreviewData';
import { useAsyncBusyLabel } from './node-settings/hooks/useAsyncBusyLabel';
import { useInsightSummaries } from './node-settings/hooks/useInsightSummaries';
import { useThresholdRecommendations } from './node-settings/hooks/useThresholdRecommendations';
import { useFeatureMathState } from './node-settings/hooks/useFeatureMathState';
import { useMissingIndicatorState } from './node-settings/hooks/useMissingIndicatorState';
import { useDatasetProfiling } from './node-settings/hooks/useDatasetProfiling';
import { NodeSettingsParameterField } from './node-settings/fields/NodeSettingsParameterField';
import { NodeSettingsMultiSelectField } from './node-settings/fields/NodeSettingsMultiSelectField';
import { useImputationStrategyHandlers } from './node-settings/hooks/useImputationStrategyHandlers';
import { useAliasStrategyHandlers } from './node-settings/hooks/useAliasStrategyHandlers';
import { useDateStrategyHandlers } from './node-settings/hooks/useDateStrategyHandlers';
import { useFeatureMathHandlers } from './node-settings/hooks/useFeatureMathHandlers';

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

  const catalogFlags = useCatalogFlags(node);
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
  } = catalogFlags;
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

  const {
    binningColumnsParameter,
    scalingColumnsParameter,
    removeDuplicatesColumnsParameter,
    removeDuplicatesKeepParameter,
    replaceAliasesCustomPairsParameter,
    trimWhitespaceColumnsParameter,
    trimWhitespaceModeParameter,
    removeSpecialColumnsParameter,
    removeSpecialModeParameter,
    removeSpecialReplacementParameter,
    replaceInvalidColumnsParameter,
    replaceInvalidModeParameter,
    replaceInvalidMinValueParameter,
    replaceInvalidMaxValueParameter,
    regexCleanupColumnsParameter,
    regexCleanupModeParameter,
    regexCleanupPatternParameter,
    regexCleanupReplacementParameter,
    normalizeCaseColumnsParameter,
    normalizeCaseModeParameter,
    labelEncodingColumnsParameter,
    labelEncodingAutoDetectParameter,
    labelEncodingMaxUniqueParameter,
    labelEncodingOutputSuffixParameter,
    labelEncodingDropOriginalParameter,
    labelEncodingMissingStrategyParameter,
    labelEncodingMissingCodeParameter,
    hashEncodingColumnsParameter,
    hashEncodingAutoDetectParameter,
    hashEncodingMaxCategoriesParameter,
    hashEncodingBucketsParameter,
    hashEncodingOutputSuffixParameter,
    hashEncodingDropOriginalParameter,
    hashEncodingEncodeMissingParameter,
    polynomialColumnsParameter,
    polynomialAutoDetectParameter,
    polynomialDegreeParameter,
    polynomialIncludeBiasParameter,
    polynomialInteractionOnlyParameter,
    polynomialIncludeInputFeaturesParameter,
    polynomialOutputPrefixParameter,
    featureSelectionColumnsParameter,
    featureSelectionAutoDetectParameter,
    featureSelectionTargetColumnParameter,
    featureSelectionMethodParameter,
    featureSelectionScoreFuncParameter,
    featureSelectionProblemTypeParameter,
    featureSelectionKParameter,
    featureSelectionPercentileParameter,
    featureSelectionAlphaParameter,
    featureSelectionThresholdParameter,
    featureSelectionModeParameter,
    featureSelectionEstimatorParameter,
    featureSelectionStepParameter,
    featureSelectionMinFeaturesParameter,
    featureSelectionMaxFeaturesParameter,
    featureSelectionDropUnselectedParameter,
    featureMathErrorHandlingParameter,
    featureMathAllowOverwriteParameter,
    featureMathDefaultTimezoneParameter,
    featureMathEpsilonParameter,
    resamplingMethodParameter,
    resamplingTargetColumnParameter,
    resamplingSamplingStrategyParameter,
    resamplingRandomStateParameter,
    resamplingKNeighborsParameter,
    resamplingReplacementParameter,
    featureTargetSplitTargetColumnParameter,
    trainModelTargetColumnParameter,
    trainModelProblemTypeParameter,
    trainModelModelTypeParameter,
    trainModelHyperparametersParameter,
    hyperparameterTuningSearchStrategyParameter,
    hyperparameterTuningSearchIterationsParameter,
    hyperparameterTuningSearchRandomStateParameter,
    hyperparameterTuningScoringMetricParameter,
    trainModelCvEnabledParameter,
    trainModelCvStrategyParameter,
    trainModelCvFoldsParameter,
    trainModelCvShuffleParameter,
    trainModelCvRandomStateParameter,
    trainModelCvRefitStrategyParameter,
    targetEncodingColumnsParameter,
    targetEncodingTargetColumnParameter,
    targetEncodingAutoDetectParameter,
    targetEncodingMaxCategoriesParameter,
    targetEncodingOutputSuffixParameter,
    targetEncodingDropOriginalParameter,
    targetEncodingSmoothingParameter,
    targetEncodingEncodeMissingParameter,
    targetEncodingHandleUnknownParameter,
    ordinalEncodingColumnsParameter,
    ordinalEncodingAutoDetectParameter,
    ordinalEncodingMaxCategoriesParameter,
    ordinalEncodingOutputSuffixParameter,
    ordinalEncodingDropOriginalParameter,
    ordinalEncodingEncodeMissingParameter,
    ordinalEncodingHandleUnknownParameter,
    ordinalEncodingUnknownValueParameter,
    dummyEncodingColumnsParameter,
    dummyEncodingAutoDetectParameter,
    dummyEncodingMaxCategoriesParameter,
    dummyEncodingDropFirstParameter,
    dummyEncodingIncludeMissingParameter,
    dummyEncodingDropOriginalParameter,
    dummyEncodingPrefixSeparatorParameter,
    oneHotEncodingColumnsParameter,
    oneHotEncodingAutoDetectParameter,
    oneHotEncodingMaxCategoriesParameter,
    oneHotEncodingDropFirstParameter,
    oneHotEncodingIncludeMissingParameter,
    oneHotEncodingDropOriginalParameter,
    oneHotEncodingPrefixSeparatorParameter,
    missingIndicatorColumnsParameter,
    missingIndicatorSuffixParameter,
    scalingDefaultMethodParameter,
    scalingAutoDetectParameter,
    dropRowsAnyParameter,
  } = useNodeSpecificParameters(getParameter, catalogFlags);
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
    outlierPreviewSignal,
    transformerAuditSignal,
  } = usePreviewSignals({
    previewState,
    nodeId,
    isFeatureMathNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
    isOutlierNode,
    isTransformerAuditNode,
  });

  useFeatureSelectionAutoConfig({
    isFeatureSelectionNode,
    featureSelectionSignal,
    upstreamTargetColumn,
    setConfigState,
  });

  const {
    featureMathOperations,
    featureMathSummaries,
    collapsedFeatureMath,
    setCollapsedFeatureMath,
  } = useFeatureMathState(isFeatureMathNode, configState, featureMathSignals);

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

  const {
    activeFlagSuffix,
    missingIndicatorColumns,
    missingIndicatorInsights,
  } = useMissingIndicatorState(
    isMissingIndicatorNode,
    configState,
    node,
    availableColumns,
    columnMissingMap
  );

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

  const {
    updateAliasStrategies,
    handleAddAliasStrategy,
    handleRemoveAliasStrategy,
    toggleAliasStrategySection,
    handleAliasModeChange,
    handleAliasColumnToggle,
    handleAliasColumnsChange,
    handleAliasAutoDetectToggle,
  } = useAliasStrategyHandlers({
    isReplaceAliasesNode,
    node,
    setConfigState,
    setCollapsedStrategies,
    aliasColumnSummary,
    aliasStrategyCount,
  });

  const {
    updateDateStrategies,
    handleAddDateStrategy,
    handleRemoveDateStrategy,
    toggleDateStrategySection,
    handleDateStrategyModeChange,
    handleDateStrategyColumnsChange,
    handleDateStrategyColumnToggle,
    handleDateStrategyAutoDetectToggle,
  } = useDateStrategyHandlers({
    isStandardizeDatesNode,
    node,
    setConfigState,
    setCollapsedStrategies,
    dateStrategies,
    standardizeDatesColumnSummary,
    standardizeDatesMode,
  });

  const {
    updateFeatureMathOperations,
    handleAddFeatureMathOperation,
    handleDuplicateFeatureMathOperation,
    handleRemoveFeatureMathOperation,
    handleReorderFeatureMathOperation,
    handleToggleFeatureMathOperation,
    handleFeatureMathOperationChange,
  } = useFeatureMathHandlers({
    isFeatureMathNode,
    setConfigState,
    setCollapsedFeatureMath,
  });

  const handleRefreshBinnedDistribution = useCallback(() => {
    if (!sourceId) {
      return;
    }
    refreshBinnedDistribution();
  }, [refreshBinnedDistribution, sourceId]);

  const binnedDistributionCards = useBinnedDistributionCards(
    isBinnedDistributionNode,
    binnedDistributionData
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
