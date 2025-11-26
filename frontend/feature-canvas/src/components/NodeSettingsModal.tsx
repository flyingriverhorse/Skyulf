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
    isPreviewNode,
    isDatasetProfileNode,
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

  const { canResetNode, headerCanResetNode, footerCanResetNode } = useResetPermissions({
    isResetAvailable,
    defaultConfigTemplate,
    catalogFlags,
  });

  const nodeParams = useNodeSpecificParameters(getParameter, catalogFlags);
  const filteredParameters = useFilteredParameters(parameters, catalogFlags);

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
    catalogFlags,
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
    catalogFlags,
    upstreamTargetColumn,
    nodeId,
    modelTypeOptions: nodeParams.trainModel.modelType?.options ?? null,
  });  const upstreamConfigFingerprints = useMemo(() => {
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
    catalogFlags,
    resamplingTargetColumn: resamplingConfig?.targetColumn ?? null,
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
    catalogFlags,
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
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
    setAvailableColumns,
    setColumnMissingMap,
  });

  const {
    labelEncoding,
    targetEncoding,
    hashEncoding,
    ordinalEncoding,
    dummyEncoding,
    oneHotEncoding,
  } = useEncodingRecommendationsState({
    catalogFlags,
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
    datasetProfileController,
    profileState,
  } = useDatasetProfiling({
    node,
    graphContext,
    hasReachableSource,
    sourceId,
    formatRelativeTime,
    catalogFlags,
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
    catalogFlags,
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
    catalogFlags,
  });

  useFeatureSelectionAutoConfig({
    catalogFlags,
    featureSelectionSignal,
    upstreamTargetColumn,
    setConfigState,
  });

  const {
    featureMathOperations,
    featureMathSummaries,
    collapsedFeatureMath,
    setCollapsedFeatureMath,
  } = useFeatureMathState(catalogFlags, configState, featureMathSignals);

  const {
    outlierData,
    outlierError,
    isFetchingOutliers,
    refreshOutliers,
  } = useOutlierRecommendations({
    catalogFlags,
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
    catalogFlags,
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
    catalogFlags,
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
    catalogFlags,
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
    catalogFlags,
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
  } = useMissingIndicatorState({
    catalogFlags,
    configState,
    node,
    availableColumns,
    columnMissingMap,
  });

  const canRefreshSkewnessDistributions = Boolean(sourceId) && hasReachableSource;

  const nodeColumns = useMemo(() => ensureArrayOfString(node?.data?.columns), [node?.data?.columns]);

  const {
    alias,
    trimWhitespace,
    removeSpecial,
    regexCleanup,
    normalizeCase,
    replaceInvalid,
    standardizeDates,
  } = useDataCleaningState({
    catalogFlags,
    configState,
    nodeConfig: node?.data?.config ?? null,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const shouldAnalyzeNumericColumns = isBinningNode || isScalingNode || isOutlierNode;
  const { numericExcludedColumns } = useNumericColumnAnalysis({
    catalogFlags,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });
  
  const {
    scaling,
    outlier,
    binning,
    selectedColumns,
  } = useNumericAnalysisState({
    catalogFlags,
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

  const skewness = useSkewnessConfiguration({
    catalogFlags,
    skewnessData,
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
    catalogFlags,
    imputerStrategies,
    availableColumns,
    columnMissingMap,
    previewColumns,
    previewColumnStats,
    nodeColumns: node?.data?.columns,
    imputerMissingFilter,
  });

  const thresholdParameterName = nodeParams.dropMissing.threshold?.name ?? null;

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
    isDropMissingRowsNode && Boolean(nodeParams.dropMissing.threshold || nodeParams.dropRows.any);

  const {
    handleManualBoundChange,
    handleClearManualBound,
    handleToggleColumn,
    handleApplyAllRecommended,
    handleSelectAllColumns,
    handleClearColumns,
    handleRemoveColumn,
  } = useColumnSelectionHandlers({
    catalogFlags,
    setConfigState,
    binningExcludedColumns: binning.state.excludedColumns,
    scalingExcludedColumns: scaling.state.excludedColumns,
    availableColumns,
    recommendations,
  });

  const { handleSave, handleResetNode } = useNodeSaveHandlers({
    configState,
    catalogFlags,
    nodeId,
    onUpdateConfig,
    onClose,
    sourceId,
    graphSnapshot: graphSnapshot ?? null,
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
    relativeOutlierGeneratedAt,
    scalingSampleSize,
    binningSampleSize,
    binnedSampleSize,
    outlierSampleSize,
  } = useInsightSummaries({
    recommendationsGeneratedAt,
    scalingData,
    binningData,
    binnedDistributionData,
    outlierData,
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
    isFetchingHashEncoding: hashEncoding.isFetching,
    isFetchingBinnedDistribution,
    isFetchingRecommendations,
    catalogFlags,
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
    catalogFlags,
    node,
    setConfigState,
    setCollapsedStrategies,
    aliasColumnSummary: alias.columnSummary,
    aliasStrategyCount: alias.strategyCount,
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
    catalogFlags,
    node,
    setConfigState,
    setCollapsedStrategies,
    dateStrategies: standardizeDates.strategies,
    standardizeDatesColumnSummary: standardizeDates.columnSummary,
    standardizeDatesMode: standardizeDates.mode,
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
    catalogFlags,
    setConfigState,
    setCollapsedFeatureMath,
  });

  const handleRefreshBinnedDistribution = useCallback(() => {
    if (!sourceId) {
      return;
    }
    refreshBinnedDistribution();
  }, [refreshBinnedDistribution, sourceId]);

  const binnedDistributionCards = useBinnedDistributionCards({
    catalogFlags,
    binnedDistributionData,
  });





  const renderMultiSelectField = useCallback(
    (parameter: FeatureNodeParameter) => {
      return (
        <NodeSettingsMultiSelectField
          parameter={parameter}
          previewStateStatus={previewState.status}
          isBinningNode={isBinningNode}
          isScalingNode={isScalingNode}
          binningAllNumericColumns={binning.state.allNumericColumns}
          binningRecommendedColumnSet={binning.state.recommendedColumnSet}
          selectedColumns={selectedColumns}
          availableColumns={availableColumns}
          scalingExcludedColumns={scaling.state.excludedColumns}
          normalizedColumnSearch={normalizedColumnSearch}
          filteredColumnOptions={filteredColumnOptions}
          binningExcludedColumns={binning.state.excludedColumns}
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
          handleBinningApplyColumns={binning.handlers.handleApplyColumns}
          handleSelectAllColumns={handleSelectAllColumns}
          handleClearColumns={handleClearColumns}
          columnSearch={columnSearch}
          setColumnSearch={setColumnSearch}
          columnMissingMap={columnMissingMap}
          columnTypeMap={columnTypeMap}
          binningColumnPreviewMap={binning.state.columnPreviewMap}
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
      binning.state.allNumericColumns,
      numericExcludedColumns,
      filteredColumnOptions,
      filteredRecommendations,
      formatSignalName,
      handleApplyAllRecommended,
      binning.handlers.handleApplyColumns,
      handleClearColumns,
      handleRemoveColumn,
      handleSelectAllColumns,
      handleToggleColumn,
      hasReachableSource,
      isCastNode,
      isFetchingRecommendations,
      isBinningNode,
      binning.state.recommendedColumnSet,
      binning.state.columnPreviewMap,
      isImputerNode,
      isScalingNode,
      binning.state.excludedColumns,
      previewState.status,
      recommendations,
      recommendationsError,
      relativeGeneratedAt,
      selectedColumns,
      selectionCount,
      setActiveFilterId,
      setColumnSearch,
      scaling.state.excludedColumns,
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
              thresholdParameter={nodeParams.dropMissing.threshold ?? null}
              dropIfAnyParameter={nodeParams.dropRows.any}
              renderParameterField={renderParameterField}
            />
          )}
          {dropColumnParameter && (
            <section className="canvas-modal__section">
              <div className="canvas-modal__section-header">
                <h3>Missingness recommendations</h3>
              </div>
              {nodeParams.dropMissing.threshold && renderParameterField(nodeParams.dropMissing.threshold)}
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
              {(nodeParams.missingIndicator.columns || nodeParams.missingIndicator.suffix) && (
                <section className="canvas-modal__section">
                  <div className="canvas-modal__section-header">
                    <h3>Missing indicator settings</h3>
                  </div>
                  <div className="canvas-modal__parameter-list">
                    {nodeParams.missingIndicator.columns ? renderParameterField(nodeParams.missingIndicator.columns) : null}
                    {nodeParams.missingIndicator.suffix ? renderParameterField(nodeParams.missingIndicator.suffix) : null}
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
              columnsParameter={nodeParams.trimWhitespace.columns}
              modeParameter={nodeParams.trimWhitespace.mode}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={trimWhitespace.columnSummary}
              modeDetails={trimWhitespace.modeDetails}
              sampleMap={trimWhitespace.sampleMap}
            />
          )}
          {isRemoveSpecialCharsNode && (
            <RemoveSpecialCharactersSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={nodeParams.removeSpecial.columns}
              modeParameter={nodeParams.removeSpecial.mode}
              replacementParameter={nodeParams.removeSpecial.replacement}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={removeSpecial.columnSummary}
              modeDetails={removeSpecial.modeDetails}
              sampleMap={removeSpecial.sampleMap}
            />
          )}
          {isReplaceInvalidValuesNode && (
            <ReplaceInvalidValuesSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={nodeParams.replaceInvalid.columns}
              modeParameter={nodeParams.replaceInvalid.mode}
              minValueParameter={nodeParams.replaceInvalid.minValue}
              maxValueParameter={nodeParams.replaceInvalid.maxValue}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={replaceInvalid.columnSummary}
              modeDetails={replaceInvalid.modeDetails}
              sampleMap={replaceInvalid.sampleMap}
              selectedMode={replaceInvalid.mode}
              minValue={replaceInvalid.minValue}
              maxValue={replaceInvalid.maxValue}
            />
          )}
          {isRegexCleanupNode && (
            <RegexCleanupSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={nodeParams.regexCleanup.columns}
              modeParameter={nodeParams.regexCleanup.mode}
              patternParameter={nodeParams.regexCleanup.pattern}
              replacementParameter={nodeParams.regexCleanup.replacement}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={regexCleanup.columnSummary}
              modeDetails={regexCleanup.modeDetails}
              sampleMap={regexCleanup.sampleMap}
              selectedMode={regexCleanup.selectedMode}
              replacementValue={regexCleanup.replacementValue}
            />
          )}
          {isNormalizeTextCaseNode && (
            <NormalizeTextCaseSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              columnsParameter={nodeParams.normalizeCase.columns}
              modeParameter={nodeParams.normalizeCase.mode}
              renderMultiSelectField={renderMultiSelectField}
              renderParameterField={renderParameterField}
              columnSummary={normalizeCase.columnSummary}
              modeDetails={normalizeCase.modeDetails}
              sampleMap={normalizeCase.sampleMap}
              selectedMode={normalizeCase.selectedMode}
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
              errorHandlingParameter={nodeParams.featureMath.errorHandling}
              allowOverwriteParameter={nodeParams.featureMath.allowOverwrite}
              defaultTimezoneParameter={nodeParams.featureMath.defaultTimezone}
              epsilonParameter={nodeParams.featureMath.epsilon}
              renderParameterField={renderParameterField}
            />
          )}
          {isOrdinalEncodingNode && (
            <OrdinalEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={ordinalEncoding.isFetching}
              error={ordinalEncoding.error}
              suggestions={ordinalEncoding.suggestions}
              metadata={ordinalEncoding.metadata}
              columnsParameter={nodeParams.ordinalEncoding.columns}
              autoDetectParameter={nodeParams.ordinalEncoding.autoDetect}
              maxCategoriesParameter={nodeParams.ordinalEncoding.maxCategories}
              outputSuffixParameter={nodeParams.ordinalEncoding.outputSuffix}
              dropOriginalParameter={nodeParams.ordinalEncoding.dropOriginal}
              encodeMissingParameter={nodeParams.ordinalEncoding.encodeMissing}
              handleUnknownParameter={nodeParams.ordinalEncoding.handleUnknown}
              unknownValueParameter={nodeParams.ordinalEncoding.unknownValue}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={ordinalEncoding.handleApplyRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isTargetEncodingNode && (
            <TargetEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={targetEncoding.isFetching}
              error={targetEncoding.error}
              suggestions={targetEncoding.suggestions}
              metadata={targetEncoding.metadata}
              columnsParameter={nodeParams.targetEncoding.columns}
              targetColumnParameter={nodeParams.targetEncoding.targetColumn}
              autoDetectParameter={nodeParams.targetEncoding.autoDetect}
              maxCategoriesParameter={nodeParams.targetEncoding.maxCategories}
              outputSuffixParameter={nodeParams.targetEncoding.outputSuffix}
              dropOriginalParameter={nodeParams.targetEncoding.dropOriginal}
              smoothingParameter={nodeParams.targetEncoding.smoothing}
              encodeMissingParameter={nodeParams.targetEncoding.encodeMissing}
              handleUnknownParameter={nodeParams.targetEncoding.handleUnknown}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={targetEncoding.handleApplyRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isDummyEncodingNode && (
            <DummyEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={dummyEncoding.isFetching}
              error={dummyEncoding.error}
              suggestions={dummyEncoding.suggestions}
              metadata={dummyEncoding.metadata}
              columnsParameter={nodeParams.dummyEncoding.columns}
              autoDetectParameter={nodeParams.dummyEncoding.autoDetect}
              maxCategoriesParameter={nodeParams.dummyEncoding.maxCategories}
              dropFirstParameter={nodeParams.dummyEncoding.dropFirst}
              includeMissingParameter={nodeParams.dummyEncoding.includeMissing}
              dropOriginalParameter={nodeParams.dummyEncoding.dropOriginal}
              prefixSeparatorParameter={nodeParams.dummyEncoding.prefixSeparator}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={dummyEncoding.handleApplyRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isOneHotEncodingNode && (
            <OneHotEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={oneHotEncoding.isFetching}
              error={oneHotEncoding.error}
              suggestions={oneHotEncoding.suggestions}
              metadata={oneHotEncoding.metadata}
              columnsParameter={nodeParams.oneHotEncoding.columns}
              autoDetectParameter={nodeParams.oneHotEncoding.autoDetect}
              maxCategoriesParameter={nodeParams.oneHotEncoding.maxCategories}
              dropFirstParameter={nodeParams.oneHotEncoding.dropFirst}
              includeMissingParameter={nodeParams.oneHotEncoding.includeMissing}
              dropOriginalParameter={nodeParams.oneHotEncoding.dropOriginal}
              prefixSeparatorParameter={nodeParams.oneHotEncoding.prefixSeparator}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={oneHotEncoding.handleApplyRecommended}
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
              targetColumnParameter={nodeParams.featureTargetSplit.targetColumn}
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
              methodParameter={nodeParams.resampling.method}
              targetColumnParameter={nodeParams.resampling.targetColumn}
              samplingStrategyParameter={nodeParams.resampling.samplingStrategy}
              randomStateParameter={nodeParams.resampling.randomState}
              kNeighborsParameter={isClassOversamplingNode ? nodeParams.resampling.kNeighbors : null}
              replacementParameter={isClassUndersamplingNode ? nodeParams.resampling.replacement : null}
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
              targetColumnParameter={nodeParams.trainModel.targetColumn}
              problemTypeParameter={nodeParams.trainModel.problemType}
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
                modelTypeParameter={nodeParams.trainModel.modelType}
                modelTypeOptions={filteredModelTypeOptions}
                hyperparametersParameter={nodeParams.trainModel.hyperparameters}
                cvEnabledParameter={nodeParams.trainModel.cvEnabled}
                cvStrategyParameter={nodeParams.trainModel.cvStrategy}
                cvFoldsParameter={nodeParams.trainModel.cvFolds}
                cvShuffleParameter={nodeParams.trainModel.cvShuffle}
                cvRandomStateParameter={nodeParams.trainModel.cvRandomState}
                cvRefitStrategyParameter={nodeParams.trainModel.cvRefitStrategy}
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
                modelTypeParameter={nodeParams.trainModel.modelType}
                modelTypeOptions={filteredModelTypeOptions}
                searchStrategyParameter={nodeParams.hyperparameterTuning.searchStrategy}
                searchIterationsParameter={nodeParams.hyperparameterTuning.searchIterations}
                searchRandomStateParameter={nodeParams.hyperparameterTuning.searchRandomState}
                scoringMetricParameter={nodeParams.hyperparameterTuning.scoringMetric}
                setDraftConfigState={setConfigState}
                onSaveDraftConfig={handleSave}
              />
            </div>
          )}
          {isLabelEncodingNode && (
            <LabelEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={labelEncoding.isFetching}
              error={labelEncoding.error}
              suggestions={labelEncoding.suggestions}
              metadata={labelEncoding.metadata}
              columnsParameter={nodeParams.labelEncoding.columns}
              autoDetectParameter={nodeParams.labelEncoding.autoDetect}
              maxUniqueParameter={nodeParams.labelEncoding.maxUnique}
              outputSuffixParameter={nodeParams.labelEncoding.outputSuffix}
              dropOriginalParameter={nodeParams.labelEncoding.dropOriginal}
              missingStrategyParameter={nodeParams.labelEncoding.missingStrategy}
              missingCodeParameter={nodeParams.labelEncoding.missingCode}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={labelEncoding.handleApplyRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isHashEncodingNode && (
            <HashEncodingSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetching={hashEncoding.isFetching}
              error={hashEncoding.error}
              suggestions={hashEncoding.suggestions}
              metadata={hashEncoding.metadata}
              columnsParameter={nodeParams.hashEncoding.columns}
              autoDetectParameter={nodeParams.hashEncoding.autoDetect}
              maxCategoriesParameter={nodeParams.hashEncoding.maxCategories}
              bucketsParameter={nodeParams.hashEncoding.buckets}
              outputSuffixParameter={nodeParams.hashEncoding.outputSuffix}
              dropOriginalParameter={nodeParams.hashEncoding.dropOriginal}
              encodeMissingParameter={nodeParams.hashEncoding.encodeMissing}
              selectedColumns={selectedColumns}
              renderParameterField={renderParameterField}
              onToggleColumn={handleToggleColumn}
              onApplyRecommended={hashEncoding.handleApplyRecommended}
              formatMissingPercentage={formatMissingPercentage}
            />
          )}
          {isPolynomialFeaturesNode && (
            <PolynomialFeaturesSection
              columnsParameter={nodeParams.polynomial.columns}
              autoDetectParameter={nodeParams.polynomial.autoDetect}
              degreeParameter={nodeParams.polynomial.degree}
              includeBiasParameter={nodeParams.polynomial.includeBias}
              interactionOnlyParameter={nodeParams.polynomial.interactionOnly}
              includeInputFeaturesParameter={nodeParams.polynomial.includeInputFeatures}
              outputPrefixParameter={nodeParams.polynomial.outputPrefix}
              renderParameterField={renderParameterField}
              signal={polynomialSignal}
            />
          )}
          {isFeatureSelectionNode && (
            <FeatureSelectionSection
              columnsParameter={nodeParams.featureSelection.columns}
              autoDetectParameter={nodeParams.featureSelection.autoDetect}
              targetColumnParameter={nodeParams.featureSelection.targetColumn}
              methodParameter={nodeParams.featureSelection.method}
              scoreFuncParameter={nodeParams.featureSelection.scoreFunc}
              problemTypeParameter={nodeParams.featureSelection.problemType}
              kParameter={nodeParams.featureSelection.k}
              percentileParameter={nodeParams.featureSelection.percentile}
              alphaParameter={nodeParams.featureSelection.alpha}
              thresholdParameter={nodeParams.featureSelection.threshold}
              modeParameter={nodeParams.featureSelection.mode}
              estimatorParameter={nodeParams.featureSelection.estimator}
              stepParameter={nodeParams.featureSelection.step}
              minFeaturesParameter={nodeParams.featureSelection.minFeatures}
              maxFeaturesParameter={nodeParams.featureSelection.maxFeatures}
              dropUnselectedParameter={nodeParams.featureSelection.dropUnselected}
              renderParameterField={renderParameterField}
              signal={featureSelectionSignal}
            />
          )}
          {isStandardizeDatesNode && (
            <StandardizeDatesSection
              hasSource={Boolean(sourceId)}
              hasReachableSource={hasReachableSource}
              strategies={standardizeDates.strategies}
              columnSummary={standardizeDates.columnSummary}
              columnOptions={standardizeDates.columnOptions}
              sampleMap={standardizeDates.sampleMap}
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
              strategies={alias.strategies}
              columnSummary={alias.columnSummary}
              columnOptions={alias.columnOptions}
              sampleMap={alias.sampleMap}
              customPairSummary={alias.customPairSummary}
              customPairsParameter={nodeParams.replaceAliases.customPairs}
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
              outlierConfig={outlier.config}
              outlierMethodOptions={outlier.state.methodOptions}
              outlierMethodDetailMap={outlier.state.methodDetailMap}
              outlierDefaultDetail={outlier.state.defaultDetail}
              outlierAutoDetectEnabled={outlier.state.autoDetectEnabled}
              outlierSelectedCount={outlier.state.selectedCount}
              outlierDefaultLabel={outlier.state.defaultLabel}
              outlierOverrideCount={outlier.state.overrideCount}
              outlierParameterOverrideCount={outlier.state.parameterOverrideCount}
              outlierOverrideExampleSummary={outlier.state.overrideExampleSummary}
              outlierHasOverrides={outlier.state.hasOverrides}
              outlierRecommendationRows={outlier.state.recommendationRows}
              outlierHasRecommendations={outlier.state.hasRecommendations}
              outlierStatusMessage={outlier.state.statusMessage}
              outlierError={outlierError}
              outlierSampleSize={outlierSampleSize}
              relativeOutlierGeneratedAt={relativeOutlierGeneratedAt}
              outlierPreviewSignal={outlierPreviewSignal}
              formatMetricValue={formatMetricValue}
              formatNumericStat={formatNumericStat}
              onApplyAllRecommendations={outlier.handlers.handleApplyAllRecommendations}
              onClearOverrides={outlier.handlers.handleClearOverrides}
              onDefaultMethodChange={outlier.handlers.handleDefaultMethodChange}
              onAutoDetectToggle={outlier.handlers.handleAutoDetectToggle}
              onOverrideSelect={outlier.handlers.handleOverrideSelect}
              onMethodParameterChange={outlier.handlers.handleMethodParameterChange}
              onColumnParameterChange={outlier.handlers.handleColumnParameterChange}
            />
          )}
          {isScalingNode && (
            <ScalingInsightsSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetchingScaling={isFetchingScaling}
              scalingConfig={scaling.config}
              scalingMethodOptions={scaling.state.methodOptions}
              scalingMethodDetailMap={scaling.state.methodDetailMap}
              scalingDefaultDetail={scaling.state.defaultDetail}
              scalingAutoDetectEnabled={scaling.state.autoDetectEnabled}
              scalingSelectedCount={scaling.state.selectedCount}
              scalingDefaultLabel={scaling.state.defaultLabel}
              scalingOverrideCount={scaling.state.overrideCount}
              scalingOverrideExampleSummary={scaling.state.overrideExampleSummary}
              scalingRecommendationRows={scaling.state.recommendationRows}
              scalingHasRecommendations={scaling.state.hasRecommendations}
              scalingStatusMessage={scaling.state.statusMessage}
              scalingError={scalingError}
              scalingSampleSize={scalingSampleSize}
              relativeScalingGeneratedAt={relativeScalingGeneratedAt}
              formatMetricValue={formatMetricValue}
              formatNumericStat={formatNumericStat}
              formatMissingPercentage={formatMissingPercentage}
              onApplyAllRecommendations={scaling.handlers.handleApplyAllRecommendations}
              onClearOverrides={scaling.handlers.handleClearOverrides}
              onDefaultMethodChange={scaling.handlers.handleDefaultMethodChange}
              onAutoDetectToggle={scaling.handlers.handleAutoDetectToggle}
              onOverrideSelect={scaling.handlers.handleOverrideSelect}
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
                binningConfig={binning.config}
                binningDefaultLabel={binning.state.defaultLabel}
                binningOverrideCount={binning.state.overrideCount}
                binningOverrideSummary={binning.state.overrideSummary}
                recommendations={binning.state.insightsRecommendations}
                excludedColumns={binning.state.insightsExcludedColumns}
                numericColumnCount={binning.state.allNumericColumns.length}
                canApplyAllNumeric={binning.state.canApplyAllNumeric}
                onApplyAllNumeric={binning.handlers.handleApplyAllNumeric}
                onApplyStrategies={binning.handlers.handleApplyStrategies}
                onClearOverrides={binning.handlers.handleClearOverrides}
                customEdgeDrafts={binning.state.customEdgeDrafts}
                customLabelDrafts={binning.state.customLabelDrafts}
                onOverrideStrategyChange={binning.handlers.handleOverrideStrategyChange}
                onOverrideNumberChange={binning.handlers.handleOverrideNumberChange}
                onOverrideKbinsEncodeChange={binning.handlers.handleOverrideKbinsEncodeChange}
                onOverrideKbinsStrategyChange={binning.handlers.handleOverrideKbinsStrategyChange}
                onOverrideClear={binning.handlers.handleClearOverride}
                onCustomBinsChange={binning.handlers.handleCustomBinsChange}
                onCustomLabelsChange={binning.handlers.handleCustomLabelsChange}
                onClearCustomColumn={binning.handlers.handleClearCustomColumn}
                formatMetricValue={formatMetricValue}
                formatNumericStat={formatNumericStat}
              />
              <BinNumericColumnsSection
                config={binning.config}
                fieldIds={binning.state.fieldIds}
                onIntegerChange={binning.handlers.handleIntegerChange}
                onBooleanToggle={binning.handlers.handleBooleanToggle}
                onSuffixChange={binning.handlers.handleSuffixChange}
                onLabelFormatChange={binning.handlers.handleLabelFormatChange}
                onMissingStrategyChange={binning.handlers.handleMissingStrategyChange}
                onMissingLabelChange={binning.handlers.handleMissingLabelChange}
              />
            </>
          )}
          {isSkewnessNode && (
            <SkewnessInsightsSection
              sourceId={sourceId}
              hasReachableSource={hasReachableSource}
              isFetchingSkewness={isFetchingSkewness}
              skewnessThreshold={skewness.state.threshold}
              skewnessError={skewnessError}
              skewnessViewMode={skewness.state.viewMode}
              skewnessRecommendedCount={skewness.state.recommendedCount}
              skewnessNumericCount={skewness.state.numericCount}
              skewnessGroupByMethod={skewness.state.groupByMethod}
              skewnessRows={skewness.state.rows}
              skewnessTableGroups={skewness.state.tableGroups}
              hasSkewnessAutoRecommendations={skewness.state.hasAutoRecommendations}
              skewnessTransformationsCount={skewness.state.transformationsCount}
              isFetchingRecommendations={isFetchingRecommendations}
              getSkewnessMethodLabel={skewness.handlers.getMethodLabel}
              getSkewnessMethodStatus={skewness.handlers.getMethodStatus}
              onApplyRecommendations={skewness.handlers.applyRecommendations}
              onViewModeChange={skewness.handlers.setViewMode}
              onGroupByToggle={skewness.handlers.setGroupByMethod}
              onOverrideChange={skewness.handlers.handleOverrideChange}
              onClearSelections={skewness.handlers.clearTransformations}
            />
          )}
          {isSkewnessDistributionNode && (
            <SkewnessDistributionSection
              skewnessThreshold={skewness.state.threshold}
              isFetchingSkewness={isFetchingSkewness}
              skewnessError={skewnessError}
              skewnessDistributionCards={skewness.state.distributionCards}
              skewnessDistributionView={skewness.state.distributionView}
              setSkewnessDistributionView={skewness.handlers.setDistributionView}
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
              removeDuplicatesColumnsParameter={nodeParams.removeDuplicates.columns}
              removeDuplicatesKeepParameter={nodeParams.removeDuplicates.keep}
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
