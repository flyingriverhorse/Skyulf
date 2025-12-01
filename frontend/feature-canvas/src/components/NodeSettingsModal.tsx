import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Info } from 'lucide-react';
import type { Node } from 'react-flow-renderer';
import { DropMissingSettingsSection } from './node-settings/nodes/drop_col_rows/DropMissingSettingsSection';

import { DropMissingRowsSection } from './node-settings/nodes/drop_col_rows/DropMissingRowsSection';
import { RemoveDuplicatesSection } from './node-settings/nodes/remove_duplicates/RemoveDuplicatesSection';

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
} from './node-settings/formatting';
import { BinNumericColumnsSection } from './node-settings/nodes/binning/BinNumericColumnsSection';
import { BinningInsightsSection } from './node-settings/nodes/binning/BinningInsightsSection';
import { BinnedDistributionSection } from './node-settings/nodes/binning/BinnedDistributionSection';

import { ImputationStrategiesSection } from './node-settings/nodes/imputation/ImputationStrategiesSection';
import { ScalingInsightsSection } from './node-settings/nodes/scaling/ScalingInsightsSection';
import { SkewnessInsightsSection } from './node-settings/nodes/skewness/SkewnessInsightsSection';

import { SkewnessDistributionSection } from './node-settings/nodes/skewness/SkewnessDistributionSection';
import { MissingIndicatorInsightsSection } from './node-settings/nodes/missing_indicator/MissingIndicatorInsightsSection';
import { MissingIndicatorSettingsSection } from './node-settings/nodes/missing_indicator/MissingIndicatorSettingsSection';
import { ReplaceAliasesSection } from './node-settings/nodes/replace_aliases/ReplaceAliasesSection';
import { StandardizeDatesSection } from './node-settings/nodes/standardize_date/StandardizeDatesSection';
import { TrimWhitespaceSection } from './node-settings/nodes/trim_white_space/TrimWhitespaceSection';
import { RemoveSpecialCharactersSection } from './node-settings/nodes/remove_special_char/RemoveSpecialCharactersSection';
import { NormalizeTextCaseSection } from './node-settings/nodes/normalize_text/NormalizeTextCaseSection';
import { RegexCleanupSection } from './node-settings/nodes/regex_node/RegexCleanupSection';
import { ReplaceInvalidValuesSection } from './node-settings/nodes/replace_invalid_values/ReplaceInvalidValuesSection';
import { FeatureMathSection } from './node-settings/nodes/feature_math/FeatureMathSection';
import { LabelEncodingSection } from './node-settings/nodes/label_encoding/LabelEncodingSection';
import { TargetEncodingSection } from './node-settings/nodes/target_encoding/TargetEncodingSection';
import { HashEncodingSection } from './node-settings/nodes/hash_encoding/HashEncodingSection';
import { PolynomialFeaturesSection } from './node-settings/nodes/polynomial_features/PolynomialFeaturesSection';
import { FeatureSelectionSection } from './node-settings/nodes/feature_selection/FeatureSelectionSection';
import { OrdinalEncodingSection } from './node-settings/nodes/ordinal_encoding/OrdinalEncodingSection';
import { DummyEncodingSection } from './node-settings/nodes/dummy_encoding/DummyEncodingSection';
import { OneHotEncodingSection } from './node-settings/nodes/one_hot_encoding/OneHotEncodingSection';
import { TransformerAuditSection } from './node-settings/nodes/transformer_audit/TransformerAuditSection';
import { DataConsistencySettingsSection } from './node-settings/nodes/data_consistency/DataConsistencySettingsSection';
import { FeatureTargetSplitSection } from './node-settings/nodes/modeling/FeatureTargetSplitSection';
import { TrainTestSplitSection } from './node-settings/nodes/modeling/TrainTestSplitSection';
import { TrainModelDraftSection } from './node-settings/nodes/modeling/TrainModelDraftSection';
import { ModelTrainingSection } from './node-settings/nodes/modeling/ModelTrainingSection';
import { EvaluationPackSection } from './node-settings/nodes/modeling/EvaluationPackSection';
import { ModelRegistrySection } from './node-settings/nodes/modeling/ModelRegistrySection';
import { HyperparameterTuningSection } from './node-settings/nodes/modeling/HyperparameterTuningSection';
import { ClassResamplingSection } from './node-settings/nodes/resampling/ClassResamplingSection';

import { useBinnedDistributionCards } from './node-settings/hooks';
import { formatCellValue, formatRelativeTime } from './node-settings/utils/formatters';

import { OutlierInsightsSection } from './node-settings/nodes/outlier/OutlierInsightsSection';
import { ConnectionRequirementsSection } from './node-settings/layout/ConnectionRequirementsSection';
import { useDatasetProfiling } from './node-settings/hooks';


import { useNodeSettingsRenderers } from './node-settings/hooks';
import { useNodeSettingsPreview } from './node-settings/hooks';
import { useNodeSettingsInsights } from './node-settings/hooks';
import { useNodeSettingsEffects } from './node-settings/hooks';
import { useNodeSettingsConfiguration } from './node-settings/hooks';
import { useNodeSettingsHandlers } from './node-settings/hooks';
import { useNodeSettingsSummaries } from './node-settings/hooks';
import { useNodeSettingsData } from './node-settings/hooks';
import { useNodeSettingsState } from './node-settings/hooks';
import {
  type BinnedSamplePresetValue,
} from './node-settings/nodes/binning/binningSettings';
import {
  DATE_MODE_OPTIONS,
} from './node-settings/nodes/standardize_date/standardizeDateSettings';

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
  const {
    metadata,
    title,
    connectionReadiness,
    catalogFlags,
    datasetBadge,
    isClassResamplingNode,
    parameters,
    getParameter,
    requiresColumnCatalog,
    dropColumnParameter,
    dataConsistencyHint,
    nodeId,
    resetPermissions,
    nodeParams,
    filteredParameters,
    dataConsistencyParameters,
    shouldLoadSkewnessInsights,
    configStateResult,
    parameterHandlers,
    imputationStrategiesResult,
    skewnessStateResult,
  } = useNodeSettingsState({
    node,
    graphSnapshot: graphSnapshot ?? null,
    isResetAvailable,
    defaultConfigTemplate,
  });

  const {
    connectionInfo,
    connectedHandleKeys,
    connectedOutputHandleKeys,
    evaluationSplitConnectivity,
    connectionReady,
    gatedSectionStyle,
    gatedAriaDisabled,
  } = connectionReadiness;

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

  const { canResetNode, headerCanResetNode, footerCanResetNode } = resetPermissions;

  const {
    configState,
    setConfigState,
    stableInitialConfig,
    stableCurrentConfig,
    nodeChangeVersion,
  } = configStateResult;

  const {
    handleParameterChange,
    handleNumberChange,
    handlePercentileChange,
    handleBooleanChange,
    handleTextChange,
  } = parameterHandlers;

  const modelRegistryConfig = useMemo(() => {
    if (!isModelRegistryNode) {
      return null;
    }
    return configState ? { ...configState } : {};
  }, [configState, isModelRegistryNode]);

  const {
    imputationMethodOptions,
    imputationMethodValues,
    imputerStrategies,
    imputerStrategyCount,
  } = imputationStrategiesResult;

  const { skewnessTransformations, updateSkewnessTransformations } = skewnessStateResult;

  const {
    graphTopology,
    modelingConfig,
    previewSignatureResult,
    cachedPreviewSchema,
    setCachedPreviewSchema,
    schemaDiagnostics,
    columnCatalogState,
  } = useNodeSettingsData({
    graphSnapshot: graphSnapshot ?? null,
    nodeId,
    catalogFlags,
    configState,
    setConfigState,
    nodeParams,
    sourceId,
    requiresColumnCatalog,
    imputerStrategies,
  });

  const {
    graphContext,
    graphNodes,
    graphNodeCount,
    upstreamNodeIds,
    upstreamTargetColumn,
    hasReachableSource,
  } = graphTopology;

  const {
    featureTargetSplitConfig,
    trainTestSplitConfig,
    resamplingConfig,
    trainModelDraftConfig,
    trainModelRuntimeConfig,
    trainModelCVConfig,
    filteredModelTypeOptions,
  } = modelingConfig;

  const { upstreamConfigFingerprints, previewSignature } = previewSignatureResult;

  const {
    cachedSchemaColumns,
    oversamplingSchemaGuard,
    imputationSchemaDiagnostics,
    skipPreview,
  } = schemaDiagnostics;

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
    normalizedColumnSearch,
    filteredColumnOptions,
  } = columnCatalogState;

  const hasDropColumnParameter = Boolean(dropColumnParameter);


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
    nodePreview,
    previewSignals,
    featureMathState,
    previewData,
  } = useNodeSettingsPreview({
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
    upstreamTargetColumn,
    setConfigState,
    configState,
  });

  const {
    previewState,
    refreshPreview,
    clearCachedPreviewSchema,
    canTriggerPreview,
  } = nodePreview;

  const {
    featureMathSignals,
    polynomialSignal,
    featureSelectionSignal,
    outlierPreviewSignal,
    transformerAuditSignal,
  } = previewSignals;

  const {
    featureMathOperations,
    featureMathSummaries,
    collapsedFeatureMath,
    setCollapsedFeatureMath,
  } = featureMathState;

  const { previewColumns, previewColumnStats, previewSampleRows } = previewData;


  const {
    dropMissingColumns,
    encodingRecommendations,
    outlierRecommendations,
    scalingInsights,
    binningInsights,
    binnedDistribution,
    skewnessInsights,
    missingIndicatorState,
    dataCleaningState,
    numericColumnAnalysis,
    numericAnalysisState,
    removeDuplicatesState,
  } = useNodeSettingsInsights({
    catalogFlags,
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
    dropColumnParameter,
    node,
    configState,
    setConfigState,
    nodeChangeVersion,
    binnedSamplePreset,
    skewnessTransformations,
    availableColumns,
    columnMissingMap,
    previewSampleRows,
  });

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
  } = dropMissingColumns;

  const {
    labelEncoding,
    targetEncoding,
    hashEncoding,
    ordinalEncoding,
    dummyEncoding,
    oneHotEncoding,
  } = encodingRecommendations;

  const {
    outlierData,
    outlierError,
    isFetchingOutliers,
    refreshOutliers,
  } = outlierRecommendations;

  const {
    scalingData,
    scalingError,
    isFetchingScaling,
    refreshScaling,
  } = scalingInsights;

  const {
    binningData,
    binningError,
    isFetchingBinning,
    refreshBinning,
  } = binningInsights;

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
  } = binnedDistribution;

  const {
    skewnessData,
    skewnessError,
    isFetchingSkewness,
    refreshSkewness,
  } = skewnessInsights;

  const {
    activeFlagSuffix,
    missingIndicatorColumns,
    missingIndicatorInsights,
  } = missingIndicatorState;

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
  } = dataCleaningState;

  const shouldAnalyzeNumericColumns = isBinningNode || isScalingNode || isOutlierNode;
  const { numericExcludedColumns } = numericColumnAnalysis;
  
  const {
    scaling,
    outlier,
    binning,
    selectedColumns,
  } = numericAnalysisState;

  useNodeSettingsEffects({
    previewState,
    previewSampleRows,
    activeFlagSuffix: activeFlagSuffix ?? '',
    setColumnTypeMap,
    setColumnSuggestions,
    hasReachableSource,
    requiresColumnCatalog,
    nodeColumns,
    selectedColumns,
    setAvailableColumns,
    setColumnMissingMap,
  });





  const thresholdParameterName = nodeParams.dropMissing.threshold?.name ?? null;

  const {
    skewness,
    imputationConfiguration,
    thresholdRecommendations,
  } = useNodeSettingsConfiguration({
    catalogFlags,
    skewnessData,
    skewnessTransformations,
    availableColumns,
    previewColumns,
    columnTypeMap,
    updateSkewnessTransformations,
    imputerStrategies,
    columnMissingMap,
    previewColumnStats,
    nodeColumns: node?.data?.columns,
    imputerMissingFilter,
    suggestedThreshold,
    thresholdParameterName,
    configState,
    handleParameterChange,
  });

  const {
    imputerColumnOptions,
    imputerMissingSliderMax,
    imputerFilteredOptionCount,
    imputerMissingFilterActive,
  } = imputationConfiguration;

  const {
    normalizedSuggestedThreshold,
    thresholdMatchesSuggestion,
    canApplySuggestedThreshold,
    handleApplySuggestedThreshold,
  } = thresholdRecommendations;

  const showDropMissingRowsSection =
    isDropMissingRowsNode && Boolean(nodeParams.dropMissing.threshold || nodeParams.dropRows.any);

  const {
    columnSelectionHandlers,
    nodeSaveHandlers,
    imputationStrategyHandlers,
    aliasStrategyHandlers,
    dateStrategyHandlers,
    featureMathHandlers,
  } = useNodeSettingsHandlers({
    catalogFlags,
    setConfigState,
    binningExcludedColumns: binning.state.excludedColumns,
    scalingExcludedColumns: scaling.state.excludedColumns,
    availableColumns,
    recommendations,
    nodeId,
    onUpdateConfig,
    onClose,
    sourceId: sourceId ?? null,
    graphSnapshot: graphSnapshot ?? null,
    onUpdateNodeData,
    canResetNode,
    defaultConfigTemplate,
    onResetConfig,
    imputationMethodValues,
    imputationMethodOptions,
    imputerStrategyCount,
    setCollapsedStrategies,
    setImputerMissingFilter,
    node,
    aliasColumnSummary: alias.columnSummary,
    aliasStrategyCount: alias.strategyCount,
    dateStrategies: standardizeDates.strategies,
    standardizeDatesColumnSummary: standardizeDates.columnSummary,
    standardizeDatesMode: standardizeDates.mode,
    setCollapsedFeatureMath,
    configState,
  });

  const {
    handleManualBoundChange,
    handleClearManualBound,
    handleToggleColumn,
    handleApplyAllRecommended,
    handleSelectAllColumns,
    handleClearColumns,
    handleRemoveColumn,
  } = columnSelectionHandlers;

  const { handleSave, handleResetNode } = nodeSaveHandlers;

  const selectionCount = selectedColumns.length;

  const showRecommendations = hasDropColumnSource && Boolean(sourceId) && hasReachableSource;
  const showSaveButton = !datasetBadge && !isInspectionNode;
  const canSave = showSaveButton && stableInitialConfig !== stableCurrentConfig;
  const isProfileLoading = profileState.status === 'loading';
  const isPreviewLoading = previewState.status === 'loading';

  const {
    insightSummaries,
    asyncBusyLabel,
  } = useNodeSettingsSummaries({
    recommendationsGeneratedAt,
    scalingData,
    binningData,
    binnedDistributionData,
    outlierData,
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
    relativeGeneratedAt,
    relativeScalingGeneratedAt,
    relativeBinningGeneratedAt,
    relativeBinnedGeneratedAt,
    relativeOutlierGeneratedAt,
    scalingSampleSize,
    binningSampleSize,
    binnedSampleSize,
    outlierSampleSize,
  } = insightSummaries;

  const { hasActiveAsyncWork, busyLabel, footerBusyLabel } = asyncBusyLabel;

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
  } = imputationStrategyHandlers;

  const {
    updateAliasStrategies,
    handleAddAliasStrategy,
    handleRemoveAliasStrategy,
    toggleAliasStrategySection,
    handleAliasModeChange,
    handleAliasColumnToggle,
    handleAliasColumnsChange,
    handleAliasAutoDetectToggle,
  } = aliasStrategyHandlers;

  const {
    updateDateStrategies,
    handleAddDateStrategy,
    handleRemoveDateStrategy,
    toggleDateStrategySection,
    handleDateStrategyModeChange,
    handleDateStrategyColumnsChange,
    handleDateStrategyColumnToggle,
    handleDateStrategyAutoDetectToggle,
  } = dateStrategyHandlers;

  const {
    updateFeatureMathOperations,
    handleAddFeatureMathOperation,
    handleDuplicateFeatureMathOperation,
    handleRemoveFeatureMathOperation,
    handleReorderFeatureMathOperation,
    handleToggleFeatureMathOperation,
    handleFeatureMathOperationChange,
  } = featureMathHandlers;

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





  const { renderMultiSelectField, renderParameterField } = useNodeSettingsRenderers({
    nodeId,
    catalogFlags,
    configState,
    previewState,
    binning,
    scaling,
    selectedColumns,
    availableColumns,
    filteredColumnOptions,
    normalizedColumnSearch,
    columnSearch,
    setColumnSearch,
    columnSuggestions,
    columnMissingMap,
    columnTypeMap,
    sourceId,
    hasReachableSource,
    recommendationsState: {
      isFetching: isFetchingRecommendations,
      availableFilters,
      activeFilterId,
      setActiveFilterId,
      recommendations,
      filteredRecommendations,
      error: recommendationsError,
      relativeGeneratedAt,
      formatSignalName,
      refresh: refreshRecommendations,
      show: showRecommendations,
    },
    columnSelectionHandlers: {
      handleToggleColumn,
      handleRemoveColumn,
      handleApplyAllRecommended,
      handleSelectAllColumns,
      handleClearColumns,
    },
    parameterHandlers: {
      handleNumberChange,
      handleBooleanChange,
      handleTextChange,
    },
    thresholdRecommendations: {
      thresholdParameterName,
      normalizedSuggestedThreshold,
      canApplySuggestedThreshold,
      thresholdMatchesSuggestion,
      handleApplySuggestedThreshold,
    },
    numericExcludedColumns,
    selectionCount,
  });



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
            <DropMissingSettingsSection
              thresholdParameter={nodeParams.dropMissing.threshold ?? null}
              dropColumnParameter={dropColumnParameter}
              renderParameterField={renderParameterField}
              renderMultiSelectField={renderMultiSelectField}
            />
          )}
          {isMissingIndicatorNode && (
            <>
              <MissingIndicatorInsightsSection
                suffix={activeFlagSuffix}
                insights={missingIndicatorInsights}
                formatMissingPercentage={formatMissingPercentage}
              />
              <MissingIndicatorSettingsSection
                columnsParameter={nodeParams.missingIndicator.columns}
                suffixParameter={nodeParams.missingIndicator.suffix}
                renderParameterField={renderParameterField}
              />
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
              nodeId={node?.id}
              onUpdateNodeData={onUpdateNodeData}
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
              <p className="canvas-modal__note" style={{ display: 'flex', alignItems: 'start', gap: '0.5rem' }}>
                <Info size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
                <span>
                  <strong>{dataConsistencyHint.title}.</strong> {dataConsistencyHint.body}
                </span>
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
              removeDuplicatesKeepSelectId={removeDuplicatesState.removeDuplicatesKeepSelectId}
              removeDuplicatesKeep={removeDuplicatesState.removeDuplicatesKeep}
              onKeepChange={removeDuplicatesState.handleRemoveDuplicatesKeepChange}
            />
          )}
          {isDataConsistencyNode && (
            <DataConsistencySettingsSection
              parameters={dataConsistencyParameters}
              renderParameterField={renderParameterField}
            />
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
// End of component
