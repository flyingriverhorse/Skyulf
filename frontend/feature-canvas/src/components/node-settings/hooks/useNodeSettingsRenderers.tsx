import type { Dispatch, SetStateAction } from 'react';
import { useCallback } from 'react';
import { NodeSettingsMultiSelectField } from '../fields/NodeSettingsMultiSelectField';
import { NodeSettingsParameterField } from '../fields/NodeSettingsParameterField';
import type { FeatureNodeParameter } from '../../../api';
import type { CatalogFlagMap } from './core/useCatalogFlags';

type UseNodeSettingsRenderersArgs = {
  nodeId: string;
  catalogFlags: CatalogFlagMap;
  configState: Record<string, any>;
  previewState: any; // PreviewState
  binning: any;
  scaling: any;
  selectedColumns: string[];
  availableColumns: string[];
  filteredColumnOptions: string[];
  normalizedColumnSearch: string;
  columnSearch: string;
  setColumnSearch: (value: string) => void;
  columnSuggestions: any;
  columnMissingMap: any;
  columnTypeMap: any;
  sourceId?: string | null;
  hasReachableSource: boolean;
  recommendationsState: {
    isFetching: boolean;
    availableFilters: any[];
    activeFilterId: string | null;
    setActiveFilterId: Dispatch<SetStateAction<string | null>>;
    recommendations: any[];
    filteredRecommendations: any[];
    error: any;
    relativeGeneratedAt: string | null;
    formatSignalName: (signal?: string | null) => string | null;
    refresh: () => void;
    show: boolean;
  };
  columnSelectionHandlers: {
    handleToggleColumn: any;
    handleRemoveColumn: any;
    handleApplyAllRecommended: any;
    handleSelectAllColumns: any;
    handleClearColumns: any;
  };
  parameterHandlers: {
    handleNumberChange: any;
    handleBooleanChange: any;
    handleTextChange: any;
  };
  thresholdRecommendations: {
    thresholdParameterName: string | null;
    normalizedSuggestedThreshold: number | null;
    canApplySuggestedThreshold: boolean;
    thresholdMatchesSuggestion: boolean;
    handleApplySuggestedThreshold: () => void;
  };
  numericExcludedColumns: any;
  selectionCount: number;
};

export const useNodeSettingsRenderers = ({
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
  recommendationsState,
  columnSelectionHandlers,
  parameterHandlers,
  thresholdRecommendations,
  numericExcludedColumns,
  selectionCount,
}: UseNodeSettingsRenderersArgs) => {
  const { isBinningNode, isScalingNode, isCastNode, isImputerNode } = catalogFlags;

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
          isFetchingRecommendations={recommendationsState.isFetching}
          hasReachableSource={hasReachableSource}
          refreshRecommendations={recommendationsState.refresh}
          availableFilters={recommendationsState.availableFilters}
          activeFilterId={recommendationsState.activeFilterId}
          setActiveFilterId={recommendationsState.setActiveFilterId}
          recommendations={recommendationsState.recommendations}
          filteredRecommendations={recommendationsState.filteredRecommendations}
          recommendationsError={recommendationsState.error}
          relativeGeneratedAt={recommendationsState.relativeGeneratedAt}
          formatSignalName={recommendationsState.formatSignalName}
          handleToggleColumn={columnSelectionHandlers.handleToggleColumn}
          handleRemoveColumn={columnSelectionHandlers.handleRemoveColumn}
          handleApplyAllRecommended={columnSelectionHandlers.handleApplyAllRecommended}
          handleBinningApplyColumns={binning.handlers.handleApplyColumns}
          handleSelectAllColumns={columnSelectionHandlers.handleSelectAllColumns}
          handleClearColumns={columnSelectionHandlers.handleClearColumns}
          columnSearch={columnSearch}
          setColumnSearch={setColumnSearch}
          columnMissingMap={columnMissingMap}
          columnTypeMap={columnTypeMap}
          binningColumnPreviewMap={binning.state.columnPreviewMap}
          isImputerNode={isImputerNode}
          showRecommendations={recommendationsState.show}
        />
      );
    },
    [
      recommendationsState.activeFilterId,
      availableColumns,
      recommendationsState.availableFilters,
      columnMissingMap,
      columnSearch,
      columnSuggestions,
      columnTypeMap,
      binning.state.allNumericColumns,
      numericExcludedColumns,
      filteredColumnOptions,
      recommendationsState.filteredRecommendations,
      recommendationsState.formatSignalName,
      columnSelectionHandlers.handleApplyAllRecommended,
      binning.handlers.handleApplyColumns,
      columnSelectionHandlers.handleClearColumns,
      columnSelectionHandlers.handleRemoveColumn,
      columnSelectionHandlers.handleSelectAllColumns,
      columnSelectionHandlers.handleToggleColumn,
      hasReachableSource,
      isCastNode,
      recommendationsState.isFetching,
      isBinningNode,
      binning.state.recommendedColumnSet,
      binning.state.columnPreviewMap,
      isImputerNode,
      isScalingNode,
      binning.state.excludedColumns,
      previewState.status,
      recommendationsState.recommendations,
      recommendationsState.error,
      recommendationsState.relativeGeneratedAt,
      selectedColumns,
      selectionCount,
      recommendationsState.setActiveFilterId,
      setColumnSearch,
      scaling.state.excludedColumns,
      recommendationsState.show,
      sourceId,
      recommendationsState.refresh,
      normalizedColumnSearch,
    ]
  );

  const renderParameterField = useCallback(
    (parameter: FeatureNodeParameter) => {
      return (
        <NodeSettingsParameterField
          parameter={parameter}
          nodeId={nodeId}
          configState={configState}
          handleNumberChange={parameterHandlers.handleNumberChange}
          handleBooleanChange={parameterHandlers.handleBooleanChange}
          handleTextChange={parameterHandlers.handleTextChange}
          thresholdParameterName={thresholdRecommendations.thresholdParameterName}
          normalizedSuggestedThreshold={thresholdRecommendations.normalizedSuggestedThreshold}
          showRecommendations={recommendationsState.show}
          canApplySuggestedThreshold={thresholdRecommendations.canApplySuggestedThreshold}
          thresholdMatchesSuggestion={thresholdRecommendations.thresholdMatchesSuggestion}
          handleApplySuggestedThreshold={thresholdRecommendations.handleApplySuggestedThreshold}
          renderMultiSelect={renderMultiSelectField}
        />
      );
    },
    [
      thresholdRecommendations.canApplySuggestedThreshold,
      configState,
      thresholdRecommendations.handleApplySuggestedThreshold,
      parameterHandlers.handleBooleanChange,
      parameterHandlers.handleNumberChange,
      parameterHandlers.handleTextChange,
      nodeId,
      thresholdRecommendations.normalizedSuggestedThreshold,
      renderMultiSelectField,
      recommendationsState.show,
      thresholdRecommendations.thresholdMatchesSuggestion,
      thresholdRecommendations.thresholdParameterName,
    ]
  );

  return {
    renderMultiSelectField,
    renderParameterField,
  };
};
