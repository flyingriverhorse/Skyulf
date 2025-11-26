import { useDropMissingColumns } from './useDropMissingColumns';
import { useDropColumnRecommendations } from './useDropColumnRecommendations';
import { useEncodingRecommendationsState } from './useEncodingRecommendationsState';
import { useOutlierRecommendations } from './useOutlierRecommendations';
import { useScalingInsights } from './useScalingInsights';
import { useBinningInsights } from './useBinningInsights';
import { useBinnedDistribution } from './useBinnedDistribution';
import { useSkewnessInsights } from './useSkewnessInsights';
import { useMissingIndicatorState } from './useMissingIndicatorState';
import { useDataCleaningState } from './useDataCleaningState';
import { useNumericColumnAnalysis } from './useNumericColumnAnalysis';
import { useNumericAnalysisState } from './useNumericAnalysisState';
import type { CatalogFlagMap } from './useCatalogFlags';

type UseNodeSettingsInsightsArgs = {
  catalogFlags: CatalogFlagMap;
  sourceId?: string | null;
  nodeId: string;
  graphContext: any;
  hasReachableSource: boolean;
  columnTypeMap: any;
  setAvailableColumns: any;
  setColumnSearch: any;
  setColumnMissingMap: any;
  setColumnTypeMap: any;
  setColumnSuggestions: any;
  dropColumnParameter: any;
  node: any;
  configState: Record<string, any>;
  setConfigState: any;
  nodeChangeVersion: number;
  binnedSamplePreset: any;
  skewnessTransformations: any;
  availableColumns: string[];
  columnMissingMap: any;
  previewSampleRows: any;
};

export const useNodeSettingsInsights = ({
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
}: UseNodeSettingsInsightsArgs) => {
  const hasDropColumnParameter = Boolean(dropColumnParameter);

  const dropMissingColumns = useDropMissingColumns({
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

  const encodingRecommendations = useEncodingRecommendationsState({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    node,
    configState,
    setConfigState,
    nodeChangeVersion,
  });

  const outlierRecommendations = useOutlierRecommendations({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: nodeId || null,
  });

  const scalingInsights = useScalingInsights({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  const binningInsights = useBinningInsights({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
  });

  const binnedDistribution = useBinnedDistribution({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId: node?.id ?? null,
    samplePreset: binnedSamplePreset,
  });

  const skewnessInsights = useSkewnessInsights({
    catalogFlags,
    sourceId,
    graphContext,
    targetNodeId: node?.id ?? null,
    transformations: skewnessTransformations,
  });

  const missingIndicatorState = useMissingIndicatorState({
    catalogFlags,
    configState,
    node,
    availableColumns,
    columnMissingMap,
  });

  const dataCleaningState = useDataCleaningState({
    catalogFlags,
    configState,
    nodeConfig: node?.data?.config ?? null,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const numericColumnAnalysis = useNumericColumnAnalysis({
    catalogFlags,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const numericAnalysisState = useNumericAnalysisState({
    catalogFlags,
    configState,
    setConfigState,
    nodeId,
    numericExcludedColumns: numericColumnAnalysis.numericExcludedColumns,
    scalingData: scalingInsights.scalingData,
    outlierData: outlierRecommendations.outlierData,
    binningData: binningInsights.binningData,
    availableColumns,
    previewSampleRows,
  });

  return {
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
  };
};
