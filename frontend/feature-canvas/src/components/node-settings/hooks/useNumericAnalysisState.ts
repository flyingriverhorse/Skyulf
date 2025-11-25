import { useMemo } from 'react';
import {
  useScalingConfiguration,
} from './useScalingConfiguration';
import {
  useOutlierConfiguration,
} from './useOutlierConfiguration';
import {
  useBinningConfiguration,
} from './useBinningConfiguration';
import {
  useNumericRangeSummaries,
} from './useNumericRangeSummaries';
import {
  usePruneColumnSelections,
} from './usePruneColumnSelections';
import { ensureArrayOfString } from '../sharedUtils';
import type { BinningColumnRecommendation, BinningExcludedColumn } from '../../../api';

type UseNumericAnalysisStateProps = {
  isScalingNode: boolean;
  isOutlierNode: boolean;
  isBinningNode: boolean;
  configState: any;
  setConfigState: (updater: any) => void;
  nodeId: string;
  numericExcludedColumns: Set<string>;
  scalingData: any;
  outlierData: any;
  binningData: any;
  availableColumns: string[];
  previewSampleRows: any[];
};

export const useNumericAnalysisState = ({
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
}: UseNumericAnalysisStateProps) => {
  
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

  return {
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
  };
};
