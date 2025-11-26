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
  useBinningHandlers,
} from './useBinningHandlers';
import {
  useOutlierHandlers,
} from './useOutlierHandlers';
import {
  useScalingHandlers,
} from './useScalingHandlers';
import {
  useNumericRangeSummaries,
} from './useNumericRangeSummaries';
import {
  usePruneColumnSelections,
} from './usePruneColumnSelections';
import { ensureArrayOfString } from '../sharedUtils';
import type { BinningColumnRecommendation, BinningExcludedColumn } from '../../../api';
import type { CatalogFlagMap } from './useCatalogFlags';

type UseNumericAnalysisStateProps = {
  catalogFlags: CatalogFlagMap;
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
}: UseNumericAnalysisStateProps) => {
  const { isScalingNode, isOutlierNode, isBinningNode } = catalogFlags;
  
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
    catalogFlags,
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
    catalogFlags,
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
  } = useBinningHandlers({
    setConfigState,
    setBinningCustomEdgeDrafts,
    setBinningCustomLabelDrafts,
    binningConfig,
    binningInsightsRecommendations,
    binningAllNumericColumns,
  });

  const {
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
  } = useOutlierHandlers({
    setConfigState,
    outlierRecommendationRows,
    outlierExcludedColumns,
    outlierConfig,
  });

  const {
    handleScalingDefaultMethodChange,
    handleScalingAutoDetectToggle,
    setScalingColumnMethod,
    handleScalingClearOverrides,
    handleScalingApplyAllRecommendations,
    handleScalingSkipColumn,
    handleScalingUnskipColumn,
    handleScalingOverrideSelect,
  } = useScalingHandlers({
    setConfigState,
    scalingRecommendations,
  });

  const {
    binningExcludedColumns,
    binningColumnPreviewMap,
    manualBoundColumns,
    manualRangeFallbackMap,
  } = useNumericRangeSummaries({
    catalogFlags,
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
    catalogFlags,
    scalingExcludedColumns,
    binningExcludedColumns,
    outlierExcludedColumns,
    setConfigState,
  });

  return {
    scaling: {
      config: scalingConfig,
      state: {
        excludedColumns: scalingExcludedColumns,
        methodDetailMap: scalingMethodDetailMap,
        methodOptions: scalingMethodOptions,
        recommendations: scalingRecommendations,
        recommendationRows: scalingRecommendationRows,
        selectedCount: scalingSelectedCount,
        overrideCount: scalingOverrideCount,
        defaultDetail: scalingDefaultDetail,
        defaultLabel: scalingDefaultLabel,
        autoDetectEnabled: scalingAutoDetectEnabled,
        hasRecommendations: scalingHasRecommendations,
        statusMessage: scalingStatusMessage,
        overrideExampleSummary: scalingOverrideExampleSummary,
      },
      handlers: {
        handleDefaultMethodChange: handleScalingDefaultMethodChange,
        handleAutoDetectToggle: handleScalingAutoDetectToggle,
        setColumnMethod: setScalingColumnMethod,
        handleClearOverrides: handleScalingClearOverrides,
        handleApplyAllRecommendations: handleScalingApplyAllRecommendations,
        handleSkipColumn: handleScalingSkipColumn,
        handleUnskipColumn: handleScalingUnskipColumn,
        handleOverrideSelect: handleScalingOverrideSelect,
      },
    },
    outlier: {
      config: outlierConfig,
      state: {
        excludedColumns: outlierExcludedColumns,
        methodLabelMap: outlierMethodLabelMap,
        methodDetailMap: outlierMethodDetailMap,
        methodOptions: outlierMethodOptions,
        recommendations: outlierRecommendations,
        recommendationRows: outlierRecommendationRows,
        selectedCount: outlierSelectedCount,
        overrideCount: outlierOverrideCount,
        parameterOverrideCount: outlierParameterOverrideCount,
        defaultDetail: outlierDefaultDetail,
        defaultLabel: outlierDefaultLabel,
        autoDetectEnabled: outlierAutoDetectEnabled,
        hasRecommendations: outlierHasRecommendations,
        statusMessage: outlierStatusMessage,
        overrideExampleSummary: outlierOverrideExampleSummary,
        sampleSize: outlierSampleSize,
        relativeGeneratedAt: relativeOutlierGeneratedAt,
        hasOverrides: outlierHasOverrides,
        overrideSummaryDisplay: outlierOverrideSummaryDisplay,
      },
      handlers: {
        handleDefaultMethodChange: handleOutlierDefaultMethodChange,
        handleAutoDetectToggle: handleOutlierAutoDetectToggle,
        setColumnMethod: setOutlierColumnMethod,
        handleClearOverrides: handleOutlierClearOverrides,
        handleApplyAllRecommendations: handleOutlierApplyAllRecommendations,
        handleSkipColumn: handleOutlierSkipColumn,
        handleUnskipColumn: handleOutlierUnskipColumn,
        handleOverrideSelect: handleOutlierOverrideSelect,
        handleMethodParameterChange: handleOutlierMethodParameterChange,
        handleColumnParameterChange: handleOutlierColumnParameterChange,
      },
    },
    binning: {
      config: binningConfig,
      state: {
        selectedCount: binningSelectedCount,
        defaultLabel: binningDefaultLabel,
        overrideColumns: binningOverrideColumns,
        overrideCount: binningOverrideCount,
        overrideSummary: binningOverrideSummary,
        fieldIds: binningFieldIds,
        customEdgeDrafts: binningCustomEdgeDrafts,
        customLabelDrafts: binningCustomLabelDrafts,
        insightsRecommendations: binningInsightsRecommendations,
        recommendedColumnSet: binningRecommendedColumnSet,
        allNumericColumns: binningAllNumericColumns,
        numericColumnsNotSelected: binningNumericColumnsNotSelected,
        canApplyAllNumeric: canApplyAllBinningNumeric,
        excludedColumns: binningExcludedColumns,
        columnPreviewMap: binningColumnPreviewMap,
        manualBoundColumns: manualBoundColumns,
        manualRangeFallbackMap: manualRangeFallbackMap,
        insightsExcludedColumns: binningInsightsExcludedColumns,
      },
      handlers: {
        setCustomEdgeDrafts: setBinningCustomEdgeDrafts,
        setCustomLabelDrafts: setBinningCustomLabelDrafts,
        handleIntegerChange: handleBinningIntegerChange,
        handleBooleanToggle: handleBinningBooleanToggle,
        handleSuffixChange: handleBinningSuffixChange,
        handleLabelFormatChange: handleBinningLabelFormatChange,
        handleMissingStrategyChange: handleBinningMissingStrategyChange,
        handleMissingLabelChange: handleBinningMissingLabelChange,
        handleCustomBinsChange: handleBinningCustomBinsChange,
        handleCustomLabelsChange: handleBinningCustomLabelsChange,
        handleClearCustomColumn: handleBinningClearCustomColumn,
        updateColumnOverride: updateBinningColumnOverride,
        handleOverrideStrategyChange: handleBinningOverrideStrategyChange,
        handleOverrideNumberChange: handleBinningOverrideNumberChange,
        handleOverrideKbinsEncodeChange: handleBinningOverrideKbinsEncodeChange,
        handleOverrideKbinsStrategyChange: handleBinningOverrideKbinsStrategyChange,
        handleClearOverride: handleBinningClearOverride,
        handleClearOverrides: handleBinningClearOverrides,
        handleApplyStrategies: handleBinningApplyStrategies,
        handleApplyColumns: handleBinningApplyColumns,
        handleApplyAllNumeric: handleApplyAllBinningNumeric,
      },
    },
    selectedColumns,
  };
};
