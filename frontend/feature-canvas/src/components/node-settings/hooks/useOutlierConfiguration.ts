import { useMemo } from 'react';
import type {
  OutlierColumnInsight,
  OutlierMethodDetail,
  OutlierMethodName,
  OutlierRecommendationsResponse,
} from '../../../api';
import {
  buildOutlierOverrideSummary,
  buildOutlierRecommendationRows,
  createOutlierMethodDetailMap,
  createOutlierMethodLabelMap,
  createOutlierMethodOptions,
  normalizeOutlierConfigValue,
  OUTLIER_METHOD_FALLBACK_LABELS,
  OUTLIER_METHOD_ORDER,
  type NormalizedOutlierConfig,
  type OutlierMethodOption,
  type OutlierRecommendationRow,
} from '../nodes/outlier/outlierSettings';

export type UseOutlierConfigurationArgs = {
  isOutlierNode: boolean;
  configState: Record<string, any>;
  numericExcludedColumns: Set<string>;
  outlierData: OutlierRecommendationsResponse | null;
};

export type UseOutlierConfigurationResult = {
  outlierConfig: NormalizedOutlierConfig;
  outlierExcludedColumns: Set<string>;
  outlierMethodLabelMap: Record<OutlierMethodName, string>;
  outlierMethodDetailMap: Map<OutlierMethodName, OutlierMethodDetail>;
  outlierMethodOptions: OutlierMethodOption[];
  outlierRecommendations: OutlierColumnInsight[];
  outlierRecommendationRows: OutlierRecommendationRow[];
  outlierSelectedCount: number;
  outlierOverrideCount: number;
  outlierParameterOverrideCount: number;
  outlierDefaultDetail: OutlierMethodDetail | null;
  outlierDefaultLabel: string;
  outlierAutoDetectEnabled: boolean;
  outlierHasRecommendations: boolean;
  outlierStatusMessage: string;
  outlierOverrideExampleSummary: string | null;
};

export const useOutlierConfiguration = ({
  isOutlierNode,
  configState,
  numericExcludedColumns,
  outlierData,
}: UseOutlierConfigurationArgs): UseOutlierConfigurationResult => {
  const outlierConfig = useMemo<NormalizedOutlierConfig>(
    () => normalizeOutlierConfigValue(configState),
    [configState],
  );

  const outlierRecommendations = useMemo(() => {
    if (!Array.isArray(outlierData?.columns)) {
      return [] as OutlierColumnInsight[];
    }
    return outlierData.columns.filter((entry): entry is OutlierColumnInsight => Boolean(entry && entry.column));
  }, [outlierData?.columns]);

  const outlierSkippedColumns = useMemo(() => new Set(outlierConfig.skippedColumns), [outlierConfig.skippedColumns]);

  const outlierSelectedColumns = useMemo(() => {
    if (!isOutlierNode) {
      return new Set<string>();
    }
    return new Set(outlierConfig.columns);
  }, [isOutlierNode, outlierConfig.columns]);

  const outlierExcludedColumns = useMemo(() => {
    if (!isOutlierNode || !numericExcludedColumns.size) {
      return new Set<string>();
    }

    const exemptColumns = new Set<string>();
    outlierConfig.columns.forEach((column) => {
      const normalized = String(column ?? '').trim();
      if (normalized) {
        exemptColumns.add(normalized);
      }
    });
    outlierRecommendations.forEach((insight) => {
      const normalized = String(insight?.column ?? '').trim();
      if (normalized) {
        exemptColumns.add(normalized);
      }
    });

    const filtered = new Set<string>();
    numericExcludedColumns.forEach((column) => {
      if (!exemptColumns.has(column)) {
        filtered.add(column);
      }
    });
    return filtered;
  }, [isOutlierNode, numericExcludedColumns, outlierConfig.columns, outlierRecommendations]);

  const outlierOverrideSet = useMemo(
    () => new Set(Object.keys(outlierConfig.columnMethods)),
    [outlierConfig.columnMethods],
  );

  const outlierMethodLabelMap = useMemo(
    () => createOutlierMethodLabelMap(outlierData?.methods),
    [outlierData?.methods],
  );

  const outlierMethodDetailMap = useMemo(
    () => createOutlierMethodDetailMap(outlierData?.methods),
    [outlierData?.methods],
  );

  const outlierRecommendationRows = useMemo(
    () =>
      buildOutlierRecommendationRows({
        insights: outlierRecommendations,
        outlierConfig,
        labelMap: outlierMethodLabelMap,
        selectedColumns: outlierSelectedColumns,
        skippedColumns: outlierSkippedColumns,
        excludedColumns: outlierExcludedColumns,
      }),
    [
      outlierRecommendations,
      outlierConfig,
      outlierMethodLabelMap,
      outlierSelectedColumns,
      outlierSkippedColumns,
      outlierExcludedColumns,
    ],
  );

  const outlierMethodOptions = useMemo(
    () => createOutlierMethodOptions(outlierMethodLabelMap),
    [outlierMethodLabelMap],
  );

  const outlierOverrideColumns = useMemo(
    () => Array.from(outlierOverrideSet).sort((a, b) => a.localeCompare(b)),
    [outlierOverrideSet],
  );
  const outlierOverridePreview = useMemo(() => outlierOverrideColumns.slice(0, 4), [outlierOverrideColumns]);

  const outlierSelectedCount = outlierSelectedColumns.size;
  const outlierOverrideCount = outlierOverrideSet.size;
  const outlierParameterOverrideCount = useMemo(
    () => Object.keys(outlierConfig.columnParameters).length,
    [outlierConfig.columnParameters],
  );
  const outlierAutoDetectEnabled = outlierConfig.autoDetect;

  const outlierDefaultDetail = useMemo(
    () => outlierMethodDetailMap.get(outlierConfig.defaultMethod) ?? null,
    [outlierConfig.defaultMethod, outlierMethodDetailMap],
  );

  const outlierDefaultLabel = useMemo(() => {
    const method = outlierConfig.defaultMethod;
    return outlierMethodLabelMap[method] ?? OUTLIER_METHOD_FALLBACK_LABELS[method] ?? method;
  }, [outlierConfig.defaultMethod, outlierMethodLabelMap]);

  const outlierHasRecommendations = outlierRecommendationRows.length > 0;

  const outlierAttentionCount = useMemo(
    () =>
      outlierRecommendationRows.filter((row) => {
        if (row.isSkipped || row.isExcluded || !row.isSelected) {
          return false;
        }
        if (!row.recommendedMethod) {
          return false;
        }
        return row.recommendedMethod !== row.currentMethod;
      }).length,
    [outlierRecommendationRows],
  );

  const outlierConfiguredCount = useMemo(
    () =>
      outlierRecommendationRows.filter(
        (row) => row.isSelected && !row.isSkipped && !row.isExcluded,
      ).length,
    [outlierRecommendationRows],
  );

  const outlierSkippedCount = useMemo(
    () => outlierRecommendationRows.filter((row) => row.isSkipped && !row.isExcluded).length,
    [outlierRecommendationRows],
  );

  const outlierStatusMessage = useMemo(() => {
    if (!outlierRecommendationRows.length) {
      return 'No outlier recommendations yet—select numeric columns to evaluate.';
    }
    if (outlierAttentionCount > 0) {
      return `${outlierAttentionCount} column${outlierAttentionCount === 1 ? '' : 's'} need attention. Apply the recommended strategies or adjust manually below.`;
    }
    if (outlierConfiguredCount > 0) {
      return 'All configured columns align with the recommended outlier handling strategies.';
    }
    if (outlierSkippedCount > 0) {
      return 'All outlier recommendations are currently skipped.';
    }
    return 'Review the outlier diagnostics below.';
  }, [outlierAttentionCount, outlierConfiguredCount, outlierRecommendationRows, outlierSkippedCount]);

  const outlierOverrideExampleSummary = useMemo(
    () =>
      buildOutlierOverrideSummary(
        outlierOverridePreview,
        outlierConfig.columnMethods,
        outlierMethodLabelMap,
        outlierOverrideCount,
      ),
    [
      outlierOverridePreview,
      outlierConfig.columnMethods,
      outlierMethodLabelMap,
      outlierOverrideCount,
    ],
  );

  return {
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
  };
};
