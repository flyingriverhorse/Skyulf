import { useMemo } from 'react';
import type {
  ScalingColumnRecommendation,
  ScalingMethodDetail,
  ScalingMethodName,
  ScalingRecommendationsResponse,
} from '../../../api';
import {
  normalizeScalingConfigValue,
  SCALING_METHOD_FALLBACK_LABELS,
  buildScalingOverrideSummary,
  buildScalingRecommendationRows,
  createScalingMethodDetailMap,
  createScalingMethodLabelMap,
  createScalingMethodOptions,
  type NormalizedScalingConfig,
  type ScalingMethodOption,
  type ScalingRecommendationRow,
} from '../nodes/scaling/scalingSettings';
import type { CatalogFlagMap } from './useCatalogFlags';

export type UseScalingConfigurationArgs = {
  catalogFlags: CatalogFlagMap;
  configState: Record<string, any>;
  numericExcludedColumns: Set<string>;
  scalingData: ScalingRecommendationsResponse | null;
};

export type UseScalingConfigurationResult = {
  scalingConfig: NormalizedScalingConfig;
  scalingExcludedColumns: Set<string>;
  scalingMethodDetailMap: Map<ScalingMethodName, ScalingMethodDetail>;
  scalingMethodOptions: ScalingMethodOption[];
  scalingRecommendations: ScalingColumnRecommendation[];
  scalingRecommendationRows: ScalingRecommendationRow[];
  scalingSelectedCount: number;
  scalingOverrideCount: number;
  scalingDefaultDetail: ScalingMethodDetail | null;
  scalingDefaultLabel: string;
  scalingAutoDetectEnabled: boolean;
  scalingHasRecommendations: boolean;
  scalingStatusMessage: string;
  scalingOverrideExampleSummary: string | null;
};

export const useScalingConfiguration = ({
  catalogFlags,
  configState,
  numericExcludedColumns,
  scalingData,
}: UseScalingConfigurationArgs): UseScalingConfigurationResult => {
  const { isScalingNode } = catalogFlags;
  const scalingConfig = useMemo<NormalizedScalingConfig>(
    () => {
      return normalizeScalingConfigValue(configState);
    },
    [configState],
  );

  const scalingSkippedColumns = useMemo(
    () => new Set(scalingConfig.skippedColumns),
    [scalingConfig.skippedColumns],
  );

  const scalingExcludedColumns = useMemo(() => {
    if (!isScalingNode) {
      return new Set<string>();
    }
    return numericExcludedColumns;
  }, [isScalingNode, numericExcludedColumns]);

  const scalingSelectedSet = useMemo(() => {
    if (!isScalingNode) {
      return new Set<string>();
    }
    return new Set(scalingConfig.columns);
  }, [isScalingNode, scalingConfig.columns]);

  const scalingOverrideSet = useMemo(
    () => new Set(Object.keys(scalingConfig.columnMethods)),
    [scalingConfig.columnMethods],
  );

  const scalingRecommendations = useMemo(() => {
    if (!Array.isArray(scalingData?.columns)) {
      return [] as ScalingColumnRecommendation[];
    }
    return scalingData.columns.filter(
      (entry): entry is ScalingColumnRecommendation => Boolean(entry && entry.column),
    );
  }, [scalingData?.columns]);

  const scalingMethodLabelMap = useMemo(
    () => createScalingMethodLabelMap(scalingData?.methods),
    [scalingData?.methods],
  );

  const scalingMethodDetailMap = useMemo(
    () => createScalingMethodDetailMap(scalingData?.methods),
    [scalingData?.methods],
  );

  const scalingRecommendationRows = useMemo(
    () =>
      buildScalingRecommendationRows({
        recommendations: scalingRecommendations,
        scalingConfig,
        scalingMethodLabelMap,
        scalingSelectedSet,
        scalingSkippedColumns,
        scalingExcludedColumns,
      }),
    [
      scalingRecommendations,
      scalingConfig,
      scalingMethodLabelMap,
      scalingSelectedSet,
      scalingSkippedColumns,
      scalingExcludedColumns,
    ],
  );

  const scalingOverrideColumns = useMemo(
    () => Array.from(scalingOverrideSet).sort((a, b) => a.localeCompare(b)),
    [scalingOverrideSet],
  );

  const scalingOverridesPreview = useMemo(
    () => scalingOverrideColumns.slice(0, 4),
    [scalingOverrideColumns],
  );

  const scalingRecommendationCount = scalingRecommendationRows.length;
  const scalingOverrideCount = scalingOverrideSet.size;
  const scalingSelectedCount = scalingSelectedSet.size;
  const scalingAutoDetectEnabled = scalingConfig.autoDetect;

  const scalingDefaultDetail = useMemo(
    () => scalingMethodDetailMap.get(scalingConfig.defaultMethod) ?? null,
    [scalingConfig.defaultMethod, scalingMethodDetailMap],
  );

  const scalingDefaultLabel = useMemo(() => {
    const method = scalingConfig.defaultMethod;
    return (
      scalingMethodLabelMap[method] ??
      SCALING_METHOD_FALLBACK_LABELS[method] ??
      method
    );
  }, [scalingConfig.defaultMethod, scalingMethodLabelMap]);

  const scalingHasRecommendations = scalingRecommendationCount > 0;

  const scalingAttentionCount = useMemo(
    () =>
      scalingRecommendationRows.filter(
        (row) => !row.isSkipped && !row.isExcluded && row.isSelected && row.recommendedMethod !== row.currentMethod,
      ).length,
    [scalingRecommendationRows],
  );

  const scalingConfiguredCount = useMemo(
    () =>
      scalingRecommendationRows.filter(
        (row) => row.isSelected && !row.isSkipped && !row.isExcluded,
      ).length,
    [scalingRecommendationRows],
  );

  const scalingSkippedCount = useMemo(
    () =>
      scalingRecommendationRows.filter((row) => row.isSkipped && !row.isExcluded).length,
    [scalingRecommendationRows],
  );

  const scalingStatusMessage = useMemo(() => {
    if (!scalingRecommendationRows.length) {
      return 'No scaling recommendations yet—select numeric columns to evaluate.';
    }
    if (scalingAttentionCount > 0) {
      return `${scalingAttentionCount} column${scalingAttentionCount === 1 ? '' : 's'} need attention. Apply the recommended scalers or override them manually below.`;
    }
    if (scalingConfiguredCount > 0) {
      return 'All configured columns align with the recommended scalers.';
    }
    if (scalingSkippedCount > 0) {
      return 'All recommendations are currently skipped.';
    }
    return 'Review the recommendations below.';
  }, [
    scalingAttentionCount,
    scalingConfiguredCount,
    scalingRecommendationRows,
    scalingSkippedCount,
  ]);

  const scalingMethodOptions = useMemo(
    () => createScalingMethodOptions(scalingMethodLabelMap),
    [scalingMethodLabelMap],
  );

  const scalingOverrideExampleSummary = useMemo(
    () =>
      buildScalingOverrideSummary(
        scalingOverridesPreview,
        scalingConfig.columnMethods,
        scalingMethodLabelMap,
        scalingOverrideCount,
      ),
    [
      scalingOverridesPreview,
      scalingConfig.columnMethods,
      scalingMethodLabelMap,
      scalingOverrideCount,
    ],
  );

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
  };
};
