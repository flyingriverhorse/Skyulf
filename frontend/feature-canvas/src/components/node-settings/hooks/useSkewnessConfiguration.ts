import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type {
  SkewnessColumnDistribution,
  SkewnessColumnRecommendation,
  SkewnessMethodStatus,
  SkewnessRecommendationsResponse,
} from '../../../api';
import {
  buildSkewnessRows,
  buildSkewnessTableGroups,
  createSkewnessMethodLabelMap,
  isSkewnessTransformationMethod,
  SKEWNESS_DIRECTION_LABEL,
  type SkewnessTableGroup,
  type SkewnessTableRow,
  type SkewnessTransformationConfig,
  type SkewnessTransformationMethod,
  type SkewnessViewMode,
} from '../nodes/skewness/skewnessSettings';

const NUMERIC_TYPE_TOKENS = ['int', 'float', 'double', 'numeric', 'number'];

const isNumericType = (dtype?: string | null): boolean => {
  if (!dtype) {
    return false;
  }
  const normalized = dtype.toLowerCase();
  return NUMERIC_TYPE_TOKENS.some((token) => normalized.includes(token));
};

export type SkewnessDistributionView = 'before' | 'after';

export type SkewnessDistributionCard = {
  column: string;
  skewness: number | null;
  magnitudeLabel: string | null;
  directionLabel: string | null;
  summary: string | null;
  recommendedLabel: string | null;
  appliedLabel: string | null;
  distributionBefore: SkewnessColumnDistribution;
  distributionAfter: SkewnessColumnDistribution | null;
};

export type UseSkewnessConfigurationArgs = {
  skewnessData: SkewnessRecommendationsResponse | null;
  shouldLoadInsights: boolean;
  skewnessTransformations: SkewnessTransformationConfig[];
  availableColumns: string[];
  previewColumns: string[];
  columnTypeMap: Record<string, string>;
  updateSkewnessTransformations: (
    updater: (current: SkewnessTransformationConfig[]) => SkewnessTransformationConfig[],
  ) => void;
};

export type UseSkewnessConfigurationResult = {
  skewnessThreshold: number | null;
  skewnessViewMode: SkewnessViewMode;
  setSkewnessViewMode: Dispatch<SetStateAction<SkewnessViewMode>>;
  skewnessGroupByMethod: boolean;
  setSkewnessGroupByMethod: Dispatch<SetStateAction<boolean>>;
  skewnessDistributionView: SkewnessDistributionView;
  setSkewnessDistributionView: Dispatch<SetStateAction<SkewnessDistributionView>>;
  skewnessRecommendedCount: number;
  skewnessNumericCount: number;
  skewnessTransformationsCount: number;
  hasSkewnessAutoRecommendations: boolean;
  skewnessRows: SkewnessTableRow[];
  skewnessTableGroups: SkewnessTableGroup[];
  skewnessDistributionCards: SkewnessDistributionCard[];
  getSkewnessMethodLabel: (method: SkewnessTransformationMethod) => string;
  getSkewnessMethodStatus: (
    column: string,
    method: SkewnessTransformationMethod,
  ) => SkewnessMethodStatus | { status: 'ready'; reason?: string };
  applySkewnessRecommendations: () => void;
  handleSkewnessOverrideChange: (column: string, nextValue: string) => void;
  clearSkewnessTransformations: () => void;
};

export const useSkewnessConfiguration = ({
  skewnessData,
  shouldLoadInsights,
  skewnessTransformations,
  availableColumns,
  previewColumns,
  columnTypeMap,
  updateSkewnessTransformations,
}: UseSkewnessConfigurationArgs): UseSkewnessConfigurationResult => {
  const [skewnessViewMode, setSkewnessViewMode] = useState<SkewnessViewMode>('recommended');
  const [skewnessGroupByMethod, setSkewnessGroupByMethod] = useState(false);
  const [skewnessDistributionView, setSkewnessDistributionView] =
    useState<SkewnessDistributionView>('before');

  const skewnessTransformationsCount = skewnessTransformations.length;

  const skewnessMethods = useMemo(
    () => (skewnessData?.methods ? [...skewnessData.methods] : []),
    [skewnessData?.methods],
  );

  const skewnessMethodLabelMap = useMemo(
    () => createSkewnessMethodLabelMap(skewnessMethods),
    [skewnessMethods],
  );

  const skewnessRecommendations = useMemo(
    () => (skewnessData?.columns ? [...skewnessData.columns] : []),
    [skewnessData?.columns],
  );

  const skewnessColumnMap = useMemo(() => {
    const map = new Map<string, SkewnessColumnRecommendation>();
    skewnessRecommendations.forEach((item) => {
      if (item && item.column) {
        map.set(item.column, item);
      }
    });
    return map;
  }, [skewnessRecommendations]);

  const skewnessColumnOptions = useMemo(() => {
    const entries = new Set<string>();
    skewnessRecommendations.forEach((item) => {
      if (item?.column) {
        entries.add(item.column);
      }
    });
    // Only include numeric columns from availableColumns and previewColumns
    availableColumns.forEach((column) => {
      if (column && isNumericType(columnTypeMap[column] ?? null)) {
        entries.add(column);
      }
    });
    previewColumns.forEach((column) => {
      const normalized = typeof column === 'string' ? column.trim() : '';
      if (normalized && isNumericType(columnTypeMap[normalized] ?? null)) {
        entries.add(normalized);
      }
    });
    return Array.from(entries).sort((a, b) => a.localeCompare(b));
  }, [availableColumns, previewColumns, skewnessRecommendations, columnTypeMap]);

  useEffect(() => {
    if (
      skewnessViewMode === 'recommended' &&
      skewnessRecommendations.length === 0 &&
      skewnessColumnOptions.length > 0
    ) {
      setSkewnessViewMode('all');
    }
  }, [skewnessColumnOptions.length, skewnessRecommendations.length, skewnessViewMode]);

  const skewnessThreshold = skewnessData?.skewness_threshold ?? null;

  const getSkewnessMethodLabel = useCallback(
    (method: SkewnessTransformationMethod) => skewnessMethodLabelMap[method] ?? method,
    [skewnessMethodLabelMap],
  );

  const getSkewnessMethodStatus = useCallback(
    (column: string, method: SkewnessTransformationMethod) => {
      const recommendation = skewnessColumnMap.get(column);
      if (!recommendation || !recommendation.method_status) {
        return { status: 'ready' as const, reason: undefined };
      }
      return recommendation.method_status[method] ?? { status: 'ready' as const, reason: undefined };
    },
    [skewnessColumnMap],
  );

  const applySkewnessRecommendations = useCallback(() => {
    const recommended = skewnessRecommendations
      .map((entry) => {
        if (!entry || !entry.column) {
          return null;
        }
        const method = Array.isArray(entry.recommended_methods)
          ? entry.recommended_methods.find(isSkewnessTransformationMethod)
          : undefined;
        if (!method) {
          return null;
        }
        return { column: entry.column, method } as SkewnessTransformationConfig;
      })
      .filter((item): item is SkewnessTransformationConfig => Boolean(item));

    if (!recommended.length) {
      return;
    }

    updateSkewnessTransformations((current) => {
      const map = new Map<string, SkewnessTransformationMethod>();

      current.forEach((item) => {
        const normalized = typeof item.column === 'string' ? item.column.trim() : '';
        if (normalized && isSkewnessTransformationMethod(item.method)) {
          map.set(normalized, item.method);
        }
      });

      recommended.forEach((item) => {
        const normalized = typeof item.column === 'string' ? item.column.trim() : '';
        if (normalized) {
          map.set(normalized, item.method);
        }
      });

      return Array.from(map.entries()).map(([column, method]) => ({ column, method }));
    });
  }, [skewnessRecommendations, updateSkewnessTransformations]);

  const skewnessTransformationMap = useMemo(() => {
    const map = new Map<string, SkewnessTransformationMethod>();
    skewnessTransformations.forEach((entry) => {
      const normalized = typeof entry.column === 'string' ? entry.column.trim() : '';
      if (!normalized || map.has(normalized)) {
        return;
      }
      map.set(normalized, entry.method);
    });
    return map;
  }, [skewnessTransformations]);

  const setSkewnessTransformationForColumn = useCallback(
    (column: string, method: SkewnessTransformationMethod | null) => {
      const normalized = typeof column === 'string' ? column.trim() : '';
      if (!normalized) {
        return;
      }
      updateSkewnessTransformations((current) => {
        const filtered = current.filter((entry) => entry.column !== normalized);
        if (!method) {
          return filtered;
        }
        return [...filtered, { column: normalized, method }];
      });
    },
    [updateSkewnessTransformations],
  );

  const handleSkewnessOverrideChange = useCallback(
    (column: string, nextValue: string) => {
      if (!column) {
        return;
      }
      if (!nextValue) {
        setSkewnessTransformationForColumn(column, null);
        return;
      }
      if (!isSkewnessTransformationMethod(nextValue)) {
        return;
      }
      setSkewnessTransformationForColumn(column, nextValue);
    },
    [setSkewnessTransformationForColumn],
  );

  const clearSkewnessTransformations = useCallback(() => {
    updateSkewnessTransformations(() => []);
  }, [updateSkewnessTransformations]);

  const skewnessRecommendedCount = skewnessRecommendations.length;

  const hasSkewnessAutoRecommendations = useMemo(
    () =>
      skewnessRecommendations.some(
        (entry) =>
          entry &&
          typeof entry.column === 'string' &&
          Array.isArray(entry.recommended_methods) &&
          entry.recommended_methods.some(isSkewnessTransformationMethod),
      ),
    [skewnessRecommendations],
  );

  const skewnessNumericCount = skewnessColumnOptions.length;

  const skewnessRows = useMemo(
    () =>
      buildSkewnessRows({
        viewMode: skewnessViewMode,
        recommendations: skewnessRecommendations,
        columnOptions: skewnessColumnOptions,
        recommendationMap: skewnessColumnMap,
        transformationMap: skewnessTransformationMap,
      }),
    [
      skewnessViewMode,
      skewnessRecommendations,
      skewnessColumnOptions,
      skewnessColumnMap,
      skewnessTransformationMap,
    ],
  );

  const skewnessDistributionCards = useMemo<SkewnessDistributionCard[]>(() => {
    if (!shouldLoadInsights) {
      return [];
    }
    return skewnessRecommendations
      .map((entry) => {
        if (!entry || !entry.column) {
          return null;
        }

        const before = entry.distribution_before ?? entry.distribution ?? null;
        if (!before) {
          return null;
        }
        if (
          !Array.isArray(before.bin_edges) ||
          !Array.isArray(before.counts) ||
          before.counts.length === 0 ||
          before.bin_edges.length !== before.counts.length + 1
        ) {
          return null;
        }

        let after: SkewnessColumnDistribution | null = entry.distribution_after ?? null;
        if (after) {
          if (
            !Array.isArray(after.bin_edges) ||
            !Array.isArray(after.counts) ||
            after.counts.length === 0 ||
            after.bin_edges.length !== after.counts.length + 1
          ) {
            after = null;
          }
        }

        const primaryMethod = entry.recommended_methods?.find(isSkewnessTransformationMethod) ?? null;
        const magnitudeLabel = entry.magnitude
          ? `${entry.magnitude.charAt(0).toUpperCase()}${entry.magnitude.slice(1)}`
          : null;
        const directionLabel = entry.direction
          ? SKEWNESS_DIRECTION_LABEL[entry.direction] ?? entry.direction
          : null;
        const backendApplied = entry.applied_method && isSkewnessTransformationMethod(entry.applied_method)
          ? entry.applied_method
          : null;
        const configuredMethod = skewnessTransformationMap.get(entry.column) ?? null;
        const appliedMethod = backendApplied ?? configuredMethod ?? null;
        const appliedLabel = appliedMethod ? getSkewnessMethodLabel(appliedMethod) : null;

        return {
          column: entry.column,
          skewness: typeof entry.skewness === 'number' ? entry.skewness : null,
          magnitudeLabel,
          directionLabel,
          summary: entry.summary ?? null,
          recommendedLabel: appliedLabel ? null : primaryMethod ? getSkewnessMethodLabel(primaryMethod) : null,
          appliedLabel,
          distributionBefore: before,
          distributionAfter: after,
        } as SkewnessDistributionCard;
      })
      .filter((item): item is SkewnessDistributionCard => Boolean(item))
      .sort((a, b) => {
        const aAbs = a.skewness !== null ? Math.abs(a.skewness) : -Infinity;
        const bAbs = b.skewness !== null ? Math.abs(b.skewness) : -Infinity;
        if (aAbs !== bAbs) {
          return bAbs - aAbs;
        }
        return a.column.localeCompare(b.column);
      });
  }, [
    getSkewnessMethodLabel,
    shouldLoadInsights,
    skewnessRecommendations,
    skewnessTransformationMap,
  ]);

  const skewnessTableGroups = useMemo(
    () => buildSkewnessTableGroups(skewnessRows, skewnessGroupByMethod, getSkewnessMethodLabel),
    [skewnessRows, skewnessGroupByMethod, getSkewnessMethodLabel],
  );

  return {
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
    applySkewnessRecommendations,
    handleSkewnessOverrideChange,
    clearSkewnessTransformations,
  };
};
