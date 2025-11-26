import { useCallback, type Dispatch, type SetStateAction } from 'react';
import { ensureArrayOfString } from '../sharedUtils';
import { SCALING_METHOD_ORDER } from '../nodes/scaling/scalingSettings';
import type { CatalogFlagMap } from './useCatalogFlags';

export type ColumnRecommendation = {
  name?: string | null;
};

export type ColumnSelectionState = Record<string, any>;

export type UseColumnSelectionHandlersArgs = {
  catalogFlags: CatalogFlagMap;
  setConfigState: Dispatch<SetStateAction<ColumnSelectionState>>;
  binningExcludedColumns: Set<string>;
  scalingExcludedColumns: Set<string>;
  availableColumns: string[];
  recommendations: ColumnRecommendation[];
};

const sanitizeColumnName = (column: string) => String(column ?? '').trim();

export const useColumnSelectionHandlers = ({
  catalogFlags,
  setConfigState,
  binningExcludedColumns,
  scalingExcludedColumns,
  availableColumns,
  recommendations,
}: UseColumnSelectionHandlersArgs) => {
  const { isBinningNode, isScalingNode } = catalogFlags;

  const handleManualBoundChange = useCallback(
    (column: string, bound: 'lower' | 'upper', rawValue: string) => {
      const normalizedColumn = sanitizeColumnName(column);
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous) => {
        const source =
          previous.manual_bounds &&
          typeof previous.manual_bounds === 'object' &&
          !Array.isArray(previous.manual_bounds)
            ? { ...previous.manual_bounds }
            : {};

        const existing = source[normalizedColumn] && typeof source[normalizedColumn] === 'object'
          ? { ...source[normalizedColumn] }
          : { lower: null, upper: null };

        if (rawValue === '') {
          if (bound === 'lower') {
            existing.lower = null;
          } else {
            existing.upper = null;
          }
        } else {
          const numeric = Number(rawValue);
          if (!Number.isFinite(numeric)) {
            return previous;
          }
          if (bound === 'lower') {
            existing.lower = numeric;
          } else {
            existing.upper = numeric;
          }
        }

        const normalizedLower = existing.lower ?? null;
        const normalizedUpper = existing.upper ?? null;

        if (normalizedLower === null && normalizedUpper === null) {
          delete source[normalizedColumn];
        } else {
          source[normalizedColumn] = {
            lower: normalizedLower,
            upper: normalizedUpper,
          };
        }

        return {
          ...previous,
          manual_bounds: source,
        };
      });
    },
    [setConfigState]
  );

  const handleClearManualBound = useCallback(
    (column: string) => {
      const normalizedColumn = sanitizeColumnName(column);
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous) => {
        const source =
          previous.manual_bounds &&
          typeof previous.manual_bounds === 'object' &&
          !Array.isArray(previous.manual_bounds)
            ? previous.manual_bounds
            : null;
        if (!source || !Object.prototype.hasOwnProperty.call(source, normalizedColumn)) {
          return previous;
        }
        const nextBounds = { ...source };
        delete nextBounds[normalizedColumn];
        return {
          ...previous,
          manual_bounds: nextBounds,
        };
      });
    },
    [setConfigState]
  );

  const handleToggleColumn = useCallback(
    (column: string) => {
      const normalizedColumn = sanitizeColumnName(column);
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous) => {
        const current = ensureArrayOfString(previous.columns);
        const exists = current.includes(normalizedColumn);
        const nextColumns = exists
          ? current.filter((item) => item !== normalizedColumn)
          : [...current, normalizedColumn];

        let manualChanged = false;
        let manualBounds = previous.manual_bounds;
        if (
          exists &&
          manualBounds &&
          typeof manualBounds === 'object' &&
          !Array.isArray(manualBounds) &&
          Object.prototype.hasOwnProperty.call(manualBounds, normalizedColumn)
        ) {
          manualBounds = { ...manualBounds };
          delete manualBounds[normalizedColumn];
          manualChanged = true;
        }

        const result: Record<string, any> = {
          ...previous,
          columns: nextColumns,
        };

        if (manualChanged) {
          result.manual_bounds = manualBounds;
        }

        if (isBinningNode) {
          const selectedSet = new Set(nextColumns);

          if (
            previous.custom_bins &&
            typeof previous.custom_bins === 'object' &&
            !Array.isArray(previous.custom_bins)
          ) {
            const nextBins: Record<string, number[]> = {};
            Object.entries(previous.custom_bins).forEach(([key, value]) => {
              if (selectedSet.has(key) && Array.isArray(value) && value.length) {
                nextBins[key] = [...value];
              }
            });
            if (Object.keys(nextBins).length) {
              result.custom_bins = nextBins;
            } else if ('custom_bins' in result) {
              delete result.custom_bins;
            }
          } else if ('custom_bins' in result) {
            delete result.custom_bins;
          }

          if (
            previous.custom_labels &&
            typeof previous.custom_labels === 'object' &&
            !Array.isArray(previous.custom_labels)
          ) {
            const nextLabels: Record<string, string[]> = {};
            Object.entries(previous.custom_labels).forEach(([key, value]) => {
              if (selectedSet.has(key) && Array.isArray(value) && value.length) {
                nextLabels[key] = value.map((entry) => String(entry));
              }
            });
            if (Object.keys(nextLabels).length) {
              result.custom_labels = nextLabels;
            } else if ('custom_labels' in result) {
              delete result.custom_labels;
            }
          } else if ('custom_labels' in result) {
            delete result.custom_labels;
          }

          if (
            previous.column_overrides &&
            typeof previous.column_overrides === 'object' &&
            !Array.isArray(previous.column_overrides)
          ) {
            const nextOverrides: Record<string, any> = {};
            Object.entries(previous.column_overrides as Record<string, any>).forEach(([key, value]) => {
              if (selectedSet.has(key) && value && typeof value === 'object' && !Array.isArray(value)) {
                nextOverrides[key] = { ...(value as Record<string, any>) };
              }
            });
            if (Object.keys(nextOverrides).length) {
              result.column_overrides = nextOverrides;
              result.column_strategies = nextOverrides;
            } else {
              if ('column_overrides' in result) {
                delete result.column_overrides;
              }
              if ('column_strategies' in result) {
                delete result.column_strategies;
              }
            }
          } else {
            if ('column_overrides' in result) {
              delete result.column_overrides;
            }
            if ('column_strategies' in result) {
              delete result.column_strategies;
            }
          }
        }

        if (isScalingNode) {
          let nextMethods: Record<string, string> | null = null;
          if (
            previous.column_methods &&
            typeof previous.column_methods === 'object' &&
            !Array.isArray(previous.column_methods)
          ) {
            nextMethods = {};
            Object.entries(previous.column_methods as Record<string, any>).forEach(([key, value]) => {
              const methodKey = typeof value === 'string' ? value.trim() : null;
              if (!methodKey || !SCALING_METHOD_ORDER.includes(methodKey as any)) {
                return;
              }
              if (exists && key === normalizedColumn) {
                return;
              }
              nextMethods![key] = methodKey;
            });
          }

          if (nextMethods && Object.keys(nextMethods).length) {
            result.column_methods = nextMethods;
          } else if (exists && Object.prototype.hasOwnProperty.call(result, 'column_methods')) {
            delete result.column_methods;
          }
        }

        return result;
      });
    },
    [isBinningNode, isScalingNode, setConfigState]
  );

  const handleApplyAllRecommended = useCallback(() => {
    if (!recommendations.length) {
      return;
    }
    setConfigState((previous) => {
      const next = new Set(ensureArrayOfString(previous.columns));
      recommendations.forEach((candidate) => {
        if (candidate?.name) {
          next.add(String(candidate.name));
        }
      });
      return {
        ...previous,
        columns: Array.from(next),
      };
    });
  }, [recommendations, setConfigState]);

  const handleSelectAllColumns = useCallback(() => {
    if (!availableColumns.length) {
      return;
    }
    setConfigState((previous) => {
      const aggregate = new Set(ensureArrayOfString(previous.columns));
      const eligibleColumns = availableColumns.filter((column) => {
        if (isBinningNode) {
          return !binningExcludedColumns.has(column);
        }
        if (isScalingNode) {
          return !scalingExcludedColumns.has(column);
        }
        return true;
      });
      eligibleColumns.forEach((column) => aggregate.add(column));
      return {
        ...previous,
        columns: Array.from(aggregate).sort((a, b) => a.localeCompare(b)),
      };
    });
  }, [availableColumns, binningExcludedColumns, isBinningNode, isScalingNode, scalingExcludedColumns, setConfigState]);

  const handleClearColumns = useCallback(() => {
    setConfigState((previous) => {
      const next: Record<string, any> = {
        ...previous,
        columns: [],
      };
      if (
        previous?.manual_bounds &&
        typeof previous.manual_bounds === 'object' &&
        !Array.isArray(previous.manual_bounds) &&
        Object.keys(previous.manual_bounds).length
      ) {
        next.manual_bounds = {};
      }
      if (isBinningNode) {
        if ('custom_bins' in next) {
          delete next.custom_bins;
        }
        if ('custom_labels' in next) {
          delete next.custom_labels;
        }
        if ('column_overrides' in next) {
          delete next.column_overrides;
        }
        if ('column_strategies' in next) {
          delete next.column_strategies;
        }
      }
      if (isScalingNode && 'column_methods' in next) {
        delete next.column_methods;
      }
      return next;
    });
  }, [isBinningNode, isScalingNode, setConfigState]);

  const handleRemoveColumn = useCallback(
    (column: string) => {
      const normalizedColumn = sanitizeColumnName(column);
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous) => {
        const current = ensureArrayOfString(previous.columns);
        if (!current.includes(normalizedColumn)) {
          return previous;
        }
        const nextColumns = current.filter((item) => item !== normalizedColumn);
        let manualChanged = false;
        let manualBounds = previous.manual_bounds;
        if (
          manualBounds &&
          typeof manualBounds === 'object' &&
          !Array.isArray(manualBounds) &&
          Object.prototype.hasOwnProperty.call(manualBounds, normalizedColumn)
        ) {
          manualBounds = { ...manualBounds };
          delete manualBounds[normalizedColumn];
          manualChanged = true;
        }
        const result: Record<string, any> = {
          ...previous,
          columns: nextColumns,
        };

        if (manualChanged) {
          result.manual_bounds = manualBounds;
        }

        if (isBinningNode) {
          if (
            previous.custom_bins &&
            typeof previous.custom_bins === 'object' &&
            !Array.isArray(previous.custom_bins)
          ) {
            const nextBins: Record<string, number[]> = {};
            Object.entries(previous.custom_bins).forEach(([key, value]) => {
              if (key !== normalizedColumn && Array.isArray(value) && value.length) {
                nextBins[key] = [...value];
              }
            });
            if (Object.keys(nextBins).length) {
              result.custom_bins = nextBins;
            } else if ('custom_bins' in result) {
              delete result.custom_bins;
            }
          } else if ('custom_bins' in result) {
            delete result.custom_bins;
          }

          if (
            previous.custom_labels &&
            typeof previous.custom_labels === 'object' &&
            !Array.isArray(previous.custom_labels)
          ) {
            const nextLabels: Record<string, string[]> = {};
            Object.entries(previous.custom_labels).forEach(([key, value]) => {
              if (key !== normalizedColumn && Array.isArray(value) && value.length) {
                nextLabels[key] = value.map((entry) => String(entry));
              }
            });
            if (Object.keys(nextLabels).length) {
              result.custom_labels = nextLabels;
            } else if ('custom_labels' in result) {
              delete result.custom_labels;
            }
          } else if ('custom_labels' in result) {
            delete result.custom_labels;
          }

          if (
            previous.column_overrides &&
            typeof previous.column_overrides === 'object' &&
            !Array.isArray(previous.column_overrides)
          ) {
            const nextOverrides: Record<string, any> = {};
            Object.entries(previous.column_overrides as Record<string, any>).forEach(([key, value]) => {
              if (key !== normalizedColumn && value && typeof value === 'object' && !Array.isArray(value)) {
                nextOverrides[key] = { ...(value as Record<string, any>) };
              }
            });
            if (Object.keys(nextOverrides).length) {
              result.column_overrides = nextOverrides;
              result.column_strategies = nextOverrides;
            } else {
              if ('column_overrides' in result) {
                delete result.column_overrides;
              }
              if ('column_strategies' in result) {
                delete result.column_strategies;
              }
            }
          } else {
            if ('column_overrides' in result) {
              delete result.column_overrides;
            }
            if ('column_strategies' in result) {
              delete result.column_strategies;
            }
          }
        }

        if (isScalingNode) {
          if (
            previous.column_methods &&
            typeof previous.column_methods === 'object' &&
            !Array.isArray(previous.column_methods)
          ) {
            const nextMethods: Record<string, string> = {};
            Object.entries(previous.column_methods as Record<string, any>).forEach(([key, value]) => {
              const methodKey = typeof value === 'string' ? value.trim() : null;
              if (!methodKey || !SCALING_METHOD_ORDER.includes(methodKey as any)) {
                return;
              }
              if (key === normalizedColumn) {
                return;
              }
              nextMethods[key] = methodKey;
            });
            if (Object.keys(nextMethods).length) {
              result.column_methods = nextMethods;
            } else if (Object.prototype.hasOwnProperty.call(result, 'column_methods')) {
              delete result.column_methods;
            }
          } else if (Object.prototype.hasOwnProperty.call(result, 'column_methods')) {
            delete result.column_methods;
          }
        }

        return result;
      });
    },
    [isBinningNode, isScalingNode, setConfigState]
  );

  return {
    handleManualBoundChange,
    handleClearManualBound,
    handleToggleColumn,
    handleApplyAllRecommended,
    handleSelectAllColumns,
    handleClearColumns,
    handleRemoveColumn,
  };
};
