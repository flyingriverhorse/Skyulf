import { useMemo } from 'react';
import { ensureArrayOfString } from '../sharedUtils';
import type {
  ImputationStrategyConfig,
  ImputerColumnOption,
} from '../nodes/imputation/imputationSettings';

export type UseImputationConfigurationArgs = {
  isImputerNode: boolean;
  imputerStrategies: ImputationStrategyConfig[];
  availableColumns: string[];
  columnMissingMap: Record<string, number>;
  previewColumns: string[];
  previewColumnStats: any[] | null | undefined;
  nodeColumns: any;
  imputerMissingFilter: number;
};

export type UseImputationConfigurationResult = {
  imputerColumnOptions: ImputerColumnOption[];
  imputerMissingSliderMax: number;
  imputerFilteredOptionCount: number;
  imputerMissingFilterActive: boolean;
};

export const useImputationConfiguration = ({
  isImputerNode,
  imputerStrategies,
  availableColumns,
  columnMissingMap,
  previewColumns,
  previewColumnStats,
  nodeColumns,
  imputerMissingFilter,
}: UseImputationConfigurationArgs): UseImputationConfigurationResult => {
  const imputerColumnOptions = useMemo<ImputerColumnOption[]>(() => {
    if (!isImputerNode) {
      return [];
    }

    const previewColumnSet = previewColumns.length
      ? new Set(
          previewColumns
            .map((column) => (typeof column === 'string' ? column.trim() : ''))
            .filter((column) => column.length > 0),
        )
      : null;

    const requiredColumns = new Set<string>();
    imputerStrategies.forEach((strategy) => {
      strategy.columns.forEach((column) => {
        const normalized = typeof column === 'string' ? column.trim() : '';
        if (normalized) {
          requiredColumns.add(normalized);
        }
      });
    });
    ensureArrayOfString(nodeColumns).forEach((column) => {
      if (column) {
        requiredColumns.add(column);
      }
    });

    const shouldExcludeColumn = (name: string) => {
      if (!previewColumnSet || previewColumnSet.size === 0) {
        return false;
      }
      return !previewColumnSet.has(name) && !requiredColumns.has(name);
    };

    type OptionAggregate = {
      missingPercentage: number | null;
      dtype?: string | null;
      mean?: number | null;
      median?: number | null;
      mode?: string | number | null;
    };

    const optionMap = new Map<string, OptionAggregate>();

    const registerColumn = (name?: string | null, updates?: Partial<OptionAggregate>) => {
      const normalized = typeof name === 'string' ? name.trim() : '';
      if (!normalized) {
        return;
      }

      if (shouldExcludeColumn(normalized)) {
        return;
      }

      const entry = optionMap.get(normalized) ?? {
        missingPercentage: null,
        dtype: undefined,
        mean: null,
        median: null,
        mode: null,
      };

      if (updates) {
        if (typeof updates.missingPercentage === 'number' && !Number.isNaN(updates.missingPercentage)) {
          entry.missingPercentage = Math.max(0, updates.missingPercentage);
        }
        if (updates.dtype !== undefined) {
          entry.dtype = updates.dtype;
        }
        if (typeof updates.mean === 'number' && !Number.isNaN(updates.mean)) {
          entry.mean = updates.mean;
        }
        if (typeof updates.median === 'number' && !Number.isNaN(updates.median)) {
          entry.median = updates.median;
        }
        if (updates.mode !== undefined && updates.mode !== null && updates.mode !== '') {
          entry.mode = updates.mode;
        }
      }

      optionMap.set(normalized, entry);
    };

    availableColumns.forEach((column) => {
      const missingValue = Object.prototype.hasOwnProperty.call(columnMissingMap, column)
        ? columnMissingMap[column]
        : null;
      registerColumn(column, {
        missingPercentage: typeof missingValue === 'number' ? missingValue : null,
      });
    });

    const statsArray = Array.isArray(previewColumnStats) ? previewColumnStats : [];
    statsArray.forEach((stat) => {
      if (!stat || !stat.name) {
        return;
      }
      registerColumn(stat.name, {
        missingPercentage: typeof stat.missing_percentage === 'number' ? stat.missing_percentage : null,
        dtype: stat.dtype ?? null,
        mean: typeof stat.mean === 'number' ? stat.mean : null,
        median: typeof stat.median === 'number' ? stat.median : null,
        mode: stat.mode ?? null,
      });
    });

    ensureArrayOfString(nodeColumns).forEach((column) => registerColumn(column));

    imputerStrategies.forEach((strategy) => {
      strategy.columns.forEach((column) => registerColumn(column));
    });

    const entries = Array.from(optionMap.entries()).map(([name, aggregate]) => ({
      name,
      missingPercentage:
        typeof aggregate.missingPercentage === 'number' && !Number.isNaN(aggregate.missingPercentage)
          ? Math.max(0, aggregate.missingPercentage)
          : null,
      dtype: aggregate.dtype ?? null,
      mean: typeof aggregate.mean === 'number' && !Number.isNaN(aggregate.mean) ? aggregate.mean : null,
      median: typeof aggregate.median === 'number' && !Number.isNaN(aggregate.median) ? aggregate.median : null,
      mode:
        aggregate.mode !== undefined && aggregate.mode !== null && aggregate.mode !== '' ? aggregate.mode : null,
    }));

    entries.sort((a, b) => {
      const aHasMissing = typeof a.missingPercentage === 'number';
      const bHasMissing = typeof b.missingPercentage === 'number';

      if (aHasMissing && bHasMissing && a.missingPercentage !== b.missingPercentage) {
        return (b.missingPercentage as number) - (a.missingPercentage as number);
      }

      if (aHasMissing !== bHasMissing) {
        return aHasMissing ? -1 : 1;
      }

      return a.name.localeCompare(b.name);
    });

    return entries;
  }, [
    availableColumns,
    columnMissingMap,
    imputerStrategies,
    nodeColumns,
    previewColumnStats,
    previewColumns,
    isImputerNode,
  ]);

  const imputerMissingSliderMax = useMemo(() => {
    if (!imputerColumnOptions.length) {
      return 100;
    }
    const highest = imputerColumnOptions.reduce((max, option) => {
      const value = typeof option.missingPercentage === 'number' ? option.missingPercentage : 0;
      return value > max ? value : max;
    }, 0);
    const normalized = Math.ceil(highest / 5) * 5;
    return Math.max(100, normalized || 0);
  }, [imputerColumnOptions]);

  const imputerFilteredOptionCount = useMemo(() => {
    if (!imputerColumnOptions.length) {
      return 0;
    }
    return imputerColumnOptions.reduce((count, option) => {
      const value = typeof option.missingPercentage === 'number' ? option.missingPercentage : null;
      if (value !== null) {
        if (value <= 0) {
          return count;
        }
        return value >= imputerMissingFilter ? count + 1 : count;
      }
      return imputerMissingFilter === 0 ? count + 1 : count;
    }, 0);
  }, [imputerColumnOptions, imputerMissingFilter]);

  const imputerMissingFilterActive = imputerMissingFilter > 0;

  return {
    imputerColumnOptions,
    imputerMissingSliderMax,
    imputerFilteredOptionCount,
    imputerMissingFilterActive,
  };
};
