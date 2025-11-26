import { useCallback } from 'react';
import {
  DATE_MODE_OPTIONS,
  normalizeDateFormatStrategies,
  resolveDateMode,
  serializeDateFormatStrategies,
  type DateFormatStrategyConfig,
  type DateMode,
} from '../nodes/standardize_date/standardizeDateSettings';
import { ensureArrayOfString } from '../sharedUtils';

interface UseDateStrategyHandlersProps {
  isStandardizeDatesNode: boolean;
  node: any;
  setConfigState: React.Dispatch<React.SetStateAction<any>>;
  setCollapsedStrategies: React.Dispatch<React.SetStateAction<Set<number>>>;
  dateStrategies: DateFormatStrategyConfig[];
  standardizeDatesColumnSummary: { recommendedColumns: string[] };
  standardizeDatesMode: DateMode;
}

export const useDateStrategyHandlers = ({
  isStandardizeDatesNode,
  node,
  setConfigState,
  setCollapsedStrategies,
  dateStrategies,
  standardizeDatesColumnSummary,
  standardizeDatesMode,
}: UseDateStrategyHandlersProps) => {
  const updateDateStrategies = useCallback(
    (updater: (current: DateFormatStrategyConfig[]) => DateFormatStrategyConfig[]) => {
      setConfigState((previous: any) => {
        if (!isStandardizeDatesNode) {
          return previous;
        }

        const fallbackColumns = ensureArrayOfString(previous?.columns ?? node?.data?.config?.columns);
        const fallbackMode = resolveDateMode(previous?.mode, node?.data?.config?.mode);
        const fallbackAutoDetect = fallbackColumns.length === 0;

        const currentStrategies = normalizeDateFormatStrategies(
          previous?.format_strategies ?? node?.data?.config?.format_strategies,
          {
            columns: fallbackColumns,
            mode: fallbackMode,
            autoDetect: fallbackAutoDetect,
          },
        );

        const nextStrategies = updater(currentStrategies);
        const serialized = serializeDateFormatStrategies(nextStrategies);
        const unionColumns = Array.from(
          new Set(
            nextStrategies.flatMap((strategy) =>
              strategy.columns.map((column) => column.trim()).filter(Boolean),
            ),
          ),
        ).sort((a, b) => a.localeCompare(b));
        const primaryMode = nextStrategies[0]?.mode ?? fallbackMode;

        return {
          ...previous,
          format_strategies: serialized,
          columns: unionColumns,
          mode: primaryMode,
        };
      });
    },
    [isStandardizeDatesNode, node?.data?.config?.columns, node?.data?.config?.format_strategies, node?.data?.config?.mode, setConfigState],
  );

  const handleAddDateStrategy = useCallback(() => {
    const nextIndex = dateStrategies.length;
    updateDateStrategies((current) => {
      const assigned = new Set<string>();
      current.forEach((strategy) => {
        strategy.columns.forEach((column) => {
          const normalized = column.trim();
          if (normalized) {
            assigned.add(normalized);
          }
        });
      });
      const suggested = standardizeDatesColumnSummary.recommendedColumns.find((column) => !assigned.has(column));
      const defaultMode = current.length
        ? current[current.length - 1].mode
        : DATE_MODE_OPTIONS[0]?.value ?? standardizeDatesMode;
      const nextStrategy: DateFormatStrategyConfig = {
        mode: defaultMode,
        columns: suggested ? [suggested] : [],
        autoDetect: !suggested,
      };
      return [...current, nextStrategy];
    });
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      next.delete(nextIndex);
      return next;
    });
  }, [dateStrategies.length, setCollapsedStrategies, standardizeDatesColumnSummary.recommendedColumns, standardizeDatesMode, updateDateStrategies]);

  const handleRemoveDateStrategy = useCallback(
    (index: number) => {
      updateDateStrategies((current) => current.filter((_, idx) => idx !== index));
      setCollapsedStrategies((previous) => {
        const next = new Set<number>();
        previous.forEach((value) => {
          if (value === index) {
            return;
          }
          next.add(value > index ? value - 1 : value);
        });
        return next;
      });
    },
    [setCollapsedStrategies, updateDateStrategies],
  );

  const toggleDateStrategySection = useCallback((index: number) => {
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, [setCollapsedStrategies]);

  const handleDateStrategyModeChange = useCallback(
    (index: number, mode: DateMode) => {
      updateDateStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, mode } : strategy)),
      );
    },
    [updateDateStrategies],
  );

  const handleDateStrategyColumnsChange = useCallback(
    (index: number, value: string) => {
      const normalized = value
        .split(',')
        .map((column) => column.trim())
        .filter(Boolean)
        .sort((a, b) => a.localeCompare(b));
      updateDateStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, columns: normalized } : strategy)),
      );
    },
    [updateDateStrategies],
  );

  const handleDateStrategyColumnToggle = useCallback(
    (index: number, column: string) => {
      const normalized = column.trim();
      if (!normalized) {
        return;
      }
      updateDateStrategies((current) =>
        current.map((strategy, idx) => {
          if (idx !== index) {
            return strategy;
          }
          const hasColumn = strategy.columns.includes(normalized);
          const nextColumns = hasColumn
            ? strategy.columns.filter((entry) => entry !== normalized)
            : [...strategy.columns, normalized];
          nextColumns.sort((a, b) => a.localeCompare(b));
          return {
            ...strategy,
            columns: nextColumns,
          };
        }),
      );
    },
    [updateDateStrategies],
  );

  const handleDateStrategyAutoDetectToggle = useCallback(
    (index: number, enabled: boolean) => {
      updateDateStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, autoDetect: enabled } : strategy)),
      );
    },
    [updateDateStrategies],
  );

  return {
    updateDateStrategies,
    handleAddDateStrategy,
    handleRemoveDateStrategy,
    toggleDateStrategySection,
    handleDateStrategyModeChange,
    handleDateStrategyColumnsChange,
    handleDateStrategyColumnToggle,
    handleDateStrategyAutoDetectToggle,
  };
};
