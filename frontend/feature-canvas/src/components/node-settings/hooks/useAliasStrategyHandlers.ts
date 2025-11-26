import { useCallback } from 'react';
import {
  ALIAS_MODE_OPTIONS,
  DEFAULT_ALIAS_MODE,
  normalizeAliasStrategies,
  resolveAliasMode,
  serializeAliasStrategies,
  type AliasMode,
  type AliasStrategyConfig,
} from '../nodes/replace_aliases/replaceAliasesSettings';
import { ensureArrayOfString } from '../sharedUtils';
import { pickAutoDetectValue, normalizeConfigBoolean } from '../utils/configParsers';

interface UseAliasStrategyHandlersProps {
  isReplaceAliasesNode: boolean;
  node: any;
  setConfigState: React.Dispatch<React.SetStateAction<any>>;
  setCollapsedStrategies: React.Dispatch<React.SetStateAction<Set<number>>>;
  aliasColumnSummary: { recommendedColumns: string[] };
  aliasStrategyCount: number;
}

export const useAliasStrategyHandlers = ({
  isReplaceAliasesNode,
  node,
  setConfigState,
  setCollapsedStrategies,
  aliasColumnSummary,
  aliasStrategyCount,
}: UseAliasStrategyHandlersProps) => {
  const updateAliasStrategies = useCallback(
    (updater: (current: AliasStrategyConfig[]) => AliasStrategyConfig[]) => {
      if (!isReplaceAliasesNode) {
        return;
      }

      setConfigState((previous: any) => {
        if (!isReplaceAliasesNode) {
          return previous;
        }

        const fallbackColumns = ensureArrayOfString(previous?.columns ?? node?.data?.config?.columns);
        const fallbackMode = resolveAliasMode(previous?.mode, node?.data?.config?.mode);

        const fallbackAutoDetect = (() => {
          const localValue = pickAutoDetectValue(previous as Record<string, unknown>);
          if (localValue !== undefined) {
            const normalized = normalizeConfigBoolean(localValue);
            if (normalized !== null) {
              return normalized;
            }
          }
          const nodeValue = pickAutoDetectValue(node?.data?.config as Record<string, unknown> | undefined);
          if (nodeValue !== undefined) {
            const normalized = normalizeConfigBoolean(nodeValue);
            if (normalized !== null) {
              return normalized;
            }
          }
          return fallbackColumns.length === 0;
        })();

        const rawStrategies =
          previous?.alias_strategies ??
          previous?.strategies ??
          node?.data?.config?.alias_strategies ??
          node?.data?.config?.strategies;

        const currentStrategies = normalizeAliasStrategies(rawStrategies, {
          mode: fallbackMode,
          columns: fallbackColumns,
          autoDetect: fallbackAutoDetect,
        });

        const nextStrategies = updater(currentStrategies).map((strategy) => ({
          mode: resolveAliasMode(strategy.mode, fallbackMode),
          columns: Array.from(
            new Set(strategy.columns.map((column) => column.trim()).filter(Boolean)),
          ).sort((a, b) => a.localeCompare(b)),
          autoDetect: Boolean(strategy.autoDetect),
        }));

        const serialized = serializeAliasStrategies(nextStrategies);
        const unionColumns = Array.from(
          new Set(
            nextStrategies.flatMap((strategy) => strategy.columns),
          ),
        ).sort((a, b) => a.localeCompare(b));

        const primaryMode = nextStrategies[0]?.mode ?? fallbackMode;
        const autoDetectAny = nextStrategies.length
          ? nextStrategies.some((strategy) => strategy.autoDetect)
          : fallbackAutoDetect;

        return {
          ...previous,
          alias_strategies: serialized,
          columns: unionColumns,
          mode: primaryMode,
          auto_detect: autoDetectAny,
        };
      });
    },
    [isReplaceAliasesNode, node?.data?.config, setConfigState]
  );

  const handleAddAliasStrategy = useCallback(() => {
    if (!isReplaceAliasesNode) {
      return;
    }
    const nextIndex = aliasStrategyCount;
    updateAliasStrategies((current) => {
      const assigned = new Set<string>();
      current.forEach((strategy) => {
        strategy.columns.forEach((column) => {
          const normalized = column.trim();
          if (normalized) {
            assigned.add(normalized);
          }
        });
      });
      const suggested = aliasColumnSummary.recommendedColumns.find((column) => !assigned.has(column));
      const defaultMode = current.length
        ? current[current.length - 1].mode
        : ALIAS_MODE_OPTIONS[0]?.value ?? DEFAULT_ALIAS_MODE;
      const nextStrategy: AliasStrategyConfig = {
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
  }, [aliasColumnSummary.recommendedColumns, aliasStrategyCount, isReplaceAliasesNode, setCollapsedStrategies, updateAliasStrategies]);

  const handleRemoveAliasStrategy = useCallback(
    (index: number) => {
      updateAliasStrategies((current) => current.filter((_, idx) => idx !== index));
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
    [setCollapsedStrategies, updateAliasStrategies],
  );

  const toggleAliasStrategySection = useCallback((index: number) => {
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

  const handleAliasModeChange = useCallback(
    (index: number, mode: AliasMode) => {
      updateAliasStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, mode } : strategy)),
      );
    },
    [updateAliasStrategies],
  );

  const handleAliasColumnToggle = useCallback(
    (index: number, column: string) => {
      const normalized = column.trim();
      if (!normalized) {
        return;
      }
      updateAliasStrategies((current) =>
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
    [updateAliasStrategies],
  );

  const handleAliasColumnsChange = useCallback(
    (index: number, value: string) => {
      const entries = value
        .split(',')
        .map((column) => column.trim())
        .filter(Boolean);
      const nextColumns = Array.from(new Set(entries)).sort((a, b) => a.localeCompare(b));
      updateAliasStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, columns: nextColumns } : strategy)),
      );
    },
    [updateAliasStrategies],
  );

  const handleAliasAutoDetectToggle = useCallback(
    (index: number, enabled: boolean) => {
      updateAliasStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, autoDetect: enabled } : strategy)),
      );
    },
    [updateAliasStrategies],
  );

  return {
    updateAliasStrategies,
    handleAddAliasStrategy,
    handleRemoveAliasStrategy,
    toggleAliasStrategySection,
    handleAliasModeChange,
    handleAliasColumnToggle,
    handleAliasColumnsChange,
    handleAliasAutoDetectToggle,
  };
};
