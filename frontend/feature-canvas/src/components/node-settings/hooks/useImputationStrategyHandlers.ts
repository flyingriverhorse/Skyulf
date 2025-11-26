import { useCallback, Dispatch, SetStateAction } from 'react';
import {
  ImputationMethodOption,
  ImputationStrategyConfig,
  ImputationStrategyMethod,
  ImputationStrategyOptions,
  sanitizeOptionsForMethod,
  buildDefaultOptionsForMethod,
  normalizeImputationStrategies,
  serializeImputationStrategies,
} from '../nodes/imputation/imputationSettings';

type UseImputationStrategyHandlersProps = {
  setConfigState: Dispatch<SetStateAction<any>>;
  imputationMethodValues: ImputationStrategyMethod[];
  imputationMethodOptions: ImputationMethodOption[];
  imputerStrategyCount: number;
  setCollapsedStrategies: Dispatch<SetStateAction<Set<number>>>;
  setImputerMissingFilter: (value: number) => void;
};

export const useImputationStrategyHandlers = ({
  setConfigState,
  imputationMethodValues,
  imputationMethodOptions,
  imputerStrategyCount,
  setCollapsedStrategies,
  setImputerMissingFilter,
}: UseImputationStrategyHandlersProps) => {
  const updateImputerStrategies = useCallback(
    (updater: (current: ImputationStrategyConfig[]) => ImputationStrategyConfig[]) => {
      setConfigState((previous: any) => {
        const currentStrategies = normalizeImputationStrategies(previous?.strategies, imputationMethodValues);
        const nextStrategies = updater(currentStrategies).map((strategy) => ({
          ...strategy,
          options: sanitizeOptionsForMethod(strategy.method, strategy.options),
        }));
        return {
          ...previous,
          strategies: serializeImputationStrategies(nextStrategies),
        };
      });
    },
    [imputationMethodValues, setConfigState]
  );

  const handleAddImputerStrategy = useCallback(() => {
    const nextIndex = imputerStrategyCount;
    const defaultMethod = imputationMethodOptions[0]?.value ?? 'mean';
    updateImputerStrategies((current) => [
      ...current,
      {
        method: defaultMethod,
        columns: [],
        options: sanitizeOptionsForMethod(defaultMethod, buildDefaultOptionsForMethod(defaultMethod)),
      },
    ]);
    setCollapsedStrategies((previous) => {
      const next = new Set(previous);
      next.delete(nextIndex);
      return next;
    });
  }, [imputationMethodOptions, imputerStrategyCount, updateImputerStrategies, setCollapsedStrategies]);

  const handleRemoveImputerStrategy = useCallback(
    (index: number) => {
      updateImputerStrategies((current) => current.filter((_, idx) => idx !== index));
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
    [updateImputerStrategies, setCollapsedStrategies]
  );

  const toggleImputerStrategySection = useCallback((index: number) => {
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

  const handleImputerMethodChange = useCallback(
    (index: number, method: ImputationStrategyMethod) => {
      updateImputerStrategies((current) =>
        current.map((strategy, idx) =>
          idx === index
            ? {
                ...strategy,
                method,
                options: sanitizeOptionsForMethod(method, strategy.options),
              }
            : strategy
        )
      );
    },
    [updateImputerStrategies]
  );

  const handleImputerOptionNumberChange = useCallback(
    (index: number, key: 'neighbors' | 'max_iter', rawValue: string) => {
      updateImputerStrategies((current) =>
        current.map((strategy, idx) => {
          if (idx !== index) {
            return strategy;
          }
          const nextOptions: ImputationStrategyOptions = {
            ...(strategy.options ?? {}),
          };
          if (rawValue === '') {
            delete nextOptions[key];
          } else {
            const parsed = Number(rawValue);
            if (!Number.isFinite(parsed)) {
              return strategy;
            }
            nextOptions[key] = parsed;
          }
          return {
            ...strategy,
            options: sanitizeOptionsForMethod(strategy.method, nextOptions),
          };
        })
      );
    },
    [updateImputerStrategies]
  );

  const handleImputerColumnsChange = useCallback(
    (index: number, value: string) => {
      const normalized = value
        .split(',')
        .map((column) => column.trim())
        .filter(Boolean);
      updateImputerStrategies((current) =>
        current.map((strategy, idx) => (idx === index ? { ...strategy, columns: normalized } : strategy))
      );
    },
    [updateImputerStrategies]
  );

  const handleImputerColumnToggle = useCallback(
    (index: number, column: string) => {
      const normalized = column.trim();
      if (!normalized) {
        return;
      }

      updateImputerStrategies((current) =>
        current.map((strategy, idx) => {
          if (idx !== index) {
            return strategy;
          }
          const hasColumn = strategy.columns.includes(normalized);
          const nextColumns = hasColumn
            ? strategy.columns.filter((item) => item !== normalized)
            : [...strategy.columns, normalized];
          nextColumns.sort((a, b) => a.localeCompare(b));
          return {
            ...strategy,
            columns: nextColumns,
          };
        })
      );
    },
    [updateImputerStrategies]
  );

  const handleImputerMissingFilterChange = useCallback(
    (value: number) => {
      setImputerMissingFilter(value);
    },
    [setImputerMissingFilter]
  );

  return {
    updateImputerStrategies,
    handleAddImputerStrategy,
    handleRemoveImputerStrategy,
    toggleImputerStrategySection,
    handleImputerMethodChange,
    handleImputerOptionNumberChange,
    handleImputerColumnsChange,
    handleImputerColumnToggle,
    handleImputerMissingFilterChange,
  };
};
