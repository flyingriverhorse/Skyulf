import { useCallback } from 'react';
import {
  type ScalingMethodName,
  SCALING_METHOD_ORDER,
} from '../nodes/scaling/scalingSettings';
import { ensureArrayOfString } from '../sharedUtils';

type UseScalingHandlersProps = {
  setConfigState: (updater: any) => void;
  scalingRecommendations: any[];
};

export const useScalingHandlers = ({
  setConfigState,
  scalingRecommendations,
}: UseScalingHandlersProps) => {
  const handleScalingDefaultMethodChange = useCallback(
    (method: ScalingMethodName) => {
      if (!SCALING_METHOD_ORDER.includes(method)) {
        return;
      }
      setConfigState((previous: any) => {
        if (previous?.default_method === method) {
          return previous;
        }
        return {
          ...previous,
          default_method: method,
        };
      });
    },
    [setConfigState]
  );

  const handleScalingAutoDetectToggle = useCallback(
    (value: boolean) => {
      setConfigState((previous: any) => {
        const current = typeof previous?.auto_detect === 'boolean' ? previous.auto_detect : true;
        if (current === value) {
          return previous;
        }
        return {
          ...previous,
          auto_detect: value,
        };
      });
    },
    [setConfigState]
  );

  const setScalingColumnMethod = useCallback(
    (column: string, method: ScalingMethodName | null) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }

      setConfigState((previous: any) => {
        const currentColumns = ensureArrayOfString(previous.columns);
        const hasColumn = currentColumns.includes(normalized);
        const existingMethods =
          previous.column_methods &&
          typeof previous.column_methods === 'object' &&
          !Array.isArray(previous.column_methods)
            ? { ...(previous.column_methods as Record<string, ScalingMethodName>) }
            : {};

        if (method) {
          if (!SCALING_METHOD_ORDER.includes(method)) {
            return previous;
          }
          const currentMethod = existingMethods[normalized] ?? null;
          const needsMethodUpdate = currentMethod !== method;
          const needsColumnUpdate = !hasColumn;
          if (!needsMethodUpdate && !needsColumnUpdate) {
            return previous;
          }
          existingMethods[normalized] = method;
          const nextColumns = needsColumnUpdate
            ? [...currentColumns, normalized].sort((a, b) => a.localeCompare(b))
            : currentColumns;
          return {
            ...previous,
            columns: nextColumns,
            column_methods: existingMethods,
          };
        }

        if (!Object.prototype.hasOwnProperty.call(existingMethods, normalized)) {
          return previous;
        }

        delete existingMethods[normalized];
        if (Object.keys(existingMethods).length) {
          return {
            ...previous,
            column_methods: existingMethods,
          };
        }

        const nextState: Record<string, any> = {
          ...previous,
        };
        if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
          delete nextState.column_methods;
        }
        return nextState;
      });
    },
    [setConfigState]
  );

  const handleScalingClearOverrides = useCallback(() => {
    setConfigState((previous: any) => {
      if (
        !previous.column_methods ||
        typeof previous.column_methods !== 'object' ||
        Array.isArray(previous.column_methods) ||
        !Object.keys(previous.column_methods).length
      ) {
        return previous;
      }
      const nextState = { ...previous } as Record<string, any>;
      delete nextState.column_methods;
      return nextState;
    });
  }, [setConfigState]);

  const handleScalingApplyAllRecommendations = useCallback(() => {
    if (!scalingRecommendations.length) {
      return;
    }
    setConfigState((previous: any) => {
      const currentColumns = ensureArrayOfString(previous.columns);
      const columnSet = new Set(currentColumns);
      const skippedSet = new Set(ensureArrayOfString(previous.skipped_columns));
      const nextMethods =
        previous.column_methods &&
        typeof previous.column_methods === 'object' &&
        !Array.isArray(previous.column_methods)
          ? { ...(previous.column_methods as Record<string, ScalingMethodName>) }
          : {};

      let updated = false;

      scalingRecommendations.forEach((entry) => {
        const normalized = String(entry.column ?? '').trim();
        const method = entry.recommended_method;
        if (!normalized || !SCALING_METHOD_ORDER.includes(method)) {
          return;
        }
        if (skippedSet.has(normalized)) {
          return;
        }
        if (nextMethods[normalized] !== method) {
          nextMethods[normalized] = method;
          updated = true;
        }
        if (!columnSet.has(normalized)) {
          columnSet.add(normalized);
          updated = true;
        }
      });

      if (!updated) {
        return previous;
      }

      const nextColumns = Array.from(columnSet).sort((a, b) => a.localeCompare(b));

      return {
        ...previous,
        columns: nextColumns,
        column_methods: nextMethods,
      };
    });
  }, [scalingRecommendations, setConfigState]);

  const handleScalingSkipColumn = useCallback(
    (column: string) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }
      setConfigState((previous: any) => {
        const currentColumns = ensureArrayOfString(previous.columns);
        const filteredColumns = currentColumns.filter((value) => value !== normalized);

        const skippedSet = new Set(ensureArrayOfString(previous.skipped_columns));
        const sizeBefore = skippedSet.size;
        skippedSet.add(normalized);
        const skipChanged = skippedSet.size !== sizeBefore;

        const rawColumnMethods =
          previous.column_methods && typeof previous.column_methods === 'object' && !Array.isArray(previous.column_methods)
            ? { ...(previous.column_methods as Record<string, ScalingMethodName>) }
            : {};
        const hadMethod = Object.prototype.hasOwnProperty.call(rawColumnMethods, normalized);
        if (hadMethod) {
          delete rawColumnMethods[normalized];
        }

        if (!skipChanged && filteredColumns.length === currentColumns.length && !hadMethod) {
          return previous;
        }

        return {
          ...previous,
          columns: filteredColumns.sort((a, b) => a.localeCompare(b)),
          column_methods: rawColumnMethods,
          skipped_columns: Array.from(skippedSet).sort((a, b) => a.localeCompare(b)),
        };
      });
    },
    [setConfigState]
  );

  const handleScalingUnskipColumn = useCallback(
    (column: string) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }
      setConfigState((previous: any) => {
        const skipped = ensureArrayOfString(previous.skipped_columns);
        if (!skipped.includes(normalized)) {
          return previous;
        }

        const nextSkippedSet = new Set(skipped);
        nextSkippedSet.delete(normalized);

        const currentColumns = new Set(ensureArrayOfString(previous.columns));
        const hadColumn = currentColumns.has(normalized);
        if (!hadColumn) {
          currentColumns.add(normalized);
        }

        return {
          ...previous,
          columns: Array.from(currentColumns).sort((a, b) => a.localeCompare(b)),
          skipped_columns: Array.from(nextSkippedSet).sort((a, b) => a.localeCompare(b)),
        };
      });
    },
    [setConfigState]
  );

  const handleScalingOverrideSelect = useCallback(
    (column: string, value: string) => {
      if (value === '__skip__') {
        handleScalingSkipColumn(column);
        return;
      }

      if (value === '__default__') {
        handleScalingUnskipColumn(column);
        setScalingColumnMethod(column, null);
        return;
      }

      if (SCALING_METHOD_ORDER.includes(value as ScalingMethodName)) {
        handleScalingUnskipColumn(column);
        setScalingColumnMethod(column, value as ScalingMethodName);
      }
    },
    [handleScalingSkipColumn, handleScalingUnskipColumn, setScalingColumnMethod]
  );

  return {
    handleScalingDefaultMethodChange,
    handleScalingAutoDetectToggle,
    setScalingColumnMethod,
    handleScalingClearOverrides,
    handleScalingApplyAllRecommendations,
    handleScalingSkipColumn,
    handleScalingUnskipColumn,
    handleScalingOverrideSelect,
  };
};
