import { useCallback } from 'react';
import {
  type OutlierMethodName,
  OUTLIER_METHOD_ORDER,
} from '../../nodes/outlier/outlierSettings';
import { ensureArrayOfString } from '../../sharedUtils';

type UseOutlierHandlersProps = {
  setConfigState: (updater: any) => void;
  outlierRecommendationRows: any[];
  outlierExcludedColumns: Set<string>;
  outlierConfig: any;
};

export const useOutlierHandlers = ({
  setConfigState,
  outlierRecommendationRows,
  outlierExcludedColumns,
  outlierConfig,
}: UseOutlierHandlersProps) => {
  const handleOutlierDefaultMethodChange = useCallback(
    (method: OutlierMethodName) => {
      if (!OUTLIER_METHOD_ORDER.includes(method)) {
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

  const handleOutlierAutoDetectToggle = useCallback(
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

  const setOutlierColumnMethod = useCallback(
    (column: string, method: OutlierMethodName | null) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }

      setConfigState((previous: any) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const currentColumns = ensureArrayOfString(previous.columns);
        const columnSet = new Set(currentColumns);

        const existingMethods =
          previous.column_methods &&
          typeof previous.column_methods === 'object' &&
          !Array.isArray(previous.column_methods)
            ? { ...(previous.column_methods as Record<string, OutlierMethodName>) }
            : {};

        let changed = false;

        if (method) {
          if (!OUTLIER_METHOD_ORDER.includes(method)) {
            return previous;
          }
          if (existingMethods[normalized] !== method) {
            existingMethods[normalized] = method;
            changed = true;
          }
          if (!columnSet.has(normalized)) {
            columnSet.add(normalized);
            changed = true;
          }
        } else if (Object.prototype.hasOwnProperty.call(existingMethods, normalized)) {
          delete existingMethods[normalized];
          changed = true;
        }

        if (!changed) {
          return previous;
        }

        const nextState: Record<string, any> = {
          ...previous,
        };

        nextState.columns = Array.from(columnSet).sort((a, b) => a.localeCompare(b));

        if (Object.keys(existingMethods).length) {
          nextState.column_methods = existingMethods;
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
          delete nextState.column_methods;
        }

        return nextState;
      });
    },
    [setConfigState]
  );

  const handleOutlierClearOverrides = useCallback(() => {
    setConfigState((previous: any) => {
      if (!previous || typeof previous !== 'object') {
        return previous;
      }

      const hasColumnMethods =
        previous.column_methods &&
        typeof previous.column_methods === 'object' &&
        !Array.isArray(previous.column_methods) &&
        Object.keys(previous.column_methods).length > 0;

      const hasColumnParameters =
        previous.column_parameters &&
        typeof previous.column_parameters === 'object' &&
        !Array.isArray(previous.column_parameters) &&
        Object.keys(previous.column_parameters).length > 0;

      if (!hasColumnMethods && !hasColumnParameters) {
        return previous;
      }

      const nextState: Record<string, any> = { ...previous };

      if (hasColumnMethods) {
        delete nextState.column_methods;
      }

      if (hasColumnParameters) {
        delete nextState.column_parameters;
      }

      return nextState;
    });
  }, [setConfigState]);

  const handleOutlierApplyAllRecommendations = useCallback(() => {
    if (!outlierRecommendationRows.length) {
      return;
    }

    setConfigState((previous: any) => {
      if (!previous || typeof previous !== 'object') {
        return previous;
      }

      const currentColumns = new Set(ensureArrayOfString(previous.columns));
      const skippedSet = new Set(ensureArrayOfString(previous.skipped_columns));
      const nextMethods =
        previous.column_methods &&
        typeof previous.column_methods === 'object' &&
        !Array.isArray(previous.column_methods)
          ? { ...(previous.column_methods as Record<string, OutlierMethodName>) }
          : {};
      const nextParameters =
        previous.column_parameters &&
        typeof previous.column_parameters === 'object' &&
        !Array.isArray(previous.column_parameters)
          ? { ...(previous.column_parameters as Record<string, Record<string, number>>) }
          : {};

      let updated = false;

      outlierRecommendationRows.forEach((row) => {
        if (!row || row.isExcluded || !row.recommendedMethod) {
          return;
        }

        const columnKey = String(row.column ?? '').trim();
        if (!columnKey || outlierExcludedColumns.has(columnKey)) {
          return;
        }

        const recommendedMethod = row.recommendedMethod;
        if (!OUTLIER_METHOD_ORDER.includes(recommendedMethod)) {
          return;
        }

        if (skippedSet.has(columnKey)) {
          skippedSet.delete(columnKey);
          updated = true;
        }

        if (!currentColumns.has(columnKey)) {
          currentColumns.add(columnKey);
          updated = true;
        }

        if (recommendedMethod === outlierConfig.defaultMethod) {
          if (Object.prototype.hasOwnProperty.call(nextMethods, columnKey)) {
            delete nextMethods[columnKey];
            updated = true;
          }
        } else if (nextMethods[columnKey] !== recommendedMethod) {
          nextMethods[columnKey] = recommendedMethod;
          updated = true;
        }

        if (Object.prototype.hasOwnProperty.call(nextParameters, columnKey)) {
          delete nextParameters[columnKey];
          updated = true;
        }
      });

      if (!updated) {
        return previous;
      }

      const nextState: Record<string, any> = {
        ...previous,
        columns: Array.from(currentColumns).sort((a, b) => a.localeCompare(b)),
      };

      if (skippedSet.size) {
        nextState.skipped_columns = Array.from(skippedSet).sort((a, b) => a.localeCompare(b));
      } else if (Object.prototype.hasOwnProperty.call(nextState, 'skipped_columns')) {
        delete nextState.skipped_columns;
      }

      if (Object.keys(nextMethods).length) {
        nextState.column_methods = nextMethods;
      } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
        delete nextState.column_methods;
      }

      if (Object.keys(nextParameters).length) {
        nextState.column_parameters = nextParameters;
      } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_parameters')) {
        delete nextState.column_parameters;
      }

      return nextState;
    });
  }, [outlierConfig.defaultMethod, outlierExcludedColumns, outlierRecommendationRows, setConfigState]);

  const handleOutlierSkipColumn = useCallback(
    (column: string) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }

      setConfigState((previous: any) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const currentColumns = ensureArrayOfString(previous.columns);
        const filteredColumns = currentColumns.filter((value) => value !== normalized);

        const skippedSet = new Set(ensureArrayOfString(previous.skipped_columns));
        const skipSize = skippedSet.size;
        skippedSet.add(normalized);
        const skipChanged = skipSize !== skippedSet.size;

        const existingMethods =
          previous.column_methods &&
          typeof previous.column_methods === 'object' &&
          !Array.isArray(previous.column_methods)
            ? { ...(previous.column_methods as Record<string, OutlierMethodName>) }
            : {};
        const hadMethod = Object.prototype.hasOwnProperty.call(existingMethods, normalized);
        if (hadMethod) {
          delete existingMethods[normalized];
        }

        const existingParameters =
          previous.column_parameters &&
          typeof previous.column_parameters === 'object' &&
          !Array.isArray(previous.column_parameters)
            ? { ...(previous.column_parameters as Record<string, Record<string, number>>) }
            : {};
        const hadParameters = Object.prototype.hasOwnProperty.call(existingParameters, normalized);
        if (hadParameters) {
          delete existingParameters[normalized];
        }

        const columnsChanged = filteredColumns.length !== currentColumns.length;

        if (!columnsChanged && !skipChanged && !hadMethod && !hadParameters) {
          return previous;
        }

        const nextState: Record<string, any> = {
          ...previous,
          columns: filteredColumns.sort((a, b) => a.localeCompare(b)),
          skipped_columns: Array.from(skippedSet).sort((a, b) => a.localeCompare(b)),
        };

        if (Object.keys(existingMethods).length) {
          nextState.column_methods = existingMethods;
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
          delete nextState.column_methods;
        }

        if (Object.keys(existingParameters).length) {
          nextState.column_parameters = existingParameters;
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_parameters')) {
          delete nextState.column_parameters;
        }

        return nextState;
      });
    },
    [setConfigState]
  );

  const handleOutlierUnskipColumn = useCallback(
    (column: string) => {
      const normalized = String(column ?? '').trim();
      if (!normalized) {
        return;
      }

      setConfigState((previous: any) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const skipped = ensureArrayOfString(previous.skipped_columns);
        if (!skipped.includes(normalized)) {
          return previous;
        }

        const nextSkipped = skipped.filter((value) => value !== normalized);
        const currentColumns = new Set(ensureArrayOfString(previous.columns));
        if (!currentColumns.has(normalized)) {
          currentColumns.add(normalized);
        }

        const nextState: Record<string, any> = {
          ...previous,
          columns: Array.from(currentColumns).sort((a, b) => a.localeCompare(b)),
        };

        if (nextSkipped.length) {
          nextState.skipped_columns = nextSkipped.sort((a, b) => a.localeCompare(b));
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'skipped_columns')) {
          delete nextState.skipped_columns;
        }

        return nextState;
      });
    },
    [setConfigState]
  );

  const handleOutlierOverrideSelect = useCallback(
    (column: string, value: string) => {
      if (value === '__skip__') {
        handleOutlierSkipColumn(column);
        return;
      }

      if (value === '__default__') {
        handleOutlierUnskipColumn(column);
        setOutlierColumnMethod(column, null);
        return;
      }

      if (OUTLIER_METHOD_ORDER.includes(value as OutlierMethodName)) {
        handleOutlierUnskipColumn(column);
        setOutlierColumnMethod(column, value as OutlierMethodName);
      }
    },
    [handleOutlierSkipColumn, handleOutlierUnskipColumn, setOutlierColumnMethod]
  );

  const handleOutlierMethodParameterChange = useCallback(
    (method: OutlierMethodName, parameter: string, value: number | null) => {
      const normalizedParameter = String(parameter ?? '').trim().toLowerCase();

      setConfigState((previous: any) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const baseParameters: Partial<Record<OutlierMethodName, Record<string, number>>> =
          previous.method_parameters &&
          typeof previous.method_parameters === 'object' &&
          !Array.isArray(previous.method_parameters)
            ? { ...(previous.method_parameters as Record<OutlierMethodName, Record<string, number>>) }
            : {};

        const methodParameters = {
          ...(baseParameters[method] ?? {}),
        };

        const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : null;
        let changed = false;

        if (numericValue === null) {
          if (Object.prototype.hasOwnProperty.call(methodParameters, normalizedParameter)) {
            delete methodParameters[normalizedParameter];
            changed = true;
          }
        } else if (methodParameters[normalizedParameter] !== numericValue) {
          methodParameters[normalizedParameter] = numericValue;
          changed = true;
        }

        if (!changed) {
          return previous;
        }

        if (Object.keys(methodParameters).length) {
          baseParameters[method] = methodParameters;
        } else {
          delete baseParameters[method];
        }

        if (!Object.keys(baseParameters).length) {
          const nextState = { ...previous };
          if (Object.prototype.hasOwnProperty.call(nextState, 'method_parameters')) {
            delete nextState.method_parameters;
          }
          return nextState;
        }

        return {
          ...previous,
          method_parameters: baseParameters,
        };
      });
    },
    [setConfigState]
  );

  const handleOutlierColumnParameterChange = useCallback(
    (column: string, parameter: string, value: number | null) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      const normalizedParameter = String(parameter ?? '').trim().toLowerCase();

      setConfigState((previous: any) => {
        if (!previous || typeof previous !== 'object') {
          return previous;
        }

        const baseParameters =
          previous.column_parameters &&
          typeof previous.column_parameters === 'object' &&
          !Array.isArray(previous.column_parameters)
            ? { ...(previous.column_parameters as Record<string, Record<string, number>>) }
            : {};

        const columnParameters = {
          ...(baseParameters[normalizedColumn] ?? {}),
        };

        const numericValue = typeof value === 'number' && Number.isFinite(value) ? value : null;
        let changed = false;

        if (numericValue === null) {
          if (Object.prototype.hasOwnProperty.call(columnParameters, normalizedParameter)) {
            delete columnParameters[normalizedParameter];
            changed = true;
          }
        } else if (columnParameters[normalizedParameter] !== numericValue) {
          columnParameters[normalizedParameter] = numericValue;
          changed = true;
        }

        if (!changed) {
          return previous;
        }

        if (Object.keys(columnParameters).length) {
          baseParameters[normalizedColumn] = columnParameters;
        } else {
          delete baseParameters[normalizedColumn];
        }

        if (!Object.keys(baseParameters).length) {
          const nextState = { ...previous };
          if (Object.prototype.hasOwnProperty.call(nextState, 'column_parameters')) {
            delete nextState.column_parameters;
          }
          return nextState;
        }

        return {
          ...previous,
          column_parameters: baseParameters,
        };
      });
    },
    [setConfigState]
  );

  return {
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
  };
};
