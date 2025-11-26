// Used by NodeSettingsModal to keep scaling and binning configs in sync with column exclusions.
import { useEffect } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { OutlierMethodName, ScalingMethodName } from '../../../api';
import { ensureArrayOfString } from '../sharedUtils';
import { SCALING_METHOD_ORDER } from '../nodes/scaling/scalingSettings';
import { OUTLIER_METHOD_ORDER } from '../nodes/outlier/outlierSettings';
import type { CatalogFlagMap } from './useCatalogFlags';

type UsePruneColumnSelectionsParams = {
  catalogFlags: CatalogFlagMap;
  scalingExcludedColumns: Set<string>;
  binningExcludedColumns: Set<string>;
  outlierExcludedColumns: Set<string>;
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
};

export const usePruneColumnSelections = ({
  catalogFlags,
  scalingExcludedColumns,
  binningExcludedColumns,
  outlierExcludedColumns,
  setConfigState,
}: UsePruneColumnSelectionsParams): void => {
  const { isScalingNode, isBinningNode, isOutlierNode } = catalogFlags;

  useEffect(() => {
    if (!isScalingNode || scalingExcludedColumns.size === 0) {
      return;
    }

    setConfigState((previous) => {
      if (!previous || typeof previous !== 'object') {
        return previous;
      }

      const currentColumns = ensureArrayOfString(previous.columns);
      
      if (!currentColumns.length) {
        // If there are no columns selected yet, don't prune column_methods
        // This allows manual configuration before columns are added
        return previous;
      }

      const filteredColumns = currentColumns.filter((column) => !scalingExcludedColumns.has(column));
      if (filteredColumns.length === currentColumns.length) {
        // No columns were actually excluded, no need to update
        return previous;
      }

      if (!filteredColumns.length) {
        return previous;
      }

      const nextState: Record<string, any> = {
        ...previous,
        columns: filteredColumns,
      };

      if (
        previous.column_methods &&
        typeof previous.column_methods === 'object' &&
        !Array.isArray(previous.column_methods)
      ) {
        const nextMethods: Record<string, ScalingMethodName> = {};
        let hadExcludedMethod = false;
        Object.entries(previous.column_methods as Record<string, any>).forEach(([key, value]) => {
          const normalized = String(key ?? '').trim();
          if (!normalized) {
            return;
          }
          if (scalingExcludedColumns.has(normalized)) {
            hadExcludedMethod = true;
            return;
          }
          const methodKey = typeof value === 'string' ? (value.trim() as ScalingMethodName) : null;
          if (methodKey && SCALING_METHOD_ORDER.includes(methodKey)) {
            nextMethods[normalized] = methodKey;
          }
        });
        
        // Only update column_methods if we actually removed something
        if (hadExcludedMethod) {
          if (Object.keys(nextMethods).length) {
            nextState.column_methods = nextMethods;
          } else {
            // Keep an empty object instead of deleting - allows manual configuration
            nextState.column_methods = {};
          }
        } else {
          // No changes needed, preserve existing column_methods
          nextState.column_methods = previous.column_methods;
        }
      }

      return nextState;
    });
  }, [isScalingNode, scalingExcludedColumns, setConfigState]);

  useEffect(() => {
    if (!isBinningNode || binningExcludedColumns.size === 0) {
      return;
    }

    setConfigState((previous) => {
      if (!previous || typeof previous !== 'object') {
        return previous;
      }

      const currentColumns = ensureArrayOfString(previous.columns);
      if (!currentColumns.length) {
        return previous;
      }

      const filteredColumns = currentColumns.filter((column) => !binningExcludedColumns.has(column));
      if (filteredColumns.length === currentColumns.length) {
        return previous;
      }

      const removedColumns = new Set(currentColumns.filter((column) => binningExcludedColumns.has(column)));
      const nextState: Record<string, any> = {
        ...previous,
        columns: filteredColumns,
      };

      if (
        previous.custom_bins &&
        typeof previous.custom_bins === 'object' &&
        !Array.isArray(previous.custom_bins)
      ) {
        const nextBins: Record<string, number[]> = {};
        Object.entries(previous.custom_bins).forEach(([key, value]) => {
          if (!removedColumns.has(key) && Array.isArray(value) && value.length) {
            nextBins[key] = [...value];
          }
        });
        if (Object.keys(nextBins).length) {
          nextState.custom_bins = nextBins;
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'custom_bins')) {
          delete nextState.custom_bins;
        }
      }

      if (
        previous.custom_labels &&
        typeof previous.custom_labels === 'object' &&
        !Array.isArray(previous.custom_labels)
      ) {
        const nextLabels: Record<string, string[]> = {};
        Object.entries(previous.custom_labels).forEach(([key, value]) => {
          if (!removedColumns.has(key) && Array.isArray(value) && value.length) {
            nextLabels[key] = value.map((entry) => String(entry));
          }
        });
        if (Object.keys(nextLabels).length) {
          nextState.custom_labels = nextLabels;
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'custom_labels')) {
          delete nextState.custom_labels;
        }
      }

      return nextState;
    });
  }, [binningExcludedColumns, isBinningNode, setConfigState]);

  useEffect(() => {
    if (!isOutlierNode || outlierExcludedColumns.size === 0) {
      return;
    }

    setConfigState((previous) => {
      if (!previous || typeof previous !== 'object') {
        return previous;
      }

      const currentColumns = ensureArrayOfString(previous.columns);
      const filteredColumns = currentColumns.filter((column) => !outlierExcludedColumns.has(column));
      const columnsChanged = filteredColumns.length !== currentColumns.length;

      const skippedColumns = ensureArrayOfString(previous.skipped_columns);
      const filteredSkipped = skippedColumns.filter((column) => !outlierExcludedColumns.has(column));
      const skippedChanged = filteredSkipped.length !== skippedColumns.length;

      const ensureState = (state: Record<string, any> | null): Record<string, any> => state ?? { ...previous };
      let nextState: Record<string, any> | null = null;

      if (columnsChanged) {
        nextState = ensureState(nextState);
        nextState.columns = filteredColumns.sort((a, b) => a.localeCompare(b));
      }

      if (skippedChanged) {
        nextState = ensureState(nextState);
        if (filteredSkipped.length) {
          nextState.skipped_columns = filteredSkipped.sort((a, b) => a.localeCompare(b));
        } else if (Object.prototype.hasOwnProperty.call(nextState, 'skipped_columns')) {
          delete nextState.skipped_columns;
        }
      }

      let methodsChanged = false;
      if (
        previous.column_methods &&
        typeof previous.column_methods === 'object' &&
        !Array.isArray(previous.column_methods)
      ) {
        const nextMethods: Record<string, OutlierMethodName> = {};
        Object.entries(previous.column_methods as Record<string, any>).forEach(([key, value]) => {
          const normalized = String(key ?? '').trim();
          if (!normalized || outlierExcludedColumns.has(normalized)) {
            methodsChanged = true;
            return;
          }
          const methodKey = typeof value === 'string' ? (value.trim() as OutlierMethodName) : null;
          if (methodKey && OUTLIER_METHOD_ORDER.includes(methodKey)) {
            nextMethods[normalized] = methodKey;
          } else {
            methodsChanged = true;
          }
        });

        if (methodsChanged) {
          nextState = ensureState(nextState);
          if (Object.keys(nextMethods).length) {
            nextState.column_methods = nextMethods;
          } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_methods')) {
            delete nextState.column_methods;
          }
        }
      }

      let parametersChanged = false;
      if (
        previous.column_parameters &&
        typeof previous.column_parameters === 'object' &&
        !Array.isArray(previous.column_parameters)
      ) {
        const nextParameters: Record<string, Record<string, number>> = {};
        Object.entries(previous.column_parameters as Record<string, any>).forEach(([key, value]) => {
          const normalized = String(key ?? '').trim();
          if (!normalized || outlierExcludedColumns.has(normalized)) {
            parametersChanged = true;
            return;
          }
          if (!value || typeof value !== 'object' || Array.isArray(value)) {
            parametersChanged = true;
            return;
          }
          nextParameters[normalized] = { ...(value as Record<string, number>) };
        });

        if (parametersChanged) {
          nextState = ensureState(nextState);
          if (Object.keys(nextParameters).length) {
            nextState.column_parameters = nextParameters;
          } else if (Object.prototype.hasOwnProperty.call(nextState, 'column_parameters')) {
            delete nextState.column_parameters;
          }
        }
      }

      if (!nextState) {
        return previous;
      }

      return nextState;
    });
  }, [isOutlierNode, outlierExcludedColumns, setConfigState]);
};
