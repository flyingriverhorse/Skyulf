import { useEffect, type Dispatch, type SetStateAction } from 'react';
import type { PreviewState } from '../nodes/dataset/DataSnapshotSection';
import { arraysAreEqual, inferColumnSuggestions } from '../utils/configParsers';
import { ensureArrayOfString } from '../sharedUtils';

export type UsePreviewColumnTypesArgs = {
  previewState: PreviewState;
  previewSampleRows: Record<string, any>[];
  activeFlagSuffix: string;
  setColumnTypeMap: Dispatch<SetStateAction<Record<string, string>>>;
  setColumnSuggestions: Dispatch<SetStateAction<Record<string, string[]>>>;
};

export const usePreviewColumnTypes = ({
  previewState,
  previewSampleRows,
  activeFlagSuffix,
  setColumnTypeMap,
  setColumnSuggestions,
}: UsePreviewColumnTypesArgs) => {
  useEffect(() => {
    const stats = Array.isArray(previewState.data?.column_stats) ? previewState.data.column_stats : [];
    const status = previewState.status;

    if (!stats.length) {
      if (status === 'loading') {
        return;
      }
      setColumnTypeMap((previous) => (Object.keys(previous).length ? {} : previous));
      setColumnSuggestions((previous) => (Object.keys(previous).length ? {} : previous));
      return;
    }

    const nextTypeMap: Record<string, string> = {};
    const nextSuggestionMap: Record<string, string[]> = {};

    stats.forEach((stat) => {
      if (!stat || !stat.name) {
        return;
      }
      const name = String(stat.name).trim();
      if (!name) {
        return;
      }
      if (activeFlagSuffix && name.endsWith(activeFlagSuffix)) {
        return;
      }
      const dtype = typeof stat.dtype === 'string' ? stat.dtype : null;
      if (dtype) {
        nextTypeMap[name] = dtype;
      }
      const sampledValues = previewSampleRows.map((row) =>
        row && Object.prototype.hasOwnProperty.call(row, name) ? row[name] : undefined,
      );
      const suggestions = inferColumnSuggestions(dtype, sampledValues);
      if (suggestions.length) {
        nextSuggestionMap[name] = suggestions;
      }
    });

    setColumnTypeMap((previous) => {
      const previousEntries = Object.entries(previous);
      const nextEntries = Object.entries(nextTypeMap);
      if (
        previousEntries.length === nextEntries.length &&
        previousEntries.every(
          ([key, value]) => Object.prototype.hasOwnProperty.call(nextTypeMap, key) && nextTypeMap[key] === value,
        )
      ) {
        return previous;
      }
      return nextTypeMap;
    });

    setColumnSuggestions((previous) => {
      const previousKeys = Object.keys(previous);
      const nextKeys = Object.keys(nextSuggestionMap);
      if (
        previousKeys.length === nextKeys.length &&
        previousKeys.every((key) => {
          if (!Object.prototype.hasOwnProperty.call(nextSuggestionMap, key)) {
            return false;
          }
          const existing = previous[key] ?? [];
          const next = nextSuggestionMap[key] ?? [];
          return arraysAreEqual(existing, next);
        })
      ) {
        return previous;
      }
      return nextSuggestionMap;
    });
  }, [
    activeFlagSuffix,
    previewSampleRows,
    previewState.data?.column_stats,
    previewState.status,
    setColumnSuggestions,
    setColumnTypeMap,
  ]);
};

export type UsePreviewAvailableColumnsArgs = {
  previewState: PreviewState;
  activeFlagSuffix: string;
  hasReachableSource: boolean;
  requiresColumnCatalog: boolean;
  nodeColumns: string[];
  selectedColumns: string[];
  setAvailableColumns: Dispatch<SetStateAction<string[]>>;
  setColumnMissingMap: Dispatch<SetStateAction<Record<string, number>>>;
};

export const usePreviewAvailableColumns = ({
  previewState,
  activeFlagSuffix,
  hasReachableSource,
  requiresColumnCatalog,
  nodeColumns,
  selectedColumns,
  setAvailableColumns,
  setColumnMissingMap,
}: UsePreviewAvailableColumnsArgs) => {
  useEffect(() => {
    if (!requiresColumnCatalog) {
      return;
    }

    if (!hasReachableSource) {
      return;
    }

    const nextColumns = new Set<string>();

    const previewColumns = Array.isArray(previewState.data?.columns) ? previewState.data.columns : [];
    previewColumns.forEach((column) => {
      const normalized = typeof column === 'string' ? column.trim() : '';
      if (!normalized) {
        return;
      }
      if (activeFlagSuffix && normalized.endsWith(activeFlagSuffix)) {
        return;
      }
      nextColumns.add(normalized);
    });

    const previewStats = Array.isArray(previewState.data?.column_stats) ? previewState.data.column_stats : [];
    const nextMissingMap: Record<string, number> = {};

    previewStats.forEach((stat) => {
      if (!stat || !stat.name) {
        return;
      }
      const name = String(stat.name).trim();
      if (!name) {
        return;
      }
      if (activeFlagSuffix && name.endsWith(activeFlagSuffix)) {
        return;
      }
      nextColumns.add(name);
      const numeric = Number(stat.missing_percentage);
      if (!Number.isNaN(numeric) && numeric >= 0) {
        nextMissingMap[name] = numeric;
      }
    });

    ensureArrayOfString(nodeColumns).forEach((column) => {
      if (column && !(activeFlagSuffix && column.endsWith(activeFlagSuffix))) {
        nextColumns.add(column);
      }
    });
    selectedColumns.forEach((column) => {
      if (column && !(activeFlagSuffix && column.endsWith(activeFlagSuffix))) {
        nextColumns.add(column);
      }
    });

    if (nextColumns.size === 0) {
      return;
    }

    const orderedColumns = Array.from(nextColumns).sort((a, b) => a.localeCompare(b));
    setAvailableColumns((previous) => (arraysAreEqual(previous, orderedColumns) ? previous : orderedColumns));

    if (Object.keys(nextMissingMap).length > 0) {
      setColumnMissingMap((previous) => ({ ...previous, ...nextMissingMap }));
    }
  }, [
    activeFlagSuffix,
    hasReachableSource,
    nodeColumns,
    previewState.data?.column_stats,
    previewState.data?.columns,
    requiresColumnCatalog,
    selectedColumns,
    setAvailableColumns,
    setColumnMissingMap,
  ]);
};
