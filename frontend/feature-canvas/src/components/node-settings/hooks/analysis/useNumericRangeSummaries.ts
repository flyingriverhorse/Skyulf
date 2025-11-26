// Used by NodeSettingsModal when configuring binning.
import { useMemo } from 'react';
import type { CatalogFlagMap } from '../core/useCatalogFlags';

export type NumericColumnPreview = {
  min: number | null;
  max: number | null;
  distinct: number | null;
};

export type ManualRangeFallback = {
  min: number | null;
  max: number | null;
};

type UseNumericRangeSummariesParams = {
  catalogFlags: CatalogFlagMap;
  numericExcludedColumns: Set<string>;
  selectedColumns: string[];
  previewSampleRows: Array<Record<string, any>>;
  availableColumns: string[];
  recommendedColumns?: Iterable<string> | null;
};

export type UseNumericRangeSummariesResult = {
  binningExcludedColumns: Set<string>;
  binningColumnPreviewMap: Record<string, NumericColumnPreview>;
  manualBoundColumns: string[];
  manualRangeFallbackMap: Record<string, ManualRangeFallback>;
};

const parseNumericCandidate = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return numeric;
    }
  }
  return null;
};

export const useNumericRangeSummaries = ({
  catalogFlags,
  numericExcludedColumns,
  selectedColumns,
  previewSampleRows,
  availableColumns,
  recommendedColumns,
}: UseNumericRangeSummariesParams): UseNumericRangeSummariesResult => {
  const { isBinningNode } = catalogFlags;
  const recommendedColumnSet = useMemo(() => {
    if (!recommendedColumns) {
      return null;
    }
    const normalized = new Set<string>();
    for (const entry of recommendedColumns) {
      const value = typeof entry === 'string' ? entry.trim() : '';
      if (!value) {
        continue;
      }
      normalized.add(value);
    }
    return normalized.size > 0 ? normalized : null;
  }, [recommendedColumns]);

  const selectedColumnSet = useMemo(() => {
    if (!Array.isArray(selectedColumns) || !selectedColumns.length) {
      return null;
    }
    const normalized = new Set<string>();
    selectedColumns.forEach((column) => {
      const value = typeof column === 'string' ? column.trim() : '';
      if (!value) {
        return;
      }
      normalized.add(value);
    });
    return normalized.size > 0 ? normalized : null;
  }, [selectedColumns]);

  const binningExcludedColumns = useMemo(() => {
    if (!isBinningNode) {
      return new Set<string>();
    }
    if (!numericExcludedColumns.size) {
      return numericExcludedColumns;
    }
    if (!recommendedColumnSet && !selectedColumnSet) {
      return numericExcludedColumns;
    }
    const next = new Set<string>();
    numericExcludedColumns.forEach((column) => {
      if (recommendedColumnSet?.has(column)) {
        return;
      }
      if (selectedColumnSet?.has(column)) {
        return;
      }
      next.add(column);
    });
    return next;
  }, [isBinningNode, numericExcludedColumns, recommendedColumnSet, selectedColumnSet]);

  const binningColumnPreviewMap = useMemo(() => {
    if (!isBinningNode || !previewSampleRows.length) {
      return {} as Record<string, NumericColumnPreview>;
    }

    const rangeMap: Record<string, NumericColumnPreview> = {};

    availableColumns.forEach((column) => {
      if (binningExcludedColumns.has(column)) {
        return;
      }

      let minValue: number | null = null;
      let maxValue: number | null = null;
      const distinctValues = new Set<number>();

      previewSampleRows.forEach((row) => {
        if (!row || !Object.prototype.hasOwnProperty.call(row, column)) {
          return;
        }

        const numeric = parseNumericCandidate(row[column]);
        if (numeric === null) {
          return;
        }

        if (minValue === null || numeric < minValue) {
          minValue = numeric;
        }
        if (maxValue === null || numeric > maxValue) {
          maxValue = numeric;
        }

        if (distinctValues.size <= 1000) {
          distinctValues.add(numeric);
        }
      });

      if (minValue !== null || maxValue !== null) {
        const distinct = distinctValues.size > 0 ? distinctValues.size : null;
        rangeMap[column] = {
          min: minValue,
          max: maxValue,
          distinct,
        };
      }
    });

    return rangeMap;
  }, [availableColumns, binningExcludedColumns, isBinningNode, previewSampleRows]);

  return {
    binningExcludedColumns,
    binningColumnPreviewMap,
    manualBoundColumns: [],
    manualRangeFallbackMap: {},
  };
};
