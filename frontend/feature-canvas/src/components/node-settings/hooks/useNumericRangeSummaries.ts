// Used by NodeSettingsModal when configuring binning.
import { useMemo } from 'react';

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
  isBinningNode: boolean;
  numericExcludedColumns: Set<string>;
  selectedColumns: string[];
  previewSampleRows: Array<Record<string, any>>;
  availableColumns: string[];
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
  isBinningNode,
  numericExcludedColumns,
  selectedColumns,
  previewSampleRows,
  availableColumns,
}: UseNumericRangeSummariesParams): UseNumericRangeSummariesResult => {
  const binningExcludedColumns = useMemo(() => {
    if (!isBinningNode) {
      return new Set<string>();
    }
    return numericExcludedColumns;
  }, [isBinningNode, numericExcludedColumns]);

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
