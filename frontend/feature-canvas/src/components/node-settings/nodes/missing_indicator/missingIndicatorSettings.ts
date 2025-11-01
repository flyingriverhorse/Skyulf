import { ensureArrayOfString } from '../../sharedUtils';

const DEFAULT_SUFFIX = '_was_missing';

export type MissingIndicatorInsightRow = {
  column: string;
  missingPercentage: number | null;
  flagColumn: string;
  conflicts: {
    alreadyExists: boolean;
    duplicateFlag: boolean;
  };
};

export type MissingIndicatorInsights = {
  rows: MissingIndicatorInsightRow[];
  flaggedColumnsInDataset: string[];
  conflictCount: number;
};

type BuildMissingIndicatorInsightsInput = {
  selectedColumns: string[] | null | undefined;
  availableColumns: string[];
  columnMissingMap: Record<string, number>;
  suffix: string;
};

export const resolveMissingIndicatorSuffix = (
  primary?: unknown,
  fallback?: unknown,
): string => {
  const normalizedPrimary = typeof primary === 'string' ? primary.trim() : '';
  if (normalizedPrimary) {
    return normalizedPrimary;
  }
  const normalizedFallback = typeof fallback === 'string' ? fallback.trim() : '';
  if (normalizedFallback) {
    return normalizedFallback;
  }
  return DEFAULT_SUFFIX;
};

export const isFlaggedColumnName = (column: string, suffix: string): boolean => {
  const normalizedColumn = typeof column === 'string' ? column.trim() : '';
  const normalizedSuffix = typeof suffix === 'string' ? suffix : '';
  if (!normalizedColumn || !normalizedSuffix) {
    return false;
  }
  return normalizedColumn.endsWith(normalizedSuffix);
};

export const buildMissingIndicatorInsights = ({
  selectedColumns,
  availableColumns,
  columnMissingMap,
  suffix,
}: BuildMissingIndicatorInsightsInput): MissingIndicatorInsights => {
  const normalizedSuffix = typeof suffix === 'string' ? suffix : '';
  const datasetColumns = ensureArrayOfString(availableColumns);
  const datasetColumnSet = new Set(datasetColumns);
  const uniqueSelectedColumns = Array.from(new Set(ensureArrayOfString(selectedColumns)));
  const rows: MissingIndicatorInsightRow[] = [];

  uniqueSelectedColumns.forEach((column) => {
    const flagColumn = `${column}${normalizedSuffix}`;
    const rawMissingValue = columnMissingMap[column];
    const missingPercentage =
      typeof rawMissingValue === 'number' && Number.isFinite(rawMissingValue) ? Math.max(0, rawMissingValue) : null;

    rows.push({
      column,
      missingPercentage,
      flagColumn,
      conflicts: {
        alreadyExists: datasetColumnSet.has(flagColumn),
        duplicateFlag: false,
      },
    });
  });

  if (rows.length > 0) {
    const flagCounts = new Map<string, number>();
    rows.forEach((row) => {
      flagCounts.set(row.flagColumn, (flagCounts.get(row.flagColumn) ?? 0) + 1);
    });
    rows.forEach((row) => {
      row.conflicts.duplicateFlag = (flagCounts.get(row.flagColumn) ?? 0) > 1;
    });
  }

  rows.sort((a, b) => {
    const aMissing = a.missingPercentage;
    const bMissing = b.missingPercentage;
    const aHas = typeof aMissing === 'number';
    const bHas = typeof bMissing === 'number';
    if (aHas && bHas && aMissing !== bMissing) {
      return (bMissing as number) - (aMissing as number);
    }
    if (aHas !== bHas) {
      return aHas ? -1 : 1;
    }
    return a.column.localeCompare(b.column);
  });

  const flaggedColumnsInDataset = normalizedSuffix
    ? Array.from(new Set(datasetColumns.filter((name) => isFlaggedColumnName(name, normalizedSuffix))))
    : [];

  const conflictCount = rows.reduce((total, row) => {
    if (row.conflicts.alreadyExists || row.conflicts.duplicateFlag) {
      return total + 1;
    }
    return total;
  }, 0);

  return {
    rows,
    flaggedColumnsInDataset,
    conflictCount,
  };
};
