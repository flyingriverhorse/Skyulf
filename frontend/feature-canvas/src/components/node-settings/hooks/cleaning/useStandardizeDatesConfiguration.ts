import { useMemo } from 'react';
import {
  buildDateColumnSummary,
  buildDateSampleMap,
  EMPTY_DATE_COLUMN_SUMMARY,
  normalizeDateFormatStrategies,
  resolveDateMode,
  type DateColumnOption,
  type DateColumnSummary,
  type DateFormatStrategyConfig,
  type DateMode,
  type DateSampleMap,
} from '../../nodes/standardize_date/standardizeDateSettings';
import { ensureArrayOfString } from '../../sharedUtils';
import type { CatalogFlagMap } from '../core/useCatalogFlags';

export type StandardizeDatesHookParams = {
  catalogFlags: CatalogFlagMap;
  configState: Record<string, any>;
  nodeConfig?: Record<string, any> | null;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  previewSampleRows: Array<Record<string, any>>;
};

export type StandardizeDatesHookResult = {
  selectedMode: DateMode;
  strategies: DateFormatStrategyConfig[];
  sampleMap: DateSampleMap;
  columnSummary: DateColumnSummary;
  columnOptions: DateColumnOption[];
};

const EMPTY_SAMPLE_MAP: DateSampleMap = {};

export const useStandardizeDatesConfiguration = ({
  catalogFlags,
  configState,
  nodeConfig,
  availableColumns,
  columnTypeMap,
  previewSampleRows,
}: StandardizeDatesHookParams): StandardizeDatesHookResult => {
  const { isStandardizeDatesNode } = catalogFlags;

  const selectedMode = useMemo<DateMode>(
    () =>
      resolveDateMode(
        isStandardizeDatesNode ? configState?.mode : undefined,
        isStandardizeDatesNode ? nodeConfig?.mode : undefined,
      ),
    [configState?.mode, isStandardizeDatesNode, nodeConfig?.mode],
  );

  const legacyColumns = useMemo<string[]>(() => {
    if (!isStandardizeDatesNode) {
      return [] as string[];
    }
    return ensureArrayOfString(configState?.columns ?? nodeConfig?.columns);
  }, [configState?.columns, isStandardizeDatesNode, nodeConfig?.columns]);

  const strategies = useMemo<DateFormatStrategyConfig[]>(() => {
    if (!isStandardizeDatesNode) {
      return [];
    }
    const fallback = {
      columns: legacyColumns,
      mode: selectedMode,
      autoDetect: legacyColumns.length === 0,
    };
    return normalizeDateFormatStrategies(
      configState?.format_strategies ?? nodeConfig?.format_strategies,
      fallback,
    );
  }, [configState?.format_strategies, isStandardizeDatesNode, legacyColumns, nodeConfig?.format_strategies, selectedMode]);

  const selectedColumns = useMemo<string[]>(() => {
    if (!isStandardizeDatesNode) {
      return [] as string[];
    }
    const set = new Set<string>();
    strategies.forEach((strategy) => {
      strategy.columns.forEach((column) => {
        const normalized = column.trim();
        if (normalized) {
          set.add(normalized);
        }
      });
    });
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [isStandardizeDatesNode, strategies]);

  const hasAutoDetectStrategy = useMemo(
    () => strategies.some((strategy) => strategy.autoDetect),
    [strategies],
  );

  const sampleMap = useMemo<DateSampleMap>(() => {
    if (!isStandardizeDatesNode) {
      return EMPTY_SAMPLE_MAP;
    }
    return buildDateSampleMap(previewSampleRows);
  }, [isStandardizeDatesNode, previewSampleRows]);

  const columnSummary = useMemo<DateColumnSummary>(() => {
    if (!isStandardizeDatesNode) {
      return EMPTY_DATE_COLUMN_SUMMARY;
    }
    const base = buildDateColumnSummary({
      selectedColumns,
      availableColumns,
      columnTypeMap,
      sampleMap,
    });
    if (hasAutoDetectStrategy) {
      return {
        ...base,
        autoDetectionActive: true,
      };
    }
    return base;
  }, [availableColumns, columnTypeMap, hasAutoDetectStrategy, isStandardizeDatesNode, sampleMap, selectedColumns]);

  const columnOptions = useMemo<DateColumnOption[]>(() => {
    if (!isStandardizeDatesNode) {
      return [];
    }
    const recommendedSet = new Set(columnSummary.recommendedColumns);
    const candidateSet = new Set(columnSummary.sampleCandidates);
    return availableColumns.map((column) => {
      const dtype = columnTypeMap[column] ?? null;
      const rawSamples = sampleMap[column] ?? [];
      return {
        name: column,
        dtype,
        samples: rawSamples.slice(0, 3),
        isRecommended: recommendedSet.has(column),
        isSampleCandidate: candidateSet.has(column),
      };
    });
  }, [availableColumns, columnSummary, columnTypeMap, isStandardizeDatesNode, sampleMap]);

  return {
    selectedMode,
    strategies,
    sampleMap,
    columnSummary,
    columnOptions,
  };
};
