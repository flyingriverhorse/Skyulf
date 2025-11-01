// Used by NodeSettingsModal to prepare replace-aliases specific state.
import { useMemo } from 'react';
import {
  buildAliasColumnSummary,
  buildAliasSampleMap,
  normalizeAliasStrategies,
  parseCustomAliasPairs,
  resolveAliasMode,
  type AliasColumnOption,
  type AliasColumnSummary,
  type AliasCustomPairSummary,
  type AliasSampleMap,
  type AliasStrategyConfig,
} from '../nodes/replace_aliases/replaceAliasesSettings';
import { ensureArrayOfString } from '../sharedUtils';
import { normalizeConfigBoolean, pickAutoDetectValue } from '../utils/configParsers';

type AliasAutoDetectMeta = {
  enabled: boolean;
  explicit: boolean;
};

type UseAliasConfigurationParams = {
  isReplaceAliasesNode: boolean;
  configState: Record<string, any>;
  nodeConfig?: Record<string, any> | null;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  previewSampleRows: Array<Record<string, any>>;
};

type UseAliasConfigurationResult = {
  aliasAutoDetectMeta: AliasAutoDetectMeta;
  aliasStrategies: AliasStrategyConfig[];
  aliasStrategyCount: number;
  aliasSelectedColumns: string[];
  aliasAutoDetectEnabled: boolean;
  replaceAliasesCustomPairsValue: unknown;
  aliasColumnSummary: AliasColumnSummary;
  aliasCustomPairSummary: AliasCustomPairSummary;
  aliasSampleMap: AliasSampleMap;
  aliasColumnOptions: AliasColumnOption[];
};

const DEFAULT_META: AliasAutoDetectMeta = { enabled: false, explicit: false };

const EMPTY_COLUMN_SUMMARY: AliasColumnSummary = {
  selectedColumns: [],
  nonTextSelected: [],
  autoDetectionActive: false,
  recommendedColumns: [],
  textColumnCount: 0,
  textColumns: [],
};

const EMPTY_CUSTOM_PAIR_SUMMARY: AliasCustomPairSummary = {
  totalPairs: 0,
  previewPairs: [],
  previewOverflow: 0,
  duplicates: [],
  duplicateOverflow: 0,
  invalidEntries: [],
  invalidOverflow: 0,
};

const EMPTY_SAMPLE_MAP: AliasSampleMap = {};

const EMPTY_OPTIONS: AliasColumnOption[] = [];

export const useAliasConfiguration = ({
  isReplaceAliasesNode,
  configState,
  nodeConfig,
  availableColumns,
  columnTypeMap,
  previewSampleRows,
}: UseAliasConfigurationParams): UseAliasConfigurationResult => {
  const aliasAutoDetectMeta = useMemo<AliasAutoDetectMeta>(() => {
    if (!isReplaceAliasesNode) {
      return DEFAULT_META;
    }

    const localValue = pickAutoDetectValue(configState as Record<string, unknown>);
    if (localValue !== undefined) {
      const normalized = normalizeConfigBoolean(localValue);
      return { enabled: Boolean(normalized), explicit: true };
    }

    if (nodeConfig && typeof nodeConfig === 'object') {
      const nodeValue = pickAutoDetectValue(nodeConfig as Record<string, unknown>);
      if (nodeValue !== undefined) {
        const normalized = normalizeConfigBoolean(nodeValue);
        return { enabled: Boolean(normalized), explicit: true };
      }
    }

    return DEFAULT_META;
  }, [configState, isReplaceAliasesNode, nodeConfig]);

  const aliasStrategies = useMemo<AliasStrategyConfig[]>(() => {
    if (!isReplaceAliasesNode) {
      return [];
    }

    const fallbackColumns = ensureArrayOfString(configState?.columns ?? nodeConfig?.columns);
    const fallbackMode = resolveAliasMode(
      isReplaceAliasesNode ? configState?.mode : undefined,
      isReplaceAliasesNode ? nodeConfig?.mode : undefined,
    );

    const fallbackAutoDetect = (() => {
      const localValue = pickAutoDetectValue(configState as Record<string, unknown>);
      if (localValue !== undefined) {
        const normalized = normalizeConfigBoolean(localValue);
        if (normalized !== null) {
          return normalized;
        }
      }
      if (nodeConfig && typeof nodeConfig === 'object') {
        const nodeValue = pickAutoDetectValue(nodeConfig as Record<string, unknown>);
        if (nodeValue !== undefined) {
          const normalized = normalizeConfigBoolean(nodeValue);
          if (normalized !== null) {
            return normalized;
          }
        }
      }
      return fallbackColumns.length === 0;
    })();

    const rawStrategies =
      (configState?.alias_strategies ?? configState?.strategies) ??
      (nodeConfig?.alias_strategies ?? nodeConfig?.strategies);

    return normalizeAliasStrategies(rawStrategies, {
      mode: fallbackMode,
      columns: fallbackColumns,
      autoDetect: fallbackAutoDetect,
    });
  }, [configState, isReplaceAliasesNode, nodeConfig]);

  const aliasSelectedColumns = useMemo<string[]>(() => {
    if (!isReplaceAliasesNode) {
      return [];
    }
    if (aliasStrategies.length) {
      return Array.from(
        new Set(
          aliasStrategies.flatMap((strategy) =>
            strategy.columns.map((column) => column.trim()).filter(Boolean),
          ),
        ),
      ).sort((a, b) => a.localeCompare(b));
    }
    return ensureArrayOfString(configState?.columns);
  }, [aliasStrategies, configState?.columns, isReplaceAliasesNode]);

  const aliasAutoDetectEnabled = useMemo(() => {
    if (!isReplaceAliasesNode) {
      return false;
    }
    if (aliasStrategies.length) {
      return aliasStrategies.some((strategy) => strategy.autoDetect);
    }
    if (aliasAutoDetectMeta.explicit) {
      return aliasAutoDetectMeta.enabled;
    }
    return aliasSelectedColumns.length === 0;
  }, [aliasAutoDetectMeta.enabled, aliasAutoDetectMeta.explicit, aliasSelectedColumns.length, aliasStrategies, isReplaceAliasesNode]);

  const replaceAliasesCustomPairsValue = useMemo(() => {
    if (!isReplaceAliasesNode) {
      return undefined;
    }
    return configState?.custom_pairs;
  }, [configState?.custom_pairs, isReplaceAliasesNode]);

  const aliasSampleMap = useMemo<AliasSampleMap>(() => {
    if (!isReplaceAliasesNode) {
      return EMPTY_SAMPLE_MAP;
    }
    return buildAliasSampleMap(previewSampleRows);
  }, [isReplaceAliasesNode, previewSampleRows]);

  const aliasColumnSummary = useMemo<AliasColumnSummary>(() => {
    if (!isReplaceAliasesNode) {
      return EMPTY_COLUMN_SUMMARY;
    }
    return buildAliasColumnSummary({
      selectedColumns: aliasSelectedColumns,
      availableColumns,
      columnTypeMap,
      autoDetectEnabled: aliasAutoDetectEnabled,
    });
  }, [aliasAutoDetectEnabled, aliasSelectedColumns, availableColumns, columnTypeMap, isReplaceAliasesNode]);

  const aliasCustomPairSummary = useMemo<AliasCustomPairSummary>(() => {
    if (!isReplaceAliasesNode) {
      return EMPTY_CUSTOM_PAIR_SUMMARY;
    }
    return parseCustomAliasPairs(replaceAliasesCustomPairsValue);
  }, [isReplaceAliasesNode, replaceAliasesCustomPairsValue]);

  const aliasColumnOptions = useMemo<AliasColumnOption[]>(() => {
    if (!isReplaceAliasesNode) {
      return EMPTY_OPTIONS;
    }
    const recommendedSet = new Set(aliasColumnSummary.recommendedColumns);
    const textSet = new Set(aliasColumnSummary.textColumns);
    return availableColumns.map((column) => {
      const dtype = columnTypeMap[column] ?? null;
      const samples = aliasSampleMap[column] ?? [];
      return {
        name: column,
        dtype,
        samples: samples.slice(0, 3),
        isRecommended: recommendedSet.has(column),
        isTextLike: textSet.has(column),
      } as AliasColumnOption;
    });
  }, [aliasColumnSummary.recommendedColumns, aliasColumnSummary.textColumns, aliasSampleMap, availableColumns, columnTypeMap, isReplaceAliasesNode]);

  return {
    aliasAutoDetectMeta,
    aliasStrategies,
    aliasStrategyCount: aliasStrategies.length,
    aliasSelectedColumns,
    aliasAutoDetectEnabled,
    replaceAliasesCustomPairsValue,
    aliasColumnSummary,
    aliasCustomPairSummary,
    aliasSampleMap,
    aliasColumnOptions,
  };
};
