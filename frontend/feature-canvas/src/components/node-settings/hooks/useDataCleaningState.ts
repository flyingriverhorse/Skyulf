import { useAliasConfiguration } from './useAliasConfiguration';
import { useTextCleanupConfiguration } from './useTextCleanupConfiguration';
import { useReplaceInvalidConfiguration } from './useReplaceInvalidConfiguration';
import { useStandardizeDatesConfiguration } from './useStandardizeDatesConfiguration';
import type { CatalogFlagMap } from './useCatalogFlags';

type UseDataCleaningStateProps = {
  catalogFlags: CatalogFlagMap;
  configState: any;
  nodeConfig: any;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  previewSampleRows: any[];
};

export const useDataCleaningState = ({
  catalogFlags,
  configState,
  nodeConfig,
  availableColumns,
  columnTypeMap,
  previewSampleRows,
}: UseDataCleaningStateProps) => {
  const {
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isReplaceInvalidValuesNode,
    isStandardizeDatesNode,
  } = catalogFlags;

  const {
    aliasAutoDetectMeta,
    aliasStrategies,
    aliasStrategyCount,
    aliasSelectedColumns,
    aliasAutoDetectEnabled,
    replaceAliasesCustomPairsValue,
    aliasColumnSummary,
    aliasCustomPairSummary,
    aliasSampleMap,
    aliasColumnOptions,
  } = useAliasConfiguration({
    catalogFlags,
    configState,
    nodeConfig,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const {
    trimWhitespace,
    removeSpecial,
    regexCleanup,
    normalizeCase,
  } = useTextCleanupConfiguration({
    catalogFlags,
    configState,
    nodeConfig,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const {
    selectedMode: replaceInvalidMode,
    modeDetails: replaceInvalidModeDetails,
    sampleMap: replaceInvalidSampleMap,
    columnSummary: replaceInvalidColumnSummary,
    minValue: replaceInvalidMinValue,
    maxValue: replaceInvalidMaxValue,
  } = useReplaceInvalidConfiguration({
    catalogFlags,
    configState,
    nodeConfig,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  const {
    selectedMode: standardizeDatesMode,
    strategies: dateStrategies,
    sampleMap: standardizeDatesSampleMap,
    columnSummary: standardizeDatesColumnSummary,
    columnOptions: dateColumnOptions,
  } = useStandardizeDatesConfiguration({
    catalogFlags,
    configState,
    nodeConfig,
    availableColumns,
    columnTypeMap,
    previewSampleRows,
  });

  return {
    alias: {
      autoDetectMeta: aliasAutoDetectMeta,
      strategies: aliasStrategies,
      strategyCount: aliasStrategyCount,
      selectedColumns: aliasSelectedColumns,
      autoDetectEnabled: aliasAutoDetectEnabled,
      customPairsValue: replaceAliasesCustomPairsValue,
      columnSummary: aliasColumnSummary,
      customPairSummary: aliasCustomPairSummary,
      sampleMap: aliasSampleMap,
      columnOptions: aliasColumnOptions,
    },
    trimWhitespace,
    removeSpecial,
    regexCleanup,
    normalizeCase,
    replaceInvalid: {
      mode: replaceInvalidMode,
      modeDetails: replaceInvalidModeDetails,
      sampleMap: replaceInvalidSampleMap,
      columnSummary: replaceInvalidColumnSummary,
      minValue: replaceInvalidMinValue,
      maxValue: replaceInvalidMaxValue,
    },
    standardizeDates: {
      mode: standardizeDatesMode,
      strategies: dateStrategies,
      sampleMap: standardizeDatesSampleMap,
      columnSummary: standardizeDatesColumnSummary,
      columnOptions: dateColumnOptions,
    },
  };
};
