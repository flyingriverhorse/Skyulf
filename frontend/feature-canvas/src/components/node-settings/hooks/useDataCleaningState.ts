import { useAliasConfiguration } from './useAliasConfiguration';
import { useTextCleanupConfiguration } from './useTextCleanupConfiguration';
import { useReplaceInvalidConfiguration } from './useReplaceInvalidConfiguration';
import { useStandardizeDatesConfiguration } from './useStandardizeDatesConfiguration';

type UseDataCleaningStateProps = {
  isReplaceAliasesNode: boolean;
  isTrimWhitespaceNode: boolean;
  isRemoveSpecialCharsNode: boolean;
  isRegexCleanupNode: boolean;
  isNormalizeTextCaseNode: boolean;
  isReplaceInvalidValuesNode: boolean;
  isStandardizeDatesNode: boolean;
  configState: any;
  nodeConfig: any;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  previewSampleRows: any[];
};

export const useDataCleaningState = ({
  isReplaceAliasesNode,
  isTrimWhitespaceNode,
  isRemoveSpecialCharsNode,
  isRegexCleanupNode,
  isNormalizeTextCaseNode,
  isReplaceInvalidValuesNode,
  isStandardizeDatesNode,
  configState,
  nodeConfig,
  availableColumns,
  columnTypeMap,
  previewSampleRows,
}: UseDataCleaningStateProps) => {
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
    isReplaceAliasesNode,
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
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
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
    isReplaceInvalidValuesNode,
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
    isStandardizeDatesNode,
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
