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
    trimWhitespace: {
      columnSummary: trimWhitespaceColumnSummary,
      sampleMap: trimWhitespaceSampleMap,
      modeDetails: trimWhitespaceModeDetails,
    },
    removeSpecial: {
      columnSummary: removeSpecialColumnSummary,
      sampleMap: removeSpecialSampleMap,
      modeDetails: removeSpecialModeDetails,
      selectedMode: removeSpecialMode,
    },
    regexCleanup: {
      columnSummary: regexCleanupColumnSummary,
      sampleMap: regexCleanupSampleMap,
      modeDetails: regexCleanupModeDetails,
      selectedMode: regexCleanupMode,
      replacementValue: regexCleanupReplacementValue,
    },
    normalizeCase: {
      columnSummary: normalizeCaseColumnSummary,
      sampleMap: normalizeCaseSampleMap,
      modeDetails: normalizeCaseModeDetails,
      selectedMode: normalizeCaseMode,
    },
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
    trimWhitespaceColumnSummary,
    trimWhitespaceSampleMap,
    trimWhitespaceModeDetails,
    removeSpecialColumnSummary,
    removeSpecialSampleMap,
    removeSpecialModeDetails,
    removeSpecialMode,
    regexCleanupColumnSummary,
    regexCleanupSampleMap,
    regexCleanupModeDetails,
    regexCleanupMode,
    regexCleanupReplacementValue,
    normalizeCaseColumnSummary,
    normalizeCaseSampleMap,
    normalizeCaseModeDetails,
    normalizeCaseMode,
    replaceInvalidMode,
    replaceInvalidModeDetails,
    replaceInvalidSampleMap,
    replaceInvalidColumnSummary,
    replaceInvalidMinValue,
    replaceInvalidMaxValue,
    standardizeDatesMode,
    dateStrategies,
    standardizeDatesSampleMap,
    standardizeDatesColumnSummary,
    dateColumnOptions,
  };
};
