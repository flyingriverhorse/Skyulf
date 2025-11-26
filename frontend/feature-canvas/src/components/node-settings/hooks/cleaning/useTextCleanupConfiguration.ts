import { useMemo } from 'react';
import {
  buildTrimColumnSummary,
  buildTrimSampleMap,
  EMPTY_TRIM_COLUMN_SUMMARY,
  getTrimModeDetails,
  resolveTrimMode,
  type TrimColumnSummary,
  type TrimModeDetails,
  type TrimSampleMap,
} from '../../nodes/trim_white_space/trimWhitespaceSettings';
import {
  buildSpecialColumnSummary,
  buildSpecialSampleMap,
  EMPTY_SPECIAL_COLUMN_SUMMARY,
  getRemoveSpecialModeDetails,
  resolveRemoveSpecialMode,
  type RemoveSpecialMode,
  type RemoveSpecialModeDetails,
  type SpecialColumnSummary,
  type SpecialSampleMap,
} from '../../nodes/remove_special_char/removeSpecialCharactersSettings';
import {
  buildRegexColumnSummary,
  buildRegexSampleMap,
  EMPTY_REGEX_COLUMN_SUMMARY,
  getRegexModeDetails,
  resolveRegexMode,
  type RegexColumnSummary,
  type RegexMode,
  type RegexModeDetails,
  type RegexSampleMap,
} from '../../nodes/regex_node/regexCleanupSettings';
import {
  buildCaseColumnSummary,
  buildCaseSampleMap,
  EMPTY_CASE_COLUMN_SUMMARY,
  getCaseModeDetails,
  resolveCaseMode,
  type CaseColumnSummary,
  type CaseMode,
  type CaseModeDetails,
  type CaseSampleMap,
} from '../../nodes/normalize_text/normalizeTextCaseSettings';
import { ensureArrayOfString } from '../../sharedUtils';
import type { CatalogFlagMap } from '../core/useCatalogFlags';

export type TextCleanupHookParams = {
  catalogFlags: CatalogFlagMap;
  configState: Record<string, any>;
  nodeConfig?: Record<string, any> | null;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  previewSampleRows: Array<Record<string, any>>;
};

export type TextCleanupHookResult = {
  trimWhitespace: {
    columnSummary: TrimColumnSummary;
    sampleMap: TrimSampleMap;
    modeDetails: TrimModeDetails;
  };
  removeSpecial: {
    columnSummary: SpecialColumnSummary;
    sampleMap: SpecialSampleMap;
    modeDetails: RemoveSpecialModeDetails;
    selectedMode: RemoveSpecialMode;
  };
  regexCleanup: {
    columnSummary: RegexColumnSummary;
    sampleMap: RegexSampleMap;
    modeDetails: RegexModeDetails;
    selectedMode: RegexMode;
    replacementValue: string | null;
  };
  normalizeCase: {
    columnSummary: CaseColumnSummary;
    sampleMap: CaseSampleMap;
    modeDetails: CaseModeDetails;
    selectedMode: CaseMode;
  };
};

export const useTextCleanupConfiguration = ({
  catalogFlags,
  configState,
  nodeConfig,
  availableColumns,
  columnTypeMap,
  previewSampleRows,
}: TextCleanupHookParams): TextCleanupHookResult => {
  const {
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
  } = catalogFlags;

  const trimWhitespaceMode = useMemo(
    () =>
      resolveTrimMode(
        isTrimWhitespaceNode ? configState?.mode : undefined,
        isTrimWhitespaceNode ? nodeConfig?.mode : undefined,
      ),
    [configState?.mode, isTrimWhitespaceNode, nodeConfig?.mode],
  );

  const trimWhitespaceColumns = useMemo<string[]>(() => {
    if (!isTrimWhitespaceNode) {
      return [] as string[];
    }
    return ensureArrayOfString(configState?.columns);
  }, [configState?.columns, isTrimWhitespaceNode]);

  const trimWhitespaceSampleMap = useMemo<TrimSampleMap>(() => {
    if (!isTrimWhitespaceNode) {
      return {} as TrimSampleMap;
    }
    return buildTrimSampleMap(previewSampleRows);
  }, [isTrimWhitespaceNode, previewSampleRows]);

  const trimWhitespaceColumnSummary = useMemo<TrimColumnSummary>(() => {
    if (!isTrimWhitespaceNode) {
      return EMPTY_TRIM_COLUMN_SUMMARY;
    }
    return buildTrimColumnSummary({
      selectedColumns: trimWhitespaceColumns,
      availableColumns,
      columnTypeMap,
      sampleMap: trimWhitespaceSampleMap,
    });
  }, [availableColumns, columnTypeMap, isTrimWhitespaceNode, trimWhitespaceColumns, trimWhitespaceSampleMap]);

  const trimWhitespaceModeDetails = useMemo<TrimModeDetails>(
    () => getTrimModeDetails(trimWhitespaceMode),
    [trimWhitespaceMode],
  );

  const removeSpecialMode = useMemo(
    () =>
      resolveRemoveSpecialMode(
        isRemoveSpecialCharsNode ? configState?.mode : undefined,
        isRemoveSpecialCharsNode ? nodeConfig?.mode : undefined,
      ),
    [configState?.mode, isRemoveSpecialCharsNode, nodeConfig?.mode],
  );

  const removeSpecialColumns = useMemo<string[]>(() => {
    if (!isRemoveSpecialCharsNode) {
      return [] as string[];
    }
    return ensureArrayOfString(configState?.columns);
  }, [configState?.columns, isRemoveSpecialCharsNode]);

  const removeSpecialSampleMap = useMemo<SpecialSampleMap>(() => {
    if (!isRemoveSpecialCharsNode) {
      return {} as SpecialSampleMap;
    }
    return buildSpecialSampleMap(previewSampleRows);
  }, [isRemoveSpecialCharsNode, previewSampleRows]);

  const removeSpecialColumnSummary = useMemo<SpecialColumnSummary>(() => {
    if (!isRemoveSpecialCharsNode) {
      return EMPTY_SPECIAL_COLUMN_SUMMARY;
    }
    return buildSpecialColumnSummary({
      selectedColumns: removeSpecialColumns,
      availableColumns,
      columnTypeMap,
      sampleMap: removeSpecialSampleMap,
    });
  }, [availableColumns, columnTypeMap, isRemoveSpecialCharsNode, removeSpecialColumns, removeSpecialSampleMap]);

  const removeSpecialModeDetails = useMemo<RemoveSpecialModeDetails>(
    () => getRemoveSpecialModeDetails(removeSpecialMode),
    [removeSpecialMode],
  );

  const regexCleanupMode = useMemo<RegexMode>(
    () =>
      resolveRegexMode(
        isRegexCleanupNode ? configState?.mode : undefined,
        isRegexCleanupNode ? nodeConfig?.mode : undefined,
      ),
    [configState?.mode, isRegexCleanupNode, nodeConfig?.mode],
  );

  const regexCleanupColumns = useMemo<string[]>(() => {
    if (!isRegexCleanupNode) {
      return [] as string[];
    }
    return ensureArrayOfString(configState?.columns);
  }, [configState?.columns, isRegexCleanupNode]);

  const regexCleanupSampleMap = useMemo<RegexSampleMap>(() => {
    if (!isRegexCleanupNode) {
      return {} as RegexSampleMap;
    }
    return buildRegexSampleMap(previewSampleRows);
  }, [isRegexCleanupNode, previewSampleRows]);

  const regexCleanupColumnSummary = useMemo<RegexColumnSummary>(() => {
    if (!isRegexCleanupNode) {
      return EMPTY_REGEX_COLUMN_SUMMARY;
    }
    return buildRegexColumnSummary({
      selectedColumns: regexCleanupColumns,
      availableColumns,
      columnTypeMap,
      sampleMap: regexCleanupSampleMap,
    });
  }, [availableColumns, columnTypeMap, isRegexCleanupNode, regexCleanupColumns, regexCleanupSampleMap]);

  const regexCleanupModeDetails = useMemo<RegexModeDetails>(
    () => getRegexModeDetails(regexCleanupMode),
    [regexCleanupMode],
  );

  const regexCleanupReplacementValue = useMemo(() => {
    if (!isRegexCleanupNode) {
      return null;
    }
    const configValue = configState?.replacement ?? nodeConfig?.replacement ?? null;
    return typeof configValue === 'string' ? configValue : null;
  }, [configState?.replacement, isRegexCleanupNode, nodeConfig?.replacement]);

  const normalizeCaseMode = useMemo<CaseMode>(
    () =>
      resolveCaseMode(
        isNormalizeTextCaseNode ? configState?.mode : undefined,
        isNormalizeTextCaseNode ? nodeConfig?.mode : undefined,
      ),
    [configState?.mode, isNormalizeTextCaseNode, nodeConfig?.mode],
  );

  const normalizeCaseColumns = useMemo<string[]>(() => {
    if (!isNormalizeTextCaseNode) {
      return [] as string[];
    }
    return ensureArrayOfString(configState?.columns);
  }, [configState?.columns, isNormalizeTextCaseNode]);

  const normalizeCaseSampleMap = useMemo<CaseSampleMap>(() => {
    if (!isNormalizeTextCaseNode) {
      return {} as CaseSampleMap;
    }
    return buildCaseSampleMap(previewSampleRows);
  }, [isNormalizeTextCaseNode, previewSampleRows]);

  const normalizeCaseColumnSummary = useMemo<CaseColumnSummary>(() => {
    if (!isNormalizeTextCaseNode) {
      return EMPTY_CASE_COLUMN_SUMMARY;
    }
    return buildCaseColumnSummary({
      selectedColumns: normalizeCaseColumns,
      availableColumns,
      columnTypeMap,
      sampleMap: normalizeCaseSampleMap,
      mode: normalizeCaseMode,
    });
  }, [
    availableColumns,
    columnTypeMap,
    isNormalizeTextCaseNode,
    normalizeCaseColumns,
    normalizeCaseMode,
    normalizeCaseSampleMap,
  ]);

  const normalizeCaseModeDetails = useMemo<CaseModeDetails>(
    () => getCaseModeDetails(normalizeCaseMode),
    [normalizeCaseMode],
  );

  return {
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
  };
};
