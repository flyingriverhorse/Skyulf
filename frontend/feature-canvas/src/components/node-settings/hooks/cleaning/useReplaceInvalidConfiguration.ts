import { useMemo } from 'react';
import {
  buildInvalidValueColumnSummary,
  buildInvalidValueSampleMap,
  EMPTY_INVALID_VALUE_COLUMN_SUMMARY,
  getInvalidValueModeDetails,
  resolveInvalidValueMode,
  type InvalidValueColumnSummary,
  type InvalidValueMode,
  type InvalidValueModeDetails,
  type InvalidValueSampleMap,
} from '../../nodes/replace_invalid_values/replaceInvalidValuesSettings';
import { ensureArrayOfString } from '../../sharedUtils';
import type { CatalogFlagMap } from '../core/useCatalogFlags';

export type ReplaceInvalidHookParams = {
  catalogFlags: CatalogFlagMap;
  configState: Record<string, any>;
  nodeConfig?: Record<string, any> | null;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  previewSampleRows: Array<Record<string, any>>;
};

export type ReplaceInvalidHookResult = {
  selectedColumns: string[];
  selectedMode: InvalidValueMode;
  modeDetails: InvalidValueModeDetails;
  sampleMap: InvalidValueSampleMap;
  columnSummary: InvalidValueColumnSummary;
  minValue: number | null;
  maxValue: number | null;
};

const EMPTY_SAMPLE_MAP: InvalidValueSampleMap = {};

export const useReplaceInvalidConfiguration = ({
  catalogFlags,
  configState,
  nodeConfig,
  availableColumns,
  columnTypeMap,
  previewSampleRows,
}: ReplaceInvalidHookParams): ReplaceInvalidHookResult => {
  const { isReplaceInvalidValuesNode } = catalogFlags;

  const selectedMode = useMemo<InvalidValueMode>(
    () =>
      resolveInvalidValueMode(
        isReplaceInvalidValuesNode ? configState?.mode : undefined,
        isReplaceInvalidValuesNode ? nodeConfig?.mode : undefined,
      ),
    [configState?.mode, isReplaceInvalidValuesNode, nodeConfig?.mode],
  );

  const selectedColumns = useMemo<string[]>(() => {
    if (!isReplaceInvalidValuesNode) {
      return [] as string[];
    }
    return ensureArrayOfString(configState?.columns);
  }, [configState?.columns, isReplaceInvalidValuesNode]);

  const minValue = useMemo(() => {
    if (!isReplaceInvalidValuesNode) {
      return null;
    }
    const configValue = configState?.min_value ?? nodeConfig?.min_value;
    const numeric = Number(configValue);
    return Number.isFinite(numeric) ? numeric : null;
  }, [configState?.min_value, isReplaceInvalidValuesNode, nodeConfig?.min_value]);

  const maxValue = useMemo(() => {
    if (!isReplaceInvalidValuesNode) {
      return null;
    }
    const configValue = configState?.max_value ?? nodeConfig?.max_value;
    const numeric = Number(configValue);
    return Number.isFinite(numeric) ? numeric : null;
  }, [configState?.max_value, isReplaceInvalidValuesNode, nodeConfig?.max_value]);

  const sampleMap = useMemo<InvalidValueSampleMap>(() => {
    if (!isReplaceInvalidValuesNode) {
      return EMPTY_SAMPLE_MAP;
    }
    return buildInvalidValueSampleMap(previewSampleRows);
  }, [isReplaceInvalidValuesNode, previewSampleRows]);

  const columnSummary = useMemo<InvalidValueColumnSummary>(() => {
    if (!isReplaceInvalidValuesNode) {
      return EMPTY_INVALID_VALUE_COLUMN_SUMMARY;
    }
    return buildInvalidValueColumnSummary({
      selectedColumns,
      availableColumns,
      columnTypeMap,
      sampleMap,
    });
  }, [availableColumns, columnTypeMap, isReplaceInvalidValuesNode, sampleMap, selectedColumns]);

  const modeDetails = useMemo<InvalidValueModeDetails>(
    () => getInvalidValueModeDetails(selectedMode),
    [selectedMode],
  );

  return {
    selectedColumns,
    selectedMode,
    modeDetails,
    sampleMap,
    columnSummary,
    minValue,
    maxValue,
  };
};
