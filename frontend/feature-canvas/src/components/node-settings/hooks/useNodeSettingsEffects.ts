import { usePreviewColumnTypes, usePreviewAvailableColumns } from './usePreviewColumnMetadata';

type UseNodeSettingsEffectsArgs = {
  previewState: any;
  previewSampleRows: any;
  activeFlagSuffix: string;
  setColumnTypeMap: any;
  setColumnSuggestions: any;
  hasReachableSource: boolean;
  requiresColumnCatalog: boolean;
  nodeColumns: string[];
  selectedColumns: string[];
  setAvailableColumns: any;
  setColumnMissingMap: any;
};

export const useNodeSettingsEffects = ({
  previewState,
  previewSampleRows,
  activeFlagSuffix,
  setColumnTypeMap,
  setColumnSuggestions,
  hasReachableSource,
  requiresColumnCatalog,
  nodeColumns,
  selectedColumns,
  setAvailableColumns,
  setColumnMissingMap,
}: UseNodeSettingsEffectsArgs) => {
  usePreviewColumnTypes({
    previewState,
    previewSampleRows,
    activeFlagSuffix,
    setColumnTypeMap,
    setColumnSuggestions,
  });

  usePreviewAvailableColumns({
    previewState,
    activeFlagSuffix,
    hasReachableSource,
    requiresColumnCatalog,
    nodeColumns,
    selectedColumns,
    setAvailableColumns,
    setColumnMissingMap,
  });
};
