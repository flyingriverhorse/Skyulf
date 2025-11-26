import { useSkewnessConfiguration } from './useSkewnessConfiguration';
import { useImputationConfiguration } from './useImputationConfiguration';
import { useThresholdRecommendations } from './useThresholdRecommendations';
import type { CatalogFlagMap } from './useCatalogFlags';

type UseNodeSettingsConfigurationArgs = {
  catalogFlags: CatalogFlagMap;
  skewnessData: any;
  skewnessTransformations: any;
  availableColumns: string[];
  previewColumns: any;
  columnTypeMap: any;
  updateSkewnessTransformations: any;
  imputerStrategies: any;
  columnMissingMap: any;
  previewColumnStats: any;
  nodeColumns: string[];
  imputerMissingFilter: any;
  suggestedThreshold: any;
  thresholdParameterName: string | null;
  configState: any;
  handleParameterChange: any;
};

export const useNodeSettingsConfiguration = ({
  catalogFlags,
  skewnessData,
  skewnessTransformations,
  availableColumns,
  previewColumns,
  columnTypeMap,
  updateSkewnessTransformations,
  imputerStrategies,
  columnMissingMap,
  previewColumnStats,
  nodeColumns,
  imputerMissingFilter,
  suggestedThreshold,
  thresholdParameterName,
  configState,
  handleParameterChange,
}: UseNodeSettingsConfigurationArgs) => {
  const skewness = useSkewnessConfiguration({
    catalogFlags,
    skewnessData,
    skewnessTransformations,
    availableColumns,
    previewColumns,
    columnTypeMap,
    updateSkewnessTransformations,
  });

  const imputationConfiguration = useImputationConfiguration({
    catalogFlags,
    imputerStrategies,
    availableColumns,
    columnMissingMap,
    previewColumns,
    previewColumnStats,
    nodeColumns,
    imputerMissingFilter,
  });

  const thresholdRecommendations = useThresholdRecommendations({
    suggestedThreshold,
    thresholdParameterName,
    configState,
    handleParameterChange,
  });

  return {
    skewness,
    imputationConfiguration,
    thresholdRecommendations,
  };
};
