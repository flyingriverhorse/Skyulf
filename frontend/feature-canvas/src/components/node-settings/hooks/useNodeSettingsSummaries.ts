import { useInsightSummaries } from './useInsightSummaries';
import { useAsyncBusyLabel } from './useAsyncBusyLabel';
import type { CatalogFlagMap } from './useCatalogFlags';

type UseNodeSettingsSummariesArgs = {
  recommendationsGeneratedAt: any;
  scalingData: any;
  binningData: any;
  binnedDistributionData: any;
  outlierData: any;
  isProfileLoading: boolean;
  isPreviewLoading: boolean;
  isFetchingScaling: boolean;
  isFetchingBinning: boolean;
  isFetchingHashEncoding: boolean;
  isFetchingBinnedDistribution: boolean;
  isFetchingRecommendations: boolean;
  catalogFlags: CatalogFlagMap;
};

export const useNodeSettingsSummaries = ({
  recommendationsGeneratedAt,
  scalingData,
  binningData,
  binnedDistributionData,
  outlierData,
  isProfileLoading,
  isPreviewLoading,
  isFetchingScaling,
  isFetchingBinning,
  isFetchingHashEncoding,
  isFetchingBinnedDistribution,
  isFetchingRecommendations,
  catalogFlags,
}: UseNodeSettingsSummariesArgs) => {
  const insightSummaries = useInsightSummaries({
    recommendationsGeneratedAt,
    scalingData,
    binningData,
    binnedDistributionData,
    outlierData,
  });

  const asyncBusyLabel = useAsyncBusyLabel({
    isProfileLoading,
    isPreviewLoading,
    isFetchingScaling,
    isFetchingBinning,
    isFetchingHashEncoding,
    isFetchingBinnedDistribution,
    isFetchingRecommendations,
    catalogFlags,
  });

  return {
    insightSummaries,
    asyncBusyLabel,
  };
};
