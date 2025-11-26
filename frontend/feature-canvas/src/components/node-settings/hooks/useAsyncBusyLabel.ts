import { useMemo } from 'react';

interface UseAsyncBusyLabelOptions {
  isProfileLoading: boolean;
  isPreviewLoading: boolean;
  isFetchingScaling: boolean;
  isFetchingBinning: boolean;
  isFetchingHashEncoding: boolean;
  isFetchingBinnedDistribution: boolean;
  isFetchingRecommendations: boolean;
  isPreviewNode: boolean;
}

interface UseAsyncBusyLabelResult {
  hasActiveAsyncWork: boolean;
  busyLabel: string | null;
  footerBusyLabel: string | null;
}

/**
 * Computes a consistent busy label string for the header/footer and exposes a boolean
 * that downstream buttons can use for disabled/loading states.
 */
export const useAsyncBusyLabel = ({
  isProfileLoading,
  isPreviewLoading,
  isFetchingScaling,
  isFetchingBinning,
  isFetchingHashEncoding,
  isFetchingBinnedDistribution,
  isFetchingRecommendations,
  isPreviewNode,
}: UseAsyncBusyLabelOptions): UseAsyncBusyLabelResult => {
  const hasActiveAsyncWork = useMemo(
    () =>
      isProfileLoading ||
      isPreviewLoading ||
      isFetchingScaling ||
      isFetchingBinning ||
      isFetchingHashEncoding ||
      isFetchingBinnedDistribution ||
      isFetchingRecommendations,
    [
      isFetchingBinnedDistribution,
      isFetchingBinning,
      isFetchingHashEncoding,
      isFetchingScaling,
      isFetchingRecommendations,
      isPreviewLoading,
      isProfileLoading,
    ],
  );

  const busyLabel = useMemo(() => {
    if (isProfileLoading) {
      return 'Generating dataset profile…';
    }
    if (isFetchingScaling) {
      return 'Loading scaling insights…';
    }
    if (isFetchingBinning) {
      return 'Loading binning insights…';
    }
    if (isFetchingHashEncoding) {
      return 'Loading hash encoding insights…';
    }
    if (isFetchingBinnedDistribution) {
      return 'Computing binned distributions…';
    }
    if (isFetchingRecommendations) {
      return 'Fetching column recommendations…';
    }
    if (isPreviewLoading) {
      return isPreviewNode ? 'Loading dataset preview…' : 'Analyzing node outputs…';
    }
    return null;
  }, [
    isFetchingBinnedDistribution,
    isFetchingBinning,
    isFetchingHashEncoding,
    isFetchingRecommendations,
    isFetchingScaling,
    isPreviewLoading,
    isPreviewNode,
    isProfileLoading,
  ]);

  const footerBusyLabel = busyLabel ?? (hasActiveAsyncWork ? 'Processing…' : null);

  return {
    hasActiveAsyncWork,
    busyLabel,
    footerBusyLabel,
  };
};
