import { useCallback, useMemo } from 'react';

interface UseThresholdRecommendationsArgs {
  suggestedThreshold: number | string | null;
  thresholdParameterName: string | null;
  configState: Record<string, any> | null | undefined;
  handleParameterChange: (parameter: string, nextValue: any) => void;
}

interface UseThresholdRecommendationsResult {
  normalizedSuggestedThreshold: number | null;
  thresholdMatchesSuggestion: boolean;
  canApplySuggestedThreshold: boolean;
  handleApplySuggestedThreshold: () => void;
}

export const useThresholdRecommendations = ({
  suggestedThreshold,
  thresholdParameterName,
  configState,
  handleParameterChange,
}: UseThresholdRecommendationsArgs): UseThresholdRecommendationsResult => {
  const normalizedSuggestedThreshold = useMemo(() => {
    if (suggestedThreshold === null || Number.isNaN(Number(suggestedThreshold))) {
      return null;
    }
    const numeric = Number(suggestedThreshold);
    return Math.round(numeric * 10) / 10;
  }, [suggestedThreshold]);

  const thresholdMatchesSuggestion = useMemo(() => {
    if (thresholdParameterName === null || normalizedSuggestedThreshold === null) {
      return false;
    }
    const currentValue = Number(configState?.[thresholdParameterName]);
    return !Number.isNaN(currentValue) && currentValue === normalizedSuggestedThreshold;
  }, [configState, normalizedSuggestedThreshold, thresholdParameterName]);

  const canApplySuggestedThreshold =
    normalizedSuggestedThreshold !== null && !thresholdMatchesSuggestion;

  const handleApplySuggestedThreshold = useCallback(() => {
    if (normalizedSuggestedThreshold === null || !thresholdParameterName) {
      return;
    }
    handleParameterChange(thresholdParameterName, normalizedSuggestedThreshold);
  }, [handleParameterChange, normalizedSuggestedThreshold, thresholdParameterName]);

  return {
    normalizedSuggestedThreshold,
    thresholdMatchesSuggestion,
    canApplySuggestedThreshold,
    handleApplySuggestedThreshold,
  };
};
