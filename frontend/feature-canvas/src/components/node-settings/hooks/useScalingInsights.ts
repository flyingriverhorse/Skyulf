// Used by NodeSettingsModal for scaling nodes to surface column-level recommendations.
import { useCallback, useEffect, useState } from 'react';
import {
  fetchScalingRecommendations,
  type ScalingRecommendationsResponse,
} from '../../../api';

type GraphContext = {
  nodes: any[];
  edges: any[];
} | null;

type UseScalingInsightsArgs = {
  isScalingNode: boolean;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext: GraphContext;
  targetNodeId: string | null;
};

type UseScalingInsightsResult = {
  scalingData: ScalingRecommendationsResponse | null;
  scalingError: string | null;
  isFetchingScaling: boolean;
  refreshScaling: () => void;
};

export const useScalingInsights = ({
  isScalingNode,
  sourceId,
  hasReachableSource,
  graphContext,
  targetNodeId,
}: UseScalingInsightsArgs): UseScalingInsightsResult => {
  const [scalingData, setScalingData] = useState<ScalingRecommendationsResponse | null>(null);
  const [scalingError, setScalingError] = useState<string | null>(null);
  const [isFetchingScaling, setIsFetchingScaling] = useState(false);
  const [requestId, setRequestId] = useState(0);

  const refreshScaling = useCallback(() => {
    setRequestId((previous) => previous + 1);
  }, []);

  useEffect(() => {
    let isActive = true;

    if (!isScalingNode) {
      setScalingData(null);
      setScalingError(null);
      setIsFetchingScaling(false);
      return () => {
        isActive = false;
      };
    }

    if (!sourceId) {
      setScalingData(null);
      setScalingError('Select a dataset to load scaling insights.');
      setIsFetchingScaling(false);
      return () => {
        isActive = false;
      };
    }

    if (!hasReachableSource) {
      setScalingData(null);
      setScalingError('Connect this step to an upstream output to load scaling insights.');
      setIsFetchingScaling(false);
      return () => {
        isActive = false;
      };
    }

    setIsFetchingScaling(true);
    setScalingError(null);

    fetchScalingRecommendations(sourceId, {
      graph: graphContext,
      targetNodeId,
    })
      .then((result: ScalingRecommendationsResponse | null | undefined) => {
        if (!isActive) {
          return;
        }
        setScalingData(result ?? null);
      })
      .catch((error: any) => {
        if (!isActive) {
          return;
        }
        setScalingData(null);
        setScalingError(error?.message ?? 'Unable to load scaling insights');
      })
      .finally(() => {
        if (isActive) {
          setIsFetchingScaling(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [graphContext, hasReachableSource, isScalingNode, requestId, sourceId, targetNodeId]);

  return {
    scalingData,
    scalingError,
    isFetchingScaling,
    refreshScaling,
  };
};
