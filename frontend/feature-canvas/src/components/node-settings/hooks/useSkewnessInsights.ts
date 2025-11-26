// Used by NodeSettingsModal for skewness transformation and distribution nodes.
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchSkewnessRecommendations,
  type SkewnessRecommendationsResponse,
  type SkewnessTransformationSelection,
} from '../../../api';

import { type CatalogFlagMap } from './useCatalogFlags';

type GraphContext = {
  nodes: any[];
  edges: any[];
} | null;

type UseSkewnessInsightsArgs = {
  catalogFlags: CatalogFlagMap;
  sourceId?: string | null;
  graphContext: GraphContext;
  targetNodeId: string | null;
  transformations: SkewnessTransformationSelection[];
};

type UseSkewnessInsightsResult = {
  skewnessData: SkewnessRecommendationsResponse | null;
  skewnessError: string | null;
  isFetchingSkewness: boolean;
  refreshSkewness: () => void;
};

export const useSkewnessInsights = ({
  catalogFlags,
  sourceId,
  graphContext,
  targetNodeId,
  transformations,
}: UseSkewnessInsightsArgs): UseSkewnessInsightsResult => {
  const { isSkewnessNode, isSkewnessDistributionNode } = catalogFlags;
  const shouldLoad = isSkewnessNode || isSkewnessDistributionNode;
  const [skewnessData, setSkewnessData] = useState<SkewnessRecommendationsResponse | null>(null);
  const [skewnessError, setSkewnessError] = useState<string | null>(null);
  const [isFetchingSkewness, setIsFetchingSkewness] = useState(false);
  const [requestId, setRequestId] = useState(0);
  const debounceTimerRef = useRef<number | null>(null);

  const refreshSkewness = useCallback(() => {
    setRequestId((previous) => previous + 1);
  }, []);

  useEffect(() => {
    let isActive = true;

    if (!shouldLoad) {
      setSkewnessData(null);
      setSkewnessError(null);
      setIsFetchingSkewness(false);
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
        debounceTimerRef.current = null;
      }
      return () => {
        isActive = false;
      };
    }

    if (!sourceId) {
      setSkewnessData(null);
      setSkewnessError('Select a dataset to load skewness insights.');
      setIsFetchingSkewness(false);
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
        debounceTimerRef.current = null;
      }
      return () => {
        isActive = false;
      };
    }

    // Clear any existing debounce timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Debounce transformation changes by 300ms to avoid excessive re-fetches
    debounceTimerRef.current = setTimeout(() => {
      if (!isActive) {
        return;
      }

      setIsFetchingSkewness(true);
      setSkewnessError(null);

      let transformationsPayload: SkewnessTransformationSelection[] | undefined;
      if (transformations.length > 0) {
        transformationsPayload = transformations.map((entry) => ({
          column: entry.column,
          method: entry.method,
        }));
      }

      fetchSkewnessRecommendations(sourceId, {
        transformations: transformationsPayload,
        graph: graphContext,
        targetNodeId,
      })
        .then((result: SkewnessRecommendationsResponse | null | undefined) => {
          if (!isActive) {
            return;
          }
          setSkewnessData(result ?? null);
        })
        .catch((error: any) => {
          if (!isActive) {
            return;
          }
          setSkewnessData(null);
          setSkewnessError(error?.message ?? 'Unable to load skewness insights');
        })
        .finally(() => {
          if (isActive) {
            setIsFetchingSkewness(false);
          }
        });
    }, 300);

    return () => {
      isActive = false;
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [graphContext, requestId, shouldLoad, sourceId, targetNodeId, transformations]);

  return {
    skewnessData,
    skewnessError,
    isFetchingSkewness,
    refreshSkewness,
  };
};
