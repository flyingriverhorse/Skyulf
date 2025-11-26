// Used by NodeSettingsModal for binning nodes to surface backend recommendations.
import { useCallback, useEffect, useState } from 'react';
import {
  fetchBinningRecommendations,
  type BinningRecommendationsResponse,
} from '../../../../api';

import { type CatalogFlagMap } from '../core/useCatalogFlags';

type GraphContext = {
  nodes: any[];
  edges: any[];
} | null;

type UseBinningInsightsArgs = {
  catalogFlags: CatalogFlagMap;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext: GraphContext;
  targetNodeId: string | null;
};

type UseBinningInsightsResult = {
  binningData: BinningRecommendationsResponse | null;
  binningError: string | null;
  isFetchingBinning: boolean;
  refreshBinning: () => void;
};

export const useBinningInsights = ({
  catalogFlags,
  sourceId,
  hasReachableSource,
  graphContext,
  targetNodeId,
}: UseBinningInsightsArgs): UseBinningInsightsResult => {
  const { isBinningNode } = catalogFlags;
  const [binningData, setBinningData] = useState<BinningRecommendationsResponse | null>(null);
  const [binningError, setBinningError] = useState<string | null>(null);
  const [isFetchingBinning, setIsFetchingBinning] = useState(false);
  const [requestId, setRequestId] = useState(0);

  const refreshBinning = useCallback(() => {
    setRequestId((previous) => previous + 1);
  }, []);

  useEffect(() => {
    let isActive = true;

    if (!isBinningNode) {
      setBinningData(null);
      setBinningError(null);
      setIsFetchingBinning(false);
      return () => {
        isActive = false;
      };
    }

    if (!sourceId) {
      setBinningData(null);
      setBinningError('Select a dataset to load binning insights.');
      setIsFetchingBinning(false);
      return () => {
        isActive = false;
      };
    }

    if (!hasReachableSource) {
      setBinningData(null);
      setBinningError('Connect this step to an upstream output to analyse numeric columns.');
      setIsFetchingBinning(false);
      return () => {
        isActive = false;
      };
    }

    setIsFetchingBinning(true);
    setBinningError(null);

    fetchBinningRecommendations(sourceId, {
      graph: graphContext,
      targetNodeId,
    })
      .then((result: BinningRecommendationsResponse | null | undefined) => {
        if (!isActive) {
          return;
        }
        setBinningData(result ?? null);
      })
      .catch((error: any) => {
        if (!isActive) {
          return;
        }
        setBinningData(null);
        setBinningError(error?.message ?? 'Unable to load binning insights');
      })
      .finally(() => {
        if (isActive) {
          setIsFetchingBinning(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [graphContext, hasReachableSource, isBinningNode, requestId, sourceId, targetNodeId]);

  return {
    binningData,
    binningError,
    isFetchingBinning,
    refreshBinning,
  };
};
