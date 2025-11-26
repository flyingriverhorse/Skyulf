// Used by NodeSettingsModal for binned distribution nodes to fetch histogram data.
import { useCallback, useEffect, useState } from 'react';
import {
  fetchBinnedDistribution,
  type BinnedDistributionResponse,
} from '../../../api';
import { BINNED_SAMPLE_PRESETS, type BinnedSamplePresetValue } from '../nodes/binning/binningSettings';

import { type CatalogFlagMap } from './useCatalogFlags';

type GraphContext = {
  nodes: any[];
  edges: any[];
} | null;

type UseBinnedDistributionArgs = {
  catalogFlags: CatalogFlagMap;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext: GraphContext;
  targetNodeId: string | null;
  samplePreset: BinnedSamplePresetValue;
};

type UseBinnedDistributionResult = {
  binnedDistributionData: BinnedDistributionResponse | null;
  binnedDistributionError: string | null;
  isFetchingBinnedDistribution: boolean;
  refreshBinnedDistribution: () => void;
};

export const useBinnedDistribution = ({
  catalogFlags,
  sourceId,
  hasReachableSource,
  graphContext,
  targetNodeId,
  samplePreset,
}: UseBinnedDistributionArgs): UseBinnedDistributionResult => {
  const { isBinnedDistributionNode } = catalogFlags;
  const [binnedDistributionData, setBinnedDistributionData] = useState<BinnedDistributionResponse | null>(null);
  const [binnedDistributionError, setBinnedDistributionError] = useState<string | null>(null);
  const [isFetchingBinnedDistribution, setIsFetchingBinnedDistribution] = useState(false);
  const [requestId, setRequestId] = useState(0);

  const refreshBinnedDistribution = useCallback(() => {
    setRequestId((previous) => previous + 1);
  }, []);

  useEffect(() => {
    let isActive = true;

    if (!isBinnedDistributionNode) {
      setBinnedDistributionData(null);
      setBinnedDistributionError(null);
      setIsFetchingBinnedDistribution(false);
      return () => {
        isActive = false;
      };
    }

    if (!sourceId) {
      setBinnedDistributionData(null);
      setBinnedDistributionError('Select a dataset to load binned column distributions.');
      setIsFetchingBinnedDistribution(false);
      return () => {
        isActive = false;
      };
    }

    if (!hasReachableSource) {
      setBinnedDistributionData(null);
      setBinnedDistributionError('Connect this step to an upstream output to load binned column distributions.');
      setIsFetchingBinnedDistribution(false);
      return () => {
        isActive = false;
      };
    }

    setIsFetchingBinnedDistribution(true);
    setBinnedDistributionError(null);

    const selectedPreset =
      BINNED_SAMPLE_PRESETS.find((preset) => preset.value === samplePreset) ?? BINNED_SAMPLE_PRESETS[0];

    const requestedSampleSize: number | 'all' = selectedPreset.value === 'all' ? 'all' : Number(selectedPreset.value);

    fetchBinnedDistribution(sourceId, {
      sampleSize: requestedSampleSize,
      graph: graphContext,
      targetNodeId,
    })
      .then((result: BinnedDistributionResponse | null | undefined) => {
        if (!isActive) {
          return;
        }
        setBinnedDistributionData(result ?? null);
      })
      .catch((error: any) => {
        if (!isActive) {
          return;
        }
        setBinnedDistributionData(null);
        setBinnedDistributionError(error?.message ?? 'Unable to load binned column distributions');
      })
      .finally(() => {
        if (isActive) {
          setIsFetchingBinnedDistribution(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [graphContext, hasReachableSource, isBinnedDistributionNode, requestId, samplePreset, sourceId, targetNodeId]);

  return {
    binnedDistributionData,
    binnedDistributionError,
    isFetchingBinnedDistribution,
    refreshBinnedDistribution,
  };
};
