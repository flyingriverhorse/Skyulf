import { useMemo } from 'react';
import type {
  FeatureMathNodeSignal,
  PolynomialFeaturesNodeSignal,
  FeatureSelectionNodeSignal,
} from '../../../api';
import type { PreviewState } from '../nodes/dataset/DataSnapshotSection';

type UsePreviewSignalsArgs = {
  previewState: PreviewState;
  nodeId: string;
  isFeatureMathNode: boolean;
  isPolynomialFeaturesNode: boolean;
  isFeatureSelectionNode: boolean;
};

type UsePreviewSignalsResult = {
  featureMathSignals: FeatureMathNodeSignal[];
  polynomialSignal: PolynomialFeaturesNodeSignal | null;
  featureSelectionSignal: FeatureSelectionNodeSignal | null;
};

export const usePreviewSignals = ({
  previewState,
  nodeId,
  isFeatureMathNode,
  isPolynomialFeaturesNode,
  isFeatureSelectionNode,
}: UsePreviewSignalsArgs): UsePreviewSignalsResult => {
  const featureMathSignals = useMemo<FeatureMathNodeSignal[]>(() => {
    if (!isFeatureMathNode) {
      return [];
    }
    const rawSignals = previewState.data?.signals?.feature_math;
    return Array.isArray(rawSignals) ? rawSignals : [];
  }, [isFeatureMathNode, previewState.data?.signals?.feature_math]);

  const polynomialSignal = useMemo<PolynomialFeaturesNodeSignal | null>(() => {
    if (!isPolynomialFeaturesNode) {
      return null;
    }
    const rawSignals = previewState.data?.signals?.polynomial_features;
    if (!Array.isArray(rawSignals) || rawSignals.length === 0) {
      return null;
    }
    const matching = nodeId
      ? rawSignals.find((entry) => entry && typeof entry.node_id === 'string' && entry.node_id === nodeId)
      : null;
    return matching ?? rawSignals[0] ?? null;
  }, [isPolynomialFeaturesNode, nodeId, previewState.data?.signals?.polynomial_features]);

  const featureSelectionSignal = useMemo<FeatureSelectionNodeSignal | null>(() => {
    if (!isFeatureSelectionNode) {
      return null;
    }
    const rawSignals = previewState.data?.signals?.feature_selection;
    if (!Array.isArray(rawSignals) || rawSignals.length === 0) {
      return null;
    }
    const matching = nodeId
      ? rawSignals.find((entry) => entry && typeof entry.node_id === 'string' && entry.node_id === nodeId)
      : null;
    return matching ?? rawSignals[0] ?? null;
  }, [isFeatureSelectionNode, nodeId, previewState.data?.signals?.feature_selection]);

  return {
    featureMathSignals,
    polynomialSignal,
    featureSelectionSignal,
  };
};
