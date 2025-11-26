import { useMemo } from 'react';
import type {
  FeatureMathNodeSignal,
  PolynomialFeaturesNodeSignal,
  FeatureSelectionNodeSignal,
  OutlierNodeSignal,
  TransformerAuditNodeSignal,
} from '../../../api';
import type { PreviewState } from '../nodes/dataset/DataSnapshotSection';
import type { CatalogFlagMap } from './useCatalogFlags';

type UsePreviewSignalsArgs = {
  previewState: PreviewState;
  nodeId: string;
  catalogFlags: CatalogFlagMap;
};

type UsePreviewSignalsResult = {
  featureMathSignals: FeatureMathNodeSignal[];
  polynomialSignal: PolynomialFeaturesNodeSignal | null;
  featureSelectionSignal: FeatureSelectionNodeSignal | null;
  outlierPreviewSignal: OutlierNodeSignal | null;
  transformerAuditSignal: TransformerAuditNodeSignal | null;
};

export const usePreviewSignals = ({
  previewState,
  nodeId,
  catalogFlags,
}: UsePreviewSignalsArgs): UsePreviewSignalsResult => {
  const {
    isFeatureMathNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
    isOutlierNode,
    isTransformerAuditNode,
  } = catalogFlags;

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

  const outlierPreviewSignal = useMemo<OutlierNodeSignal | null>(() => {
    if (!isOutlierNode) {
      return null;
    }
    const signals = previewState.data?.signals?.outlier_removal;
    if (!Array.isArray(signals) || !signals.length) {
      return null;
    }
    if (!nodeId) {
      return (signals[signals.length - 1] as OutlierNodeSignal | undefined) ?? null;
    }
    const match = signals.find((signal: any) => signal && signal.node_id === nodeId);
    return (match as OutlierNodeSignal | undefined) ?? ((signals[signals.length - 1] as OutlierNodeSignal | undefined) ?? null);
  }, [isOutlierNode, nodeId, previewState.data?.signals?.outlier_removal]);

  const transformerAuditSignal = useMemo<TransformerAuditNodeSignal | null>(() => {
    if (!isTransformerAuditNode) {
      return null;
    }
    const signals = previewState.data?.signals?.transformer_audit;
    if (!Array.isArray(signals) || !signals.length) {
      return null;
    }
    if (!nodeId) {
      return (signals[signals.length - 1] as TransformerAuditNodeSignal | undefined) ?? null;
    }
    const match = signals.find((signal: any) => signal && signal.node_id === nodeId);
    return (match as TransformerAuditNodeSignal | undefined) ?? ((signals[signals.length - 1] as TransformerAuditNodeSignal | undefined) ?? null);
  }, [isTransformerAuditNode, nodeId, previewState.data?.signals?.transformer_audit]);

  return {
    featureMathSignals,
    polynomialSignal,
    featureSelectionSignal,
    outlierPreviewSignal,
    transformerAuditSignal,
  };
};
