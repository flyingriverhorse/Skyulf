import { useCallback, useEffect, useMemo, type Dispatch, type SetStateAction } from 'react';
import type { Node } from 'react-flow-renderer';
import type { PipelinePreviewSchema } from '../../../../api';
import { usePipelinePreview } from './usePipelinePreview';
import type { CatalogFlagMap } from '../core/useCatalogFlags';

export type GraphSnapshot = {
  nodes: any[];
  edges: any[];
} | null;

type UseNodePreviewArgs = {
  graphSnapshot: GraphSnapshot;
  graphNodeCount: number;
  hasReachableSource: boolean;
  sourceId?: string | null;
  node: Node;
  nodeId: string;
  previewSignature: string;
  skipPreview: boolean;
  cachedPreviewSchema: PipelinePreviewSchema | null;
  setCachedPreviewSchema: Dispatch<SetStateAction<PipelinePreviewSchema | null>>;
  catalogFlags: CatalogFlagMap;
};

type UseNodePreviewResult = {
  previewState: ReturnType<typeof usePipelinePreview>['previewState'];
  refreshPreview: () => void;
  cachedPreviewSchema: PipelinePreviewSchema | null;
  clearCachedPreviewSchema: () => void;
  canTriggerPreview: boolean;
};

export const useNodePreview = ({
  graphSnapshot,
  graphNodeCount,
  hasReachableSource,
  sourceId,
  node,
  nodeId,
  previewSignature,
  skipPreview,
  catalogFlags,
  cachedPreviewSchema,
  setCachedPreviewSchema,
}: UseNodePreviewArgs): UseNodePreviewResult => {
  const {
    isPreviewNode,
    isFeatureMathNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
    isTrainTestSplitNode,
    isOutlierNode,
    isTransformerAuditNode,
    isCastNode,
  } = catalogFlags;

  const shouldFetchPreview = useMemo(() => {
    if (!graphSnapshot) {
      return false;
    }
    if (!graphNodeCount) {
      return false;
    }
    if (!hasReachableSource) {
      return false;
    }
    return true;
  }, [graphNodeCount, graphSnapshot, hasReachableSource]);

  const canTriggerPreview = useMemo(() => {
    if (!graphSnapshot) {
      return false;
    }
    if (!sourceId) {
      return false;
    }
    if (!hasReachableSource) {
      return false;
    }
    return graphNodeCount > 0;
  }, [graphNodeCount, graphSnapshot, hasReachableSource, sourceId]);

  const shouldIncludeSignals = useMemo(
    () =>
      isPreviewNode ||
      isFeatureMathNode ||
      isPolynomialFeaturesNode ||
      isFeatureSelectionNode ||
      isTrainTestSplitNode ||
      isOutlierNode ||
      isTransformerAuditNode,
    [
      isFeatureMathNode,
      isFeatureSelectionNode,
      isOutlierNode,
      isPolynomialFeaturesNode,
      isPreviewNode,
      isTrainTestSplitNode,
      isTransformerAuditNode,
    ],
  );

  const { previewState, refreshPreview } = usePipelinePreview({
    shouldFetchPreview,
    sourceId,
    canTriggerPreview,
    graphSnapshot,
    catalogFlags,
    targetNodeId: node?.id ?? null,
    previewSignature,
    skipPreview,
    requestPreviewRows: isPreviewNode || isCastNode,
    includeSignals: shouldIncludeSignals,
  });

  useEffect(() => {
    const nextSchema = previewState.data?.schema ?? null;
    if (!nextSchema) {
      return;
    }
    setCachedPreviewSchema((previous) => {
      if (previous?.signature && nextSchema.signature && previous.signature === nextSchema.signature) {
        return previous;
      }
      return nextSchema;
    });
  }, [previewState.data?.schema]);

  useEffect(() => {
    setCachedPreviewSchema(null);
  }, [sourceId]);

  useEffect(() => {
    setCachedPreviewSchema(null);
  }, [previewSignature]);

  const clearCachedPreviewSchema = useCallback(() => {
    setCachedPreviewSchema(null);
  }, []);

  return {
    previewState,
    refreshPreview,
    cachedPreviewSchema,
    clearCachedPreviewSchema,
    canTriggerPreview,
  };
};
