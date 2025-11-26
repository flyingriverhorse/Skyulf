import { useNodePreview } from './useNodePreview';
import { usePreviewSignals } from './usePreviewSignals';
import { useFeatureSelectionAutoConfig } from './useFeatureSelectionAutoConfig';
import { useFeatureMathState } from './useFeatureMathState';
import { usePreviewData } from './usePreviewData';
import type { PipelinePreviewSchema } from '../../../api';
import type { CatalogFlagMap } from './useCatalogFlags';

type UseNodeSettingsPreviewArgs = {
  graphSnapshot: { nodes: any[]; edges: any[] } | null;
  graphNodeCount: number;
  hasReachableSource: boolean;
  sourceId?: string | null;
  node: any;
  nodeId: string;
  previewSignature: string;
  skipPreview: boolean;
  cachedPreviewSchema: PipelinePreviewSchema | null;
  setCachedPreviewSchema: any;
  catalogFlags: CatalogFlagMap;
  upstreamTargetColumn: string | null;
  setConfigState: any;
  configState: Record<string, any>;
};

export const useNodeSettingsPreview = ({
  graphSnapshot,
  graphNodeCount,
  hasReachableSource,
  sourceId,
  node,
  nodeId,
  previewSignature,
  skipPreview,
  cachedPreviewSchema,
  setCachedPreviewSchema,
  catalogFlags,
  upstreamTargetColumn,
  setConfigState,
  configState,
}: UseNodeSettingsPreviewArgs) => {
  const nodePreview = useNodePreview({
    graphSnapshot,
    graphNodeCount,
    hasReachableSource,
    sourceId,
    node,
    nodeId,
    previewSignature,
    skipPreview,
    cachedPreviewSchema,
    setCachedPreviewSchema,
    catalogFlags,
  });

  const { previewState } = nodePreview;

  const previewSignals = usePreviewSignals({
    previewState,
    nodeId,
    catalogFlags,
  });

  const { featureSelectionSignal, featureMathSignals } = previewSignals;

  useFeatureSelectionAutoConfig({
    catalogFlags,
    featureSelectionSignal,
    upstreamTargetColumn: upstreamTargetColumn ?? '',
    setConfigState,
  });

  const featureMathState = useFeatureMathState(catalogFlags, configState, featureMathSignals);

  const previewData = usePreviewData(previewState);

  return {
    nodePreview,
    previewSignals,
    featureMathState,
    previewData,
  };
};
