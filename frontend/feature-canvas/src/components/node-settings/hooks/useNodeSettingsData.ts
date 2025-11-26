import { useState } from 'react';
import type { PipelinePreviewSchema } from '../../../api';
import { useGraphTopology } from './useGraphTopology';
import { useModelingConfiguration } from './useModelingConfiguration';
import { usePreviewSignature } from './usePreviewSignature';
import { useSchemaDiagnostics } from './useSchemaDiagnostics';
import { useColumnCatalogState } from './useColumnCatalogState';
import type { CatalogFlagMap } from './useCatalogFlags';

type UseNodeSettingsDataArgs = {
  graphSnapshot: { nodes: any[]; edges: any[] } | null;
  nodeId: string;
  catalogFlags: CatalogFlagMap;
  configState: Record<string, any>;
  setConfigState: any;
  nodeParams: any;
  sourceId?: string | null;
  requiresColumnCatalog: boolean;
  imputerStrategies: any[];
};

export const useNodeSettingsData = ({
  graphSnapshot,
  nodeId,
  catalogFlags,
  configState,
  setConfigState,
  nodeParams,
  sourceId,
  requiresColumnCatalog,
  imputerStrategies,
}: UseNodeSettingsDataArgs) => {
  const graphTopology = useGraphTopology({
    graphSnapshot,
    nodeId,
    catalogFlags,
  });

  const {
    graphContext,
    graphNodes,
    graphNodeCount,
    upstreamNodeIds,
    upstreamTargetColumn,
    hasReachableSource,
  } = graphTopology;

  const modelingConfig = useModelingConfiguration({
    configState,
    setConfigState,
    catalogFlags,
    upstreamTargetColumn,
    nodeId,
    modelTypeOptions: nodeParams.trainModel.modelType?.options ?? null,
  });

  const { resamplingConfig } = modelingConfig;

  const previewSignatureResult = usePreviewSignature({
    nodeId,
    sourceId,
    configState,
    upstreamNodeIds,
    graphNodes,
  });

  const { previewSignature, upstreamConfigFingerprints } = previewSignatureResult;

  const [cachedPreviewSchema, setCachedPreviewSchema] = useState<PipelinePreviewSchema | null>(null);

  const schemaDiagnostics = useSchemaDiagnostics({
    cachedPreviewSchema,
    catalogFlags,
    resamplingTargetColumn: resamplingConfig?.targetColumn ?? null,
    imputerStrategies,
  });

  const columnCatalogState = useColumnCatalogState({
    requiresColumnCatalog,
    catalogFlags,
    nodeId,
    sourceId,
    hasReachableSource,
  });

  return {
    graphTopology,
    modelingConfig,
    previewSignatureResult,
    cachedPreviewSchema,
    setCachedPreviewSchema,
    schemaDiagnostics,
    columnCatalogState,
  };
};
