import { useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import { useNodeMetadata } from './useNodeMetadata';
import { useConnectionReadiness } from './useConnectionReadiness';
import { useCatalogFlags } from './useCatalogFlags';
import { useNodeParameters } from './useNodeParameters';
import { useResetPermissions } from './useResetPermissions';
import { useNodeConfigState } from './useNodeConfigState';
import { useParameterHandlers } from './useParameterHandlers';
import { useNodeSpecificParameters } from './useNodeSpecificParameters';
import { useFilteredParameters } from './useFilteredParameters';
import { useImputationStrategies } from './useImputationStrategies';
import { useSkewnessState } from './useSkewnessState';
import { DATA_CONSISTENCY_GUIDANCE } from '../utils/guidance';

type UseNodeSettingsStateArgs = {
  node: Node;
  graphSnapshot: { nodes: any[]; edges: any[] } | null;
  isResetAvailable: boolean;
  defaultConfigTemplate?: Record<string, any> | null;
};

export const useNodeSettingsState = ({
  node,
  graphSnapshot,
  isResetAvailable,
  defaultConfigTemplate,
}: UseNodeSettingsStateArgs) => {
  const { metadata, title } = useNodeMetadata(node);

  const connectionReadiness = useConnectionReadiness(node, graphSnapshot);

  const catalogFlags = useCatalogFlags(node);
  const {
    catalogType,
    isDataset,
    isClassUndersamplingNode,
    isClassOversamplingNode,
    isDataConsistencyNode,
    isSkewnessNode,
    isSkewnessDistributionNode,
  } = catalogFlags;

  const datasetBadge = isDataset;
  const isClassResamplingNode = isClassUndersamplingNode || isClassOversamplingNode;

  const {
    parameters,
    getParameter,
    requiresColumnCatalog,
    dropColumnParameter,
  } = useNodeParameters(node);

  const dataConsistencyHint = useMemo(
    () => DATA_CONSISTENCY_GUIDANCE[catalogType] ?? null,
    [catalogType]
  );

  const nodeId = typeof node?.id === 'string' ? node.id : '';

  const resetPermissions = useResetPermissions({
    isResetAvailable,
    defaultConfigTemplate,
    catalogFlags,
  });

  const nodeParams = useNodeSpecificParameters(getParameter, catalogFlags);
  const filteredParameters = useFilteredParameters(parameters, catalogFlags);

  const dataConsistencyParameters = useMemo(
    () => (isDataConsistencyNode ? filteredParameters : []),
    [filteredParameters, isDataConsistencyNode],
  );

  const shouldLoadSkewnessInsights = isSkewnessNode || isSkewnessDistributionNode;

  const configStateResult = useNodeConfigState({ node });
  const { setConfigState, configState } = configStateResult;

  const parameterHandlers = useParameterHandlers({ setConfigState });

  const imputationStrategiesResult = useImputationStrategies(configState);
  const skewnessStateResult = useSkewnessState(configState, setConfigState);

  return {
    metadata,
    title,
    connectionReadiness,
    catalogFlags,
    datasetBadge,
    isClassResamplingNode,
    parameters,
    getParameter,
    requiresColumnCatalog,
    dropColumnParameter,
    dataConsistencyHint,
    nodeId,
    resetPermissions,
    nodeParams,
    filteredParameters,
    dataConsistencyParameters,
    shouldLoadSkewnessInsights,
    configStateResult,
    parameterHandlers,
    imputationStrategiesResult,
    skewnessStateResult,
  };
};
