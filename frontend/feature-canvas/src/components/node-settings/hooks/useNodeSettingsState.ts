import { useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import { useNodeMetadata } from './core/useNodeMetadata';
import { useConnectionReadiness } from './core/useConnectionReadiness';
import { useCatalogFlags } from './core/useCatalogFlags';
import { useNodeParameters } from './core/useNodeParameters';
import { useResetPermissions } from './core/useResetPermissions';
import { useNodeConfigState } from './core/useNodeConfigState';
import { useParameterHandlers } from './core/useParameterHandlers';
import { useNodeSpecificParameters } from './core/useNodeSpecificParameters';
import { useFilteredParameters } from './core/useFilteredParameters';
import { useImputationStrategies } from './imputation/useImputationStrategies';
import { useSkewnessState } from './skewness/useSkewnessState';
import { useNodeCatalog } from './core/useNodeCatalog';
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

  const { catalogEntryMap } = useNodeCatalog();
  const catalogEntry = useMemo(
    () => (catalogType ? catalogEntryMap.get(catalogType) : null),
    [catalogType, catalogEntryMap],
  );

  const { metadata, title } = useNodeMetadata(node, catalogEntry);

  const {
    parameters,
    getParameter,
    requiresColumnCatalog,
    dropColumnParameter,
  } = useNodeParameters(node, catalogEntry);

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
    catalogEntry,
  };
};
