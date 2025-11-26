import { useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import { useDatasetProfileController } from '../nodes/dataset/datasetProfile';
import { stableStringify } from '../utils/configParsers';
import type { CatalogFlagMap } from './useCatalogFlags';

type GraphContext = { nodes: any[]; edges: any[] } | null;

type UseDatasetProfilingArgs = {
  node: Node;
  graphContext: GraphContext;
  hasReachableSource: boolean;
  sourceId?: string | null;
  formatRelativeTime: (value?: string | null) => string | null;
  catalogFlags: CatalogFlagMap;
};

type UseDatasetProfilingResult = {
  datasetProfileController: ReturnType<typeof useDatasetProfileController>;
  profileState: ReturnType<typeof useDatasetProfileController>['profileState'];
};

export const useDatasetProfiling = ({
  node,
  graphContext,
  hasReachableSource,
  sourceId,
  formatRelativeTime,
  catalogFlags,
}: UseDatasetProfilingArgs): UseDatasetProfilingResult => {
  const { isDatasetProfileNode } = catalogFlags;

  const profilingGraphSignature = useMemo(
    () => (graphContext ? stableStringify(graphContext) : ''),
    [graphContext],
  );

  const datasetProfileController = useDatasetProfileController({
    node,
    isDatasetProfileNode,
    sourceId,
    hasReachableSource,
    graphContext,
    profilingGraphSignature,
    formatRelativeTime,
  });

  const { profileState } = datasetProfileController;

  return {
    datasetProfileController,
    profileState,
  };
};
