import { useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import { useDatasetProfileController } from '../nodes/dataset/datasetProfile';
import { stableStringify } from '../utils/configParsers';

type GraphContext = { nodes: any[]; edges: any[] } | null;

type UseDatasetProfilingArgs = {
  node: Node;
  graphContext: GraphContext;
  hasReachableSource: boolean;
  sourceId?: string | null;
  formatRelativeTime: (value?: string | null) => string | null;
};

type UseDatasetProfilingResult = {
  isPreviewNode: boolean;
  isDatasetProfileNode: boolean;
  datasetProfileController: ReturnType<typeof useDatasetProfileController>;
  profileState: ReturnType<typeof useDatasetProfileController>['profileState'];
};

export const useDatasetProfiling = ({
  node,
  graphContext,
  hasReachableSource,
  sourceId,
  formatRelativeTime,
}: UseDatasetProfilingArgs): UseDatasetProfilingResult => {
  const isPreviewNode = useMemo(() => node?.data?.catalogType === 'data_preview', [node?.data?.catalogType]);

  const isDatasetProfileNode = useMemo(
    () => node?.data?.catalogType === 'dataset_profile',
    [node?.data?.catalogType],
  );

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
    isPreviewNode,
    isDatasetProfileNode,
    datasetProfileController,
    profileState,
  };
};
