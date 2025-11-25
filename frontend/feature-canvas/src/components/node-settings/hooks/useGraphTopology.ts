import { useMemo } from 'react';

type GraphSnapshot = {
  nodes?: any[];
  edges?: any[];
} | null;

type GraphContext = {
  nodes: any[];
  edges: any[];
} | null;

type UseGraphTopologyArgs = {
  graphSnapshot: GraphSnapshot;
  nodeId: string;
  isDataset: boolean;
};

type UseGraphTopologyResult = {
  graphContext: GraphContext;
  graphNodes: any[];
  graphEdges: any[];
  graphNodeCount: number;
  upstreamNodeIds: string[];
  upstreamTargetColumn: string;
  hasReachableSource: boolean;
};

export const useGraphTopology = ({
  graphSnapshot,
  nodeId,
  isDataset,
}: UseGraphTopologyArgs): UseGraphTopologyResult => {
  const graphContext = useMemo<GraphContext>(() => {
    if (!graphSnapshot) {
      return null;
    }
    const nodes = Array.isArray(graphSnapshot.nodes) ? graphSnapshot.nodes : [];
    const edges = Array.isArray(graphSnapshot.edges) ? graphSnapshot.edges : [];
    return { nodes, edges };
  }, [graphSnapshot]);

  const graphNodes = useMemo(() => (graphSnapshot && Array.isArray(graphSnapshot.nodes) ? graphSnapshot.nodes : []), [graphSnapshot]);
  const graphEdges = useMemo(() => (graphSnapshot && Array.isArray(graphSnapshot.edges) ? graphSnapshot.edges : []), [graphSnapshot]);
  const graphNodeCount = graphNodes.length;

  const upstreamNodeIds = useMemo(() => {
    if (!nodeId || !graphEdges.length) {
      return [] as string[];
    }
    const visited = new Set<string>();
    const stack: string[] = [nodeId];
    while (stack.length) {
      const current = stack.pop();
      if (!current) {
        continue;
      }
      graphEdges.forEach((edge: any) => {
        const sourceRaw = edge && typeof edge.source === 'string' ? edge.source.trim() : '';
        const targetRaw = edge && typeof edge.target === 'string' ? edge.target.trim() : '';
        if (!sourceRaw || !targetRaw) {
          return;
        }
        if (targetRaw === current && !visited.has(sourceRaw)) {
          visited.add(sourceRaw);
          stack.push(sourceRaw);
        }
      });
    }
    const ordered = Array.from(visited);
    ordered.sort();
    return ordered;
  }, [graphEdges, nodeId]);

  const upstreamTargetColumn = useMemo(() => {
    if (!upstreamNodeIds.length) {
      return '';
    }

    let featureTargetSplitColumn = '';
    let trainTestSplitColumn = '';

    for (const upstreamId of upstreamNodeIds) {
      const upstreamNode = graphNodes.find((n: any) => n?.id === upstreamId);
      if (!upstreamNode) {
        continue;
      }

      const catalogTypeValue = String(upstreamNode?.data?.catalogType ?? '').toLowerCase().trim();

      if (catalogTypeValue === 'feature_target_split') {
        const targetCol = upstreamNode?.data?.config?.target_column;
        if (typeof targetCol === 'string' && targetCol.trim()) {
          featureTargetSplitColumn = targetCol.trim();
        }
      } else if (catalogTypeValue === 'train_test_split') {
        const targetCol = upstreamNode?.data?.config?.target_column;
        if (typeof targetCol === 'string' && targetCol.trim()) {
          trainTestSplitColumn = targetCol.trim();
        }
      }
    }

    return featureTargetSplitColumn || trainTestSplitColumn || '';
  }, [graphNodes, upstreamNodeIds]);

  const datasetNodeIds = useMemo(() => {
    const result = new Set<string>();
    graphNodes.forEach((entry: any) => {
      if (!entry) {
        return;
      }
      const entryId = typeof entry.id === 'string' ? entry.id.trim() : String(entry?.id ?? '').trim();
      if (!entryId) {
        return;
      }
      const entryData = entry?.data ?? {};
      if (entryId === 'dataset-source' || entryData?.isDataset === true || entryData?.catalogType === 'dataset') {
        result.add(entryId);
      }
    });
    if (!result.size) {
      result.add('dataset-source');
    }
    return Array.from(result);
  }, [graphNodes]);

  const hasReachableSource = useMemo(() => {
    if (!nodeId) {
      return false;
    }

    if (isDataset) {
      return true;
    }

    if (!graphEdges.length) {
      return datasetNodeIds.includes(nodeId);
    }

    const adjacency = new Map<string, string[]>();
    graphEdges.forEach((edge: any) => {
      const rawSource = edge?.source;
      const rawTarget = edge?.target;
      const source = typeof rawSource === 'string' ? rawSource.trim() : String(rawSource ?? '').trim();
      const target = typeof rawTarget === 'string' ? rawTarget.trim() : String(rawTarget ?? '').trim();
      if (!source || !target) {
        return;
      }
      const list = adjacency.get(source);
      if (list) {
        list.push(target);
      } else {
        adjacency.set(source, [target]);
      }
    });

    const visited = new Set<string>();
    const stack = [...datasetNodeIds];

    while (stack.length) {
      const current = stack.pop();
      if (!current || visited.has(current)) {
        continue;
      }
      visited.add(current);
      const neighbors = adjacency.get(current);
      if (neighbors) {
        neighbors.forEach((neighbor) => {
          if (neighbor && !visited.has(neighbor)) {
            stack.push(neighbor);
          }
        });
      }
    }

    return visited.has(nodeId);
  }, [datasetNodeIds, graphEdges, isDataset, nodeId]);

  return {
    graphContext,
    graphNodes,
    graphEdges,
    graphNodeCount,
    upstreamNodeIds,
    upstreamTargetColumn,
    hasReachableSource,
  };
};
