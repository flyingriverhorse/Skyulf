import { useMemo } from 'react';
import { useGraphStore } from '../store/useGraphStore';
import { getIncomers, Node, Edge } from '@xyflow/react';

const findDatasetId = (node: Node, nodes: Node[], edges: Edge[], visited = new Set<string>()): string | undefined => {
  if (visited.has(node.id)) return undefined;
  visited.add(node.id);

  if (node.data?.datasetId) return node.data.datasetId as string;

  const incomers = getIncomers(node, nodes, edges);
  for (const incomer of incomers) {
    const id = findDatasetId(incomer, nodes, edges, visited);
    if (id) return id;
  }
  return undefined;
};

export const useUpstreamData = (nodeId: string) => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);

  // Memoized: without this, every one of the ~25 node-settings panels
  // consuming this hook got a brand-new array/object on every render
  // (even when the graph itself hadn't changed since last render),
  // defeating any downstream useMemo/useEffect keyed on `upstreamData`.
  return useMemo(() => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return [];

    const incomers = getIncomers(node, nodes, edges);
    return incomers.map((n) => {
      const datasetId = findDatasetId(n, nodes, edges);
      if (datasetId && !n.data.datasetId) {
        return { ...n.data, datasetId };
      }
      return n.data;
    });
  }, [nodeId, nodes, edges]);
};
