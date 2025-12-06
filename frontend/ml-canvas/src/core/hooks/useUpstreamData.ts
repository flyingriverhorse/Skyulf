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
};
