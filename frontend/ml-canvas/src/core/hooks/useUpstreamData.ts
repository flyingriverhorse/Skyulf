import { useGraphStore } from '../store/useGraphStore';
import { getIncomers } from '@xyflow/react';

export const useUpstreamData = (nodeId: string) => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);

  const node = nodes.find((n) => n.id === nodeId);
  if (!node) return [];

  const incomers = getIncomers(node, nodes, edges);
  return incomers.map((n) => n.data);
};
