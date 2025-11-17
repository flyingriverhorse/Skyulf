import type { Edge, Node } from 'react-flow-renderer';
import { formatRelativeTime, formatTimestamp, parseServerTimestamp } from '../utils/time';

export const buildGraphSnapshot = (nodes: Node[], edges: Edge[]) => {
  const sanitizedNodes = nodes.map((node) => {
    const { data, ...rest } = node;
    const sanitizedData = data ? { ...data } : undefined;
    if (sanitizedData) {
      delete sanitizedData.onRemoveNode;
      delete sanitizedData.onOpenSettings;
    }
    return JSON.parse(
      JSON.stringify({
        ...rest,
        data: sanitizedData,
      })
    );
  });

  const sanitizedEdges = edges.map((edge) => JSON.parse(JSON.stringify(edge)));

  return {
    nodes: sanitizedNodes,
    edges: sanitizedEdges,
  };
};
