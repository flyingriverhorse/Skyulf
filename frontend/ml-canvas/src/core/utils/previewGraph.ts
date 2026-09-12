import type { Node, Edge } from '@xyflow/react';

/** Use the same graph for Preview execution and its canvas path labels. */
export function getPreviewGraph(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const excluded = new Set(nodes.filter(node => node.data.definitionType === 'data_preview').map(node => node.id));
  return {
    nodes: nodes.filter(node => !excluded.has(node.id)),
    edges: edges.filter(edge => !excluded.has(edge.source) && !excluded.has(edge.target)),
  };
}
