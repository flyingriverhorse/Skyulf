import type { Edge, Node } from '@xyflow/react';

/** Clustering and nested ensembles have no supported supervised base-model recipe. */
export function ensembleSourceIssue(sourceType: unknown): string | null {
  if (sourceType !== 'SegmentationNode' && sourceType !== 'EnsembleNode') return null;
  return 'Segmentation and ensemble outputs cannot be used as base models. Connect the upstream dataset or a classification or regression model instead.';
}

/** Inspect saved wires as well as new connections without making graph conversion throw. */
export function findEnsembleConnectionIssues(nodes: Node[], edges: Edge[]) {
  const nodesById = new Map(nodes.map(node => [node.id, node]));
  return edges.flatMap(edge => {
    if (nodesById.get(edge.target)?.data.definitionType !== 'EnsembleNode') return [];
    const message = ensembleSourceIssue(nodesById.get(edge.source)?.data.definitionType);
    return message ? [{ sourceId: edge.source, targetId: edge.target, message }] : [];
  });
}
