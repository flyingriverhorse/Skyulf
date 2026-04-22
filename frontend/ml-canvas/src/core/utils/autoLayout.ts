import dagre from 'dagre';
import type { Node, Edge } from '@xyflow/react';

// Approximate canvas node footprint. Real nodes vary, but dagre only needs
// rough sizing to compute spacing without overlap.
const NODE_WIDTH = 240;
const NODE_HEIGHT = 120;

/**
 * Arrange canvas nodes left-to-right in topological order using dagre.
 *
 * Used by the "Auto-Layout" toolbar button to tidy up multi-branch pipelines
 * where parallel training paths fan out and become visually chaotic.
 *
 * Returns new node objects with updated `position`. Edges are returned as-is.
 */
export function autoLayoutGraph(
  nodes: Node[],
  edges: Edge[]
): { nodes: Node[]; edges: Edge[] } {
  if (nodes.length === 0) return { nodes, edges };

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  // Left-to-right flow matches the data → preprocessing → model → eval mental
  // model. Generous separation keeps multi-branch fan-outs readable.
  g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 120, marginx: 20, marginy: 20 });

  for (const n of nodes) {
    g.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const e of edges) {
    g.setEdge(e.source, e.target);
  }

  dagre.layout(g);

  const laidOut = nodes.map((n) => {
    const pos = g.node(n.id);
    if (!pos) return n;
    return {
      ...n,
      // dagre returns center-coordinates; React Flow expects top-left.
      position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
    };
  });

  return { nodes: laidOut, edges };
}
