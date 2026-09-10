import type { Edge, Node } from '@xyflow/react';

/** Compare structural node fields while retaining the existing drag suppression. */
function equalHistoryNode(a: Node, b: Node): boolean {
  if (a.id !== b.id) return false;
  if (a.data !== b.data) return false;
  if (a.type !== b.type) return false;
  // Either side dragging suppresses positions, including the drag-end transition.
  if (a.dragging || b.dragging) return true;
  return a.position.x === b.position.x && a.position.y === b.position.y;
}

/** Edge reference changes count; node selection and equivalent replacements do not. */
export function equalGraphHistory(prev: { nodes: Node[]; edges: Edge[] }, next: { nodes: Node[]; edges: Edge[] }): boolean {
  if (prev.edges !== next.edges) return false;
  if (prev.nodes === next.nodes) return true;
  if (prev.nodes.length !== next.nodes.length) return false;
  for (let i = 0; i < prev.nodes.length; i++) {
    if (!equalHistoryNode(prev.nodes[i]!, next.nodes[i]!)) return false;
  }
  return true;
}
