import type { Edge, Node } from '@xyflow/react';

interface GraphHistoryState {
  nodes: Node[];
  edges: Edge[];
}

/** Keep live drag frames out of snapshots until the entire gesture has finished. */
export function createGraphHistoryPartializer<T extends GraphHistoryState>() {
  let settledPositions = new Map<string, Node['position']>();

  return ({ nodes, edges }: T): GraphHistoryState => {
    if (!nodes.some(node => node.dragging)) {
      settledPositions = new Map(nodes.map(node => [node.id, node.position]));
      return { nodes, edges };
    }

    return {
      nodes: nodes.map(node => ({
        ...node,
        position: settledPositions.get(node.id) ?? node.position,
        dragging: false,
      })),
      edges,
    };
  };
}

/** Compare committed node structure while ignoring selection and measured dimensions. */
function equalHistoryNode(a: Node, b: Node): boolean {
  if (a.id !== b.id) return false;
  if (a.data !== b.data) return false;
  if (a.type !== b.type) return false;
  return a.position.x === b.position.x && a.position.y === b.position.y;
}

/** Ignore edge selection while retaining every other saved edge field. */
function equalHistoryEdge(a: Edge, b: Edge): boolean {
  if (a === b) return true;
  const keys = Object.keys(a).filter(key => key !== 'selected');
  if (keys.length !== Object.keys(b).filter(key => key !== 'selected').length) return false;
  return keys.every(key => Object.hasOwn(b, key) && a[key as keyof Edge] === b[key as keyof Edge]);
}

/** Compare structural graph snapshots without recording node or edge selection. */
export function equalGraphHistory(prev: GraphHistoryState, next: GraphHistoryState): boolean {
  if (prev.edges.length !== next.edges.length) return false;
  if (!prev.edges.every((edge, index) => equalHistoryEdge(edge, next.edges[index]!))) return false;
  if (prev.nodes === next.nodes) return true;
  if (prev.nodes.length !== next.nodes.length) return false;
  return prev.nodes.every((node, index) => equalHistoryNode(node, next.nodes[index]!));
}
