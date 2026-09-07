import type { Connection, Edge, Node } from '@xyflow/react';
import { registry } from '../registry/NodeRegistry';

/** Split artifacts travel together; validation joins only when its configured share is positive. */
export function splitOutputHandles(node: Node | undefined): string[] {
  if (node?.data.definitionType === 'TrainTestSplitter') {
    return Number(node.data.validation_size ?? 0) > 0 ? ['train', 'validation', 'test'] : ['train', 'test'];
  }
  return node?.data.definitionType === 'feature_target_split' ? ['X', 'y'] : [];
}

/** Store one logical wire for a split group, irrespective of which member started the gesture. */
export function bundleConnection(nodes: Node[], connection: Connection): Connection {
  const handles = splitOutputHandles(nodes.find(node => node.id === connection.source));
  return handles.length ? { ...connection, sourceHandle: handles[0]! } : connection;
}

/** Fold older per-handle split wires on load/paste while keeping separate downstream branches. */
export function normalizeSplitEdges(nodes: Node[], edges: Edge[]): Edge[] {
  const byId = new Map(nodes.map(node => [node.id, node]));
  const groups = new Map<string, number>();
  const result: Edge[] = [];
  for (const edge of edges) {
    const handles = splitOutputHandles(byId.get(edge.source));
    if (!handles.length) { result.push(edge); continue; }
    const target = byId.get(edge.target);
    const targetHandle = edge.targetHandle ?? registry.get(String(target?.data.definitionType))?.inputs[0]?.id ?? null;
    const key = JSON.stringify([edge.source, edge.target, targetHandle]);
    const index = groups.get(key);
    if (index !== undefined) {
      const previous = result[index]!;
      result[index] = { ...previous, selected: Boolean(previous.selected || edge.selected),
        ...(edge.deletable === false ? { deletable: false } : {}) };
    } else {
      groups.set(key, result.length);
      result.push({ ...edge, type: 'custom', sourceHandle: handles[0]!, targetHandle });
    }
  }
  return result;
}
