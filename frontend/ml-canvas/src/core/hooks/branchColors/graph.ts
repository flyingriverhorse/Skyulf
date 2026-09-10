import type { Edge, Node } from '@xyflow/react';
import { EXECUTION_MODE_AWARE_TYPES } from '../../types/executionMode';
import type { IncomingEdges } from './types';

// Data Preview executes separately; ensemble-only models are specification
// providers, so only the consuming ensemble gets a terminal branch.
const TERMINAL_TYPES = new Set([...EXECUTION_MODE_AWARE_TYPES]);
const MODEL_SOURCE_TYPES = new Set(['training', 'classification', 'regression', 'text_classification']);
const ENSEMBLE_TYPE = 'EnsembleNode';

function isSpecOnlyBaseModel(node: Node, edges: Edge[], ensembleIds: Set<string>): boolean {
  if (ensembleIds.size === 0) return false;
  const defType = node.data.definitionType;
  if (!(typeof defType === 'string' && MODEL_SOURCE_TYPES.has(defType))) return false;
  const outs = edges.filter(edge => edge.source === node.id);
  return outs.length > 0 && outs.every(edge => ensembleIds.has(edge.target));
}

function isConnectedTerminal(node: Node, edges: Edge[], ensembleIds: Set<string>): boolean {
  return (TERMINAL_TYPES.has(node.data.definitionType as string) || node.data.definitionType === ENSEMBLE_TYPE)
    && edges.some(edge => edge.target === node.id)
    && !isSpecOnlyBaseModel(node, edges, ensembleIds);
}

function isPreviewLeaf(node: Node, edges: Edge[], consumed: Set<string>, known: Set<string>): boolean {
  return !known.has(node.id) && !consumed.has(node.id)
    && edges.some(edge => edge.target === node.id) && node.data.definitionType !== 'data_preview';
}

/** Include connected modeling terminals and dangling preprocessing leaves. */
export function getTerminals(nodes: Node[], edges: Edge[]): Node[] {
  const ensembleIds = new Set(nodes.filter(node => node.data.definitionType === ENSEMBLE_TYPE).map(node => node.id));
  const terminals = nodes.filter(node => isConnectedTerminal(node, edges, ensembleIds));
  const consumed = new Set(edges.map(edge => edge.source));
  const known = new Set(terminals.map(node => node.id));
  for (const node of nodes) {
    if (isPreviewLeaf(node, edges, consumed, known)) terminals.push(node);
  }
  const order = getTraversalOrder(nodes, edges);
  return terminals.sort((a, b) => (order.get(a.id) ?? 999999) - (order.get(b.id) ?? 999999));
}

/** Match the forward BFS order used by pipeline conversion and preview tabs. */
function getTraversalOrder(nodes: Node[], edges: Edge[]): Map<string, number> {
  const outgoing = getOutgoingNodes(edges);
  const roots = nodes.filter(node => !edges.some(edge => edge.target === node.id)).map(node => node.id);
  const order = new Map<string, number>();
  const queue = [...roots];
  const seen = new Set(roots);
  while (queue.length > 0) {
    const id = queue.shift()!;
    if (!order.has(id)) order.set(id, order.size);
    for (const child of outgoing.get(id) ?? []) {
      if (!seen.has(child)) { seen.add(child); queue.push(child); }
    }
  }
  return order;
}

function getOutgoingNodes(edges: Edge[]): Map<string, string[]> {
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    const children = outgoing.get(edge.source) ?? [];
    children.push(edge.target);
    outgoing.set(edge.source, children);
  }
  return outgoing;
}

/** Index incoming edges without changing their submission order. */
export function getIncomingEdges(edges: Edge[]): IncomingEdges {
  const incoming = new Map<string, Edge[]>();
  for (const edge of edges) {
    const list = incoming.get(edge.target) || [];
    list.push(edge);
    incoming.set(edge.target, list);
  }
  return incoming;
}
