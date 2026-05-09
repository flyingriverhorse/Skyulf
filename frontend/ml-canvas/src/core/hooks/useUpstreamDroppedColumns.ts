import { useMemo } from 'react';
import { getIncomers, Node, Edge } from '@xyflow/react';
import { useGraphStore } from '../store/useGraphStore';

/**
 * Walk upstream from `node` and collect every column name that has been
 * removed from the feature set by:
 *   - DropColumnsNode (definitionType === 'drop_missing_columns')
 *
 * NOTE: FeatureTargetSplit's target column is NOT considered dropped — it
 * still lives in the dataset (in the y slot) and downstream nodes such as
 * Encoding may legitimately want to act on it (e.g. label-encode the target).
 *
 * Used at config-time to hide already-removed columns from downstream
 * column selectors (e.g. encoding column picker).
 */
export function collectDroppedColumns(
  node: Node,
  nodes: Node[],
  edges: Edge[],
  visited: Set<string> = new Set()
): Set<string> {
  if (visited.has(node.id)) return new Set();
  visited.add(node.id);
  const dropped = new Set<string>();
  if (
    node.data?.definitionType === 'drop_missing_columns' &&
    Array.isArray(node.data.columns)
  ) {
    (node.data.columns as string[]).forEach((c) => dropped.add(c));
  }
  for (const incomer of getIncomers(node, nodes, edges)) {
    collectDroppedColumns(incomer, nodes, edges, visited).forEach((c) => dropped.add(c));
  }
  return dropped;
}

/**
 * Returns the set of column names that have been explicitly dropped by
 * upstream DropColumnsNode(s) on the path to `nodeId`.
 *
 * The returned Set is stable across re-renders as long as the graph
 * topology and DropColumnsNode configs do not change.
 */
export function useUpstreamDroppedColumns(nodeId: string | undefined): Set<string> {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  return useMemo(() => {
    if (!nodeId) return new Set<string>();
    const thisNode = nodes.find((n) => n.id === nodeId);
    if (!thisNode) return new Set<string>();
    // Start from incomers so the node's own drops are never hidden from itself.
    const dropped = new Set<string>();
    for (const incomer of getIncomers(thisNode, nodes, edges)) {
      collectDroppedColumns(incomer, nodes, edges).forEach((c) => dropped.add(c));
    }
    return dropped;
  }, [nodeId, nodes, edges]);
}
