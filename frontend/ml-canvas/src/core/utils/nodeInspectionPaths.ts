import type { NodeInspection } from '../types/nodeInspection';

/** Group repeated executions of a node's data path, preserving differing measurements. */
export function groupNodeInspections(entries: NodeInspection[]): NodeInspection[] {
  const seen = new Set<string>();
  return entries.flatMap(entry => {
    // Bounded samples alone cannot identify a path. Older receipts stay separate.
    if (!entry.path_id) return [entry];
    const key = JSON.stringify([entry.path_id, entry.input, entry.output]);
    if (seen.has(key)) return [];
    seen.add(key);
    return [{ ...entry, branch_label: entry.path_label || entry.branch_label }];
  });
}
