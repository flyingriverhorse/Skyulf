import { getIncomers, type Edge, type Node } from '@xyflow/react';

/** Columns a merge is predicted to receive from more than one branch. */
export interface PredictedMergeConflict {
  columns: string[];
  branchIds: string[];
}

/**
 * Config keys holding the column list a node rewrites, per node type.
 *
 * Only value-modifying steps are listed. A node absent from this map is
 * assumed to write nothing, so an unrecognised type never produces a false
 * conflict warning — the engine's post-run advisory remains the source of
 * truth.
 */
const COLUMN_WRITERS: Record<string, string> = {
  scale_numeric_features: 'columns',
  imputation_node: 'columns',
  encoding: 'columns',
  BinningNode: 'columns',
  casting: 'columns',
  outlier: 'columns',
  InvalidValueReplacement: 'columns',
  AliasReplacement: 'columns',
  TextCleaning: 'columns',
};

/** Column names a single node rewrites, based on its current configuration. */
export function columnsWrittenBy(node: Node): string[] {
  const definitionType = node.data?.definitionType as string | undefined;
  if (!definitionType) return [];

  // TransformationNode nests its targets one level deeper, as a list of rules.
  if (definitionType === 'TransformationNode') {
    const rules = node.data.transformations;
    if (!Array.isArray(rules)) return [];
    return rules.flatMap((rule) => {
      const columns = (rule as { columns?: unknown })?.columns;
      return Array.isArray(columns) ? (columns as string[]) : [];
    });
  }

  const key = COLUMN_WRITERS[definitionType];
  if (!key) return [];
  const columns = node.data[key];
  return Array.isArray(columns) ? (columns as string[]) : [];
}

/** Every ancestor of `nodeId`, walking upstream through all input edges. */
function ancestorsOf(nodeId: string, nodes: Node[], edges: Edge[]): Set<string> {
  const seen = new Set<string>();
  const stack = [nodeId];
  while (stack.length) {
    const current = stack.pop()!;
    const node = nodes.find((n) => n.id === current);
    if (!node) continue;
    for (const parent of getIncomers(node, nodes, edges)) {
      if (seen.has(parent.id)) continue;
      seen.add(parent.id);
      stack.push(parent.id);
    }
  }
  return seen;
}

/**
 * Predict, before a run, which columns two parallel branches would both rewrite.
 *
 * The engine decides conflicts by diffing real values against the nearest
 * shared ancestor, which is only knowable after execution. This is the
 * config-time approximation: work done *before* the branches split is shared
 * by both and therefore excluded, and only columns named by 2+ branches are
 * reported. Returns `null` when nothing is predicted to contend.
 */
export function predictMergeConflict(
  targetNodeId: string,
  nodes: Node[],
  edges: Edge[]
): PredictedMergeConflict | null {
  const target = nodes.find((n) => n.id === targetNodeId);
  if (!target) return null;

  const branchRoots = getIncomers(target, nodes, edges);
  const branchIds = [...new Set(branchRoots.map((n) => n.id))];
  if (branchIds.length < 2) return null;

  const ancestorsPerBranch = branchIds.map((id) => {
    const set = ancestorsOf(id, nodes, edges);
    set.add(id);
    return set;
  });
  const shared = ancestorsPerBranch.reduce((acc, set) => new Set([...acc].filter((id) => set.has(id))));

  const writesPerBranch = ancestorsPerBranch.map((branch) => {
    const columns = new Set<string>();
    for (const id of branch) {
      if (shared.has(id)) continue;
      const node = nodes.find((n) => n.id === id);
      if (node) columnsWrittenBy(node).forEach((c) => columns.add(c));
    }
    return columns;
  });

  const counts = new Map<string, number>();
  for (const columns of writesPerBranch) {
    for (const column of columns) counts.set(column, (counts.get(column) ?? 0) + 1);
  }
  const contested = [...counts.entries()].filter(([, n]) => n > 1).map(([c]) => c);
  if (!contested.length) return null;

  return {
    columns: contested.sort(),
    branchIds: branchIds.filter((_, i) => contested.some((c) => writesPerBranch[i]!.has(c))),
  };
}
