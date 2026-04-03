import { useMemo } from 'react';
import type { Node, Edge } from '@xyflow/react';

const TRAINING_TYPES = new Set(['basic_training', 'advanced_tuning']);

/** Generate n evenly-spaced, high-saturation HSL colors with a golden-angle offset for variety. */
function generateBranchColors(count: number): string[] {
  const colors: string[] = [];
  const goldenAngle = 137.508; // degrees — maximises hue separation
  for (let i = 0; i < count; i++) {
    const hue = (i * goldenAngle) % 360;
    colors.push(`hsl(${Math.round(hue)}, 80%, 65%)`);
  }
  return colors;
}

/** Convert snake_case model id to a readable name: "random_forest_classifier" → "Random Forest" */
function prettifyModelType(modelType: string): string {
  return modelType
    .replace(/_classifier$|_regressor$/, '')
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export interface BranchEdgeInfo {
  color: string;
  /** Label only set on the terminal edge (edge entering the training node). */
  label: string | null;
  /** True when this edge feeds multiple training nodes (shared infrastructure). */
  shared: boolean;
}

/**
 * Assigns branch colors to edges based on which training/tuning terminal
 * they belong to. Shared edges (belonging to multiple branches)
 * return null (keep default gradient).
 *
 * The branch label is only attached to the **last edge** entering the
 * terminal node — not every edge in the branch.
 */
export function useBranchColors(nodes: Node[], edges: Edge[]): Map<string, BranchEdgeInfo> {
  return useMemo(() => {
    const colorMap = new Map<string, BranchEdgeInfo>();

    // Find connected training/tuning nodes
    const terminals = nodes.filter(
      n => TRAINING_TYPES.has(n.data.definitionType as string) && edges.some(e => e.target === n.id)
    );

    // Need 2+ terminals for branch coloring to matter
    if (terminals.length < 2) return colorMap;

    const colors = generateBranchColors(terminals.length);

    // Build adjacency: target → source edges (for BFS backwards)
    const incomingMap = new Map<string, Edge[]>();
    for (const edge of edges) {
      const list = incomingMap.get(edge.target) || [];
      list.push(edge);
      incomingMap.set(edge.target, list);
    }

    // Collect the terminal-entering edge ids so we can tag them with labels
    const terminalEdgeIds = new Set<string>();

    // For each terminal, BFS backwards to collect all ancestor edges
    const branchEdgeSets: Set<string>[] = [];
    const branchLabels: string[] = [];
    for (let i = 0; i < terminals.length; i++) {
      const terminal = terminals[i];
      const modelType = terminal.data.model_type as string | undefined;
      const modelName = modelType ? prettifyModelType(modelType) : '';
      const label = modelName ? `Branch ${i + 1} · ${modelName}` : `Branch ${i + 1}`;
      branchLabels.push(label);

      // Mark only the first edge entering this terminal for the label
      const terminalIncoming = incomingMap.get(terminal.id) || [];
      if (terminalIncoming.length > 0) {
        terminalEdgeIds.add(terminalIncoming[0].id);
      }

      const visited = new Set<string>();
      const branchEdges = new Set<string>();
      const queue = [terminal.id];

      while (queue.length > 0) {
        const nodeId = queue.shift()!;
        if (visited.has(nodeId)) continue;
        visited.add(nodeId);

        const incoming = incomingMap.get(nodeId) || [];
        for (const edge of incoming) {
          branchEdges.add(edge.id);
          queue.push(edge.source);
        }
      }
      branchEdgeSets.push(branchEdges);
    }

    // Count how many branches each edge belongs to
    const edgeBranchCount = new Map<string, number>();
    const edgeFirstBranch = new Map<string, number>();
    for (let i = 0; i < branchEdgeSets.length; i++) {
      for (const edgeId of branchEdgeSets[i]) {
        edgeBranchCount.set(edgeId, (edgeBranchCount.get(edgeId) || 0) + 1);
        if (!edgeFirstBranch.has(edgeId)) {
          edgeFirstBranch.set(edgeId, i);
        }
      }
    }

    // Assign colors — multi-branch edges get first branch color + shared flag
    for (const [edgeId, count] of edgeBranchCount) {
      const branchIdx = edgeFirstBranch.get(edgeId)!;
      colorMap.set(edgeId, {
        color: colors[branchIdx],
        label: terminalEdgeIds.has(edgeId) ? branchLabels[branchIdx] : null,
        shared: count > 1,
      });
    }

    return colorMap;
  }, [nodes, edges]);
}
