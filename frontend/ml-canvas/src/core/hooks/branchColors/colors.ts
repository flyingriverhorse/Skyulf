import type { Edge, Node } from '@xyflow/react';
import { getBranches, getBranchEdges } from './branches';
import { getIncomingEdges, getTerminals } from './graph';
import { generateBranchColors, getBranchLabel, getTerminalSuffixes } from './labels';
import type { BranchEdgeInfo } from './types';

function getEdgeMembership(branchEdges: Set<string>[]) {
  const counts = new Map<string, number>();
  const first = new Map<string, number>();
  for (let index = 0; index < branchEdges.length; index++) {
    for (const id of branchEdges[index]!) {
      counts.set(id, (counts.get(id) || 0) + 1);
      if (!first.has(id)) first.set(id, index);
    }
  }
  return { counts, first };
}

/** Color shared edges from their first branch while retaining terminal badges. */
function assignColors(branchEdges: Set<string>[], labeledEdges: Set<string>, labels: string[]): Map<string, BranchEdgeInfo> {
  const colors = generateBranchColors(branchEdges.length);
  const { counts, first } = getEdgeMembership(branchEdges);
  const result = new Map<string, BranchEdgeInfo>();
  for (const [id, count] of counts) {
    const index = first.get(id)!;
    result.set(id, { color: colors[index]!,
      label: labeledEdges.has(id) ? (labels[index] ?? null) : null, shared: count > 1 });
  }
  return result;
}

/** Compute the edge presentation map without modifying graph inputs. */
export function getBranchColors(nodes: Node[], edges: Edge[]): Map<string, BranchEdgeInfo> {
  const terminals = getTerminals(nodes, edges);
  if (terminals.length === 0) return new Map();
  const incoming = getIncomingEdges(edges);
  const branches = getBranches(terminals, incoming, nodes);
  if (branches.length < 2) return new Map();
  const suffixes = getTerminalSuffixes(terminals);
  const labeledEdges = new Set<string>();
  const labels = branches.map((branch, index) => getBranchLabel(branch, index, terminals, nodes, suffixes));
  const branchEdges = branches.map(branch => getBranchEdges(branch, incoming, labeledEdges));
  return assignColors(branchEdges, labeledEdges, labels);
}
