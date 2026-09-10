import type { Edge, Node } from '@xyflow/react';
import { isParallelExecution } from '../../types/executionMode';
import type { BranchDef, IncomingEdges } from './types';

function isYPassthrough(edge: Edge, nodes: Node[]): boolean {
  if (edge.sourceHandle !== 'y') return false;
  return nodes.find(node => node.id === edge.source)?.data.definitionType === 'feature_target_split';
}

/** Split parallel terminals by source node, keeping multi-handle outputs together. */
function getParallelBranches(terminal: Node, mainEdges: Edge[], yHelperEdges: Edge[]): BranchDef[] {
  const branches: BranchDef[] = [];
  const seen = new Set<string>();
  for (const edge of mainEdges) {
    if (seen.has(edge.source)) continue;
    seen.add(edge.source);
    branches.push({ terminal, inputEdge: edge,
      allTerminalEdges: mainEdges.filter(candidate => candidate.source === edge.source),
      yHelperEdges, localIndex: branches.length });
  }
  return branches;
}

/** Target-label passthrough edges cooperate with data inputs instead of branching. */
export function getBranches(terminals: Node[], incoming: IncomingEdges, nodes: Node[]): BranchDef[] {
  const branches: BranchDef[] = [];
  for (const terminal of terminals) {
    const terminalEdges = incoming.get(terminal.id) || [];
    if (terminalEdges.length === 0) continue;
    const mainEdges = terminalEdges.filter(edge => !isYPassthrough(edge, nodes));
    const yHelperEdges = terminalEdges.filter(edge => isYPassthrough(edge, nodes));
    const sourceCount = new Set(mainEdges.map(edge => edge.source)).size;
    const parallel = isParallelExecution(terminal.data, sourceCount);
    if (parallel && sourceCount > 1) {
      for (const branch of getParallelBranches(terminal, mainEdges, yHelperEdges)) branches.push(branch);
    } else {
      branches.push({ terminal, inputEdge: null, allTerminalEdges: [], yHelperEdges, localIndex: 0 });
    }
  }
  return branches;
}

/** Traverse ancestors once per branch, including cycles without revisiting nodes. */
function visitAncestors(queue: string[], incoming: IncomingEdges, branchEdges: Set<string>): Set<string> {
  const visited = new Set<string>();
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    for (const edge of incoming.get(nodeId) || []) {
      branchEdges.add(edge.id);
      queue.push(edge.source);
    }
  }
  return visited;
}

/** Collect one branch and register only its first terminal edge for a path badge. */
export function getBranchEdges(branch: BranchDef, incoming: IncomingEdges, labeledEdges: Set<string>): Set<string> {
  const branchEdges = new Set<string>();
  if (!branch.inputEdge) {
    const terminalEdges = incoming.get(branch.terminal.id) || [];
    if (terminalEdges.length > 0) labeledEdges.add(terminalEdges[0]!.id);
    visitAncestors([branch.terminal.id], incoming, branchEdges);
    return branchEdges;
  }
  const terminalEdges = branch.allTerminalEdges.length > 0 ? branch.allTerminalEdges : [branch.inputEdge];
  labeledEdges.add(terminalEdges[0]!.id);
  for (const edge of terminalEdges) branchEdges.add(edge.id);
  const visited = visitAncestors(terminalEdges.map(edge => edge.source), incoming, branchEdges);
  for (const edge of branch.yHelperEdges) {
    if (visited.has(edge.source)) branchEdges.add(edge.id);
  }
  return branchEdges;
}
