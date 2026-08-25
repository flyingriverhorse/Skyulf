import { NodeConfigModel } from '../api/client';

/**
 * Client-side mirror of the backend's pre-execution cycle guard
 * (`backend/ml_pipeline/_execution/_cycle_validation.py`).
 *
 * Connect-time checks block cycles one edge at a time, but bulk graph
 * loads (saved projects) apply nodes/edges verbatim, so a loop can still
 * sit on the canvas. Surfacing it here means the validation panel flags
 * the exact loop instantly instead of the run dying late with a cryptic
 * "Artifact not found" error.
 */

export interface CycleIssue {
  /** Node ids of one loop, in walk order. Excludes nodes merely fed by the loop. */
  loopNodeIds: string[];
}

/**
 * Returns one issue per cycle in the graph, or `[]` if the graph is a DAG.
 * Kahn's algorithm finds the nodes that never reach in-degree 0 (on or
 * downstream of a cycle); those are pruned to the exact loop members so
 * each issue names only the loop itself.
 */
export function findCycleIssues(nodes: NodeConfigModel[]): CycleIssue[] {
  const known = new Set(nodes.map((n) => n.node_id));
  const inputs = new Map<string, string[]>();
  const children = new Map<string, string[]>();
  for (const n of nodes) {
    inputs.set(n.node_id, n.inputs.filter((id) => known.has(id)));
    children.set(n.node_id, []);
  }
  const inDegree = new Map<string, number>();
  for (const [nodeId, ups] of inputs) {
    inDegree.set(nodeId, ups.length);
    for (const up of ups) children.get(up)!.push(nodeId);
  }

  const ordered = new Set<string>();
  const ready = [...inDegree.entries()].filter(([, deg]) => deg === 0).map(([id]) => id);
  while (ready.length > 0) {
    const nodeId = ready.shift()!;
    ordered.add(nodeId);
    for (const child of children.get(nodeId)!) {
      const deg = inDegree.get(child)! - 1;
      inDegree.set(child, deg);
      if (deg === 0) ready.push(child);
    }
  }

  const stuck = new Set([...known].filter((id) => !ordered.has(id)));
  const issues: CycleIssue[] = [];
  while (stuck.size > 0) {
    // Drop nodes with no stuck successor: they sit downstream of a loop,
    // not inside one. Repeat until stable.
    let pruned = true;
    while (pruned) {
      pruned = false;
      for (const nodeId of [...stuck]) {
        if (!children.get(nodeId)!.some((child) => stuck.has(child))) {
          stuck.delete(nodeId);
          pruned = true;
        }
      }
    }
    if (stuck.size === 0) break;

    const loop = traceLoop(stuck, inputs);
    issues.push({ loopNodeIds: loop });
    for (const nodeId of loop) stuck.delete(nodeId);
  }
  return issues;
}

/** Follows inputs inside `stuck` until a node repeats, returning the loop. */
function traceLoop(stuck: Set<string>, inputs: Map<string, string[]>): string[] {
  const start = stuck.values().next().value!;
  const path = [start];
  const position = new Map<string, number>([[start, 0]]);
  let current = start;
  for (;;) {
    const next = inputs.get(current)!.find((up) => stuck.has(up))!;
    const seenAt = position.get(next);
    if (seenAt !== undefined) return path.slice(seenAt);
    position.set(next, path.length);
    path.push(next);
    current = next;
  }
}
