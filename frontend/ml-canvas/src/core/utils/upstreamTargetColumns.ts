import type { Edge, Node } from '@xyflow/react';

const targetSplitTypes = new Set(['feature_target_split', 'TrainTestSplitter', 'Split']);

/** Collect targets separated into y by actual upstream split nodes, including indirect inputs. */
export function upstreamTargetColumns(nodeId: string | undefined, nodes: Node[], edges: Edge[]): Set<string> {
  const targets = new Set<string>();
  const visited = new Set(nodeId ? [nodeId] : []);
  const pending = edges.filter(edge => edge.target === nodeId).map(edge => edge.source);
  while (pending.length) {
    const id = pending.pop()!;
    if (visited.has(id)) continue;
    visited.add(id);
    const data = nodes.find(node => node.id === id)?.data;
    if (targetSplitTypes.has(String(data?.definitionType)) && typeof data?.target_column === 'string' && data.target_column) {
      targets.add(data.target_column);
    }
    pending.push(...edges.filter(edge => edge.target === id).map(edge => edge.source));
  }
  return targets;
}
