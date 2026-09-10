import { getIncomers, type Edge, type Node } from '@xyflow/react';

/** Retain camel-case precedence over the legacy snake-case dataset field. */
function datasetInRecord(record: Record<string, unknown>): string | undefined {
  if (record.datasetId) return record.datasetId as string;
  if (record.dataset_id) return record.dataset_id as string;
  return undefined;
}

/** Read direct metadata first, then config and legacy params. */
function datasetInNode(node: Node): string | undefined {
  if (node.data?.datasetId) return node.data.datasetId as string;
  if (node.data?.dataset_id) return node.data.dataset_id as string;
  if (node.data?.config) {
    const datasetId = datasetInRecord(node.data.config as Record<string, unknown>);
    if (datasetId) return datasetId;
  }
  if (node.data?.params) return datasetInRecord(node.data.params as Record<string, unknown>);
  return undefined;
}

/** Find the nearest upstream dataset in graph order, tolerating cycles. */
export function findUpstreamDatasetId(currentNodeId: string, nodes: Node[], edges: Edge[]): string | undefined {
  const visited = new Set<string>();
  const queue = [currentNodeId];
  while (queue.length > 0) {
    const id = queue.shift();
    if (!id) continue;
    if (visited.has(id)) continue;
    visited.add(id);
    const node = nodes.find(n => n.id === id);
    if (!node) continue;
    if (id !== currentNodeId) {
      const datasetId = datasetInNode(node);
      if (datasetId) return datasetId;
    }
    const incomers = getIncomers(node, nodes, edges);
    for (const incomer of incomers) {
      queue.push(incomer.id);
    }
  }
  return undefined;
}
