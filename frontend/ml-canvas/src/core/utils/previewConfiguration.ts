import type { Node, Edge } from '@xyflow/react';
import type { NodeConfigModel, PipelineConfigModel } from '../api/client';
import { convertGraphToPipelineConfig } from './pipelineConverter';
import { getPreviewGraph } from './previewGraph';

/** Inspection sinks have separate jobs and are excluded from data preview. */
export function buildPreviewConfiguration(nodes: Node[], edges: Edge[]): PipelineConfigModel {
  const graph = getPreviewGraph(nodes, edges);
  return convertGraphToPipelineConfig(graph.nodes, graph.edges);
}

/** Sort object keys without changing meaningful array/merge input order. */
function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
      .filter(([, entry]) => entry !== undefined).map(([key, entry]) => [key, canonicalValue(entry)]));
  }
  return value;
}

/** Identify execution settings while ignoring generated IDs and canvas presentation. */
export function previewConfigurationKey(config: PipelineConfigModel): string {
  return JSON.stringify(canonicalValue({
    nodes: config.nodes.map(node => ({ ...node, params: executionParams(node) }))
      .sort((a, b) => a.node_id.localeCompare(b.node_id)),
    metadata: config.metadata ?? {},
  }));
}

/** Omit known UI fields only where they occur, preserving nested column names. */
function executionParams(node: NodeConfigModel): Record<string, unknown> {
  const presentationKeys = new Set(['_display_name', 'label', 'title', 'description', 'definitionType', 'isExpanded']);
  const params = Object.fromEntries(Object.entries(node.params).filter(([key]) => !presentationKeys.has(key)));
  if (node.step_type === 'FeatureMath' && Array.isArray(params.operations)) {
    params.operations = params.operations.map((operation: unknown) => operation && typeof operation === 'object'
      ? Object.fromEntries(Object.entries(operation).filter(([key]) => key !== 'isExpanded')) : operation);
  }
  return params;
}
