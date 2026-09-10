import type { Node, Edge } from '@xyflow/react';
import type { PipelineConfigModel, NodeConfigModel } from '../api/client';
import { v4 as uuidv4 } from 'uuid';
import { getMergeStrategy } from '../types/nodeData';
import { registry } from '../registry/NodeRegistry';
import { convertUnknownNode, preprocessingConverters } from './pipelineConversion/preprocessing';
import { convertSegmentationNode, convertTrainingNode } from './pipelineConversion/training';
import { convertEnsembleNode } from './pipelineConversion/ensemble';
import { pruneToTerminalAncestors, removeSpecOnlyModels } from './pipelineConversion/graph';
import type { ConvertedNode, NodeConverter } from './pipelineConversion/types';

// Task-scoped supervised nodes share the canonical backend training dispatch.
const converters = new Map<string, NodeConverter>([
  ...preprocessingConverters,
  ['training', convertTrainingNode],
  ['classification', convertTrainingNode],
  ['regression', convertTrainingNode],
  ['text_classification', convertTrainingNode],
  ['SegmentationNode', convertSegmentationNode],
]);

function convertNode(node: Node, nodes: Node[], edges: Edge[]): ConvertedNode {
  const incomingEdges = edges.filter(edge => edge.target === node.id);
  // Split handles are visual edges from one logical backend input.
  const inputs = Array.from(new Set(incomingEdges.map(edge => edge.source)));
  if (node.data.definitionType === 'EnsembleNode') {
    return convertEnsembleNode(node, nodes, edges, incomingEdges, inputs);
  }
  const convert = converters.get(node.data.definitionType as string) ?? convertUnknownNode;
  return { ...convert(node), inputs };
}

function displayName(data: Record<string, unknown>): string | undefined {
  const userLabel = (data.label as string | undefined) || (data.title as string | undefined);
  const type = data.definitionType as string | undefined;
  // Preserve registry lookup even when a custom label takes precedence.
  const registryLabel = type ? registry.get(type)?.label : undefined;
  return userLabel || registryLabel;
}

/** Attach canvas labels and non-default column-overlap policy to backend parameters. */
function attachNodeMetadata(node: Node, params: Record<string, unknown>): Record<string, unknown> {
  const strategy = getMergeStrategy(node.data);
  const label = displayName(node.data);
  const merged = { ...params };
  if (strategy && strategy !== 'last_wins') merged._merge_strategy = strategy;
  if (label) merged._display_name = label;
  return merged;
}

/** Traverse all dataset-rooted branches and serialize their backend node configurations. */
export const convertGraphToPipelineConfig = (nodes: Node[], edges: Edge[]): PipelineConfigModel => {
  const datasets = nodes.filter(node => node.data.definitionType === 'dataset_node');
  const queue = datasets.map(node => node.id);
  const visited = new Set<string>();
  const configs: NodeConfigModel[] = [];
  while (queue.length > 0) {
    const id = queue.shift();
    if (!id || visited.has(id)) continue;
    visited.add(id);
    const node = nodes.find(candidate => candidate.id === id);
    if (!node) continue;
    const converted = convertNode(node, nodes, edges);
    configs.push({
      node_id: node.id,
      step_type: converted.stepType,
      params: attachNodeMetadata(node, converted.params),
      inputs: converted.inputs ?? [],
    });
    edges.filter(edge => edge.source === id).forEach(edge => queue.push(edge.target));
  }
  const executableNodes = removeSpecOnlyModels(configs, nodes, edges);
  return {
    pipeline_id: `preview_${uuidv4()}`,
    nodes: pruneToTerminalAncestors(executableNodes),
    metadata: { dataset_source_id: datasets[0]?.data.datasetId as string },
  };
};
