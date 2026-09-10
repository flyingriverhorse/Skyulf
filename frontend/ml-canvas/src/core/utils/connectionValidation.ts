import type { Connection, Edge, Node } from '@xyflow/react';
import type { PortDefinition } from '../types/nodes';
import { registry } from '../registry/NodeRegistry';
import { splitOutputHandles } from './splitConnections';

/**
 * True if adding the edge source→target would create a cycle: either a
 * self-loop, or target can already reach source through existing edges.
 * Cyclic graphs cannot execute in order and die late with a cryptic
 * "Artifact not found" error, so they are rejected at connect time.
 */
export function wouldCreateCycle(edges: Edge[], source: string, target: string): boolean {
  if (source === target) return true;
  const stack = [target];
  const seen = new Set<string>();
  while (stack.length > 0) {
    const current = stack.pop()!;
    if (current === source) return true;
    if (seen.has(current)) continue;
    seen.add(current);
    for (const edge of edges) {
      if (edge.source === current) stack.push(edge.target);
    }
  }
  return false;
}

/** Node types whose output is a trained model (spec), not a DataFrame. */
export const MODEL_NODE_TYPES = [
  'classification',
  'regression',
  'text_classification',
  'SegmentationNode',
  'EnsembleNode',
];

/**
 * Training/model nodes are pipeline endpoints: the only node that may consume
 * a model output is EnsembleNode. Any other target is a wiring mistake,
 * regardless of which branch the target sits on.
 */
export function isModelEndpointViolation(sourceType: string, targetType: string): boolean {
  return MODEL_NODE_TYPES.includes(sourceType) && targetType !== 'EnsembleNode';
}

/** Toast wording for the two connect-time rejections (shared with FlowCanvas). */
export const CYCLE_CONNECTION_MESSAGE =
  'This connection would create a loop. Pipelines must flow in one direction — remove the backwards wire instead.';
export const MODEL_ENDPOINT_CONNECTION_MESSAGE =
  'Training nodes are the end of a pipeline: their output is a trained model, not data. Only an Ensemble can consume a model output — wire preprocessing or training from the dataset branch instead.';

/** Return the same actionable rejection for drag guidance, pickers, and graph mutations. */
export function connectionIssue(nodes: Node[], edges: Edge[], connection: Connection): string | null {
  const source = nodes.find(node => node.id === connection.source);
  const target = nodes.find(node => node.id === connection.target);
  if (!source || !target) return 'This node is no longer available. Choose an existing node.';
  if (source.connectable === false || target.connectable === false) return 'Connections are disabled for this node. Choose another node.';
  if (wouldCreateCycle(edges, source.id, target.id)) return CYCLE_CONNECTION_MESSAGE;
  const sourceType = String(source.data.definitionType);
  const targetType = String(target.data.definitionType);
  if (isModelEndpointViolation(sourceType, targetType)) return MODEL_ENDPOINT_CONNECTION_MESSAGE;
  return portConnectionIssue(source, target, sourceType, targetType, edges, connection);
}

/** Resolve registered ports before checking split, type and duplicate policies. */
function portConnectionIssue(
  source: Node, target: Node, sourceType: string, targetType: string,
  edges: Edge[], connection: Connection,
): string | null {
  const { outputs, inputs, output, input } = resolveConnectionPorts(sourceType, targetType, connection);
  if (!output || !input) return 'Choose an output and an input on registered nodes. Data sources have no input.';
  const splitHandles = splitOutputHandles(source);
  if (splitHandles.length && !splitHandles.includes(output.id)) {
    return 'Validation is disabled. Set Validation Size above zero, or connect from Train or Test.';
  }
  const typeIssue = portTypeIssue(targetType, output, input);
  if (typeIssue) return typeIssue;
  if (edges.some(edge => edge.source === source.id && edge.target === target.id &&
    (splitHandles.length > 0 || (edge.sourceHandle ?? outputs[0]?.id) === output.id) && (edge.targetHandle ?? inputs[0]?.id) === input.id)) {
    return 'These ports are already connected. Choose another input or remove the existing connection.';
  }
  return null;
}

/** Keep null-handle fallback distinct from an explicitly invalid handle. */
function resolveConnectionPorts(sourceType: string, targetType: string, connection: Connection) {
  const outputs = registry.get(sourceType)?.outputs ?? [];
  const inputs = registry.get(targetType)?.inputs ?? [];
  const output = connection.sourceHandle == null ? outputs[0] : outputs.find(port => port.id === connection.sourceHandle);
  const input = connection.targetHandle == null ? inputs[0] : inputs.find(port => port.id === connection.targetHandle);
  return { outputs, inputs, output, input };
}

/** Enforce data compatibility while allowing model specs into an ensemble. */
function portTypeIssue(targetType: string, output: PortDefinition, input: PortDefinition): string | null {
  // Ensemble's shared input accepts both datasets and model specs, including saved definitions.
  const ensembleModel = targetType === 'EnsembleNode' && output.type === 'model';
  if (!ensembleModel && output.type !== 'any' && input.type !== 'any' && output.type !== input.type) {
    return `${output.label} provides ${output.type}, but ${input.label} needs ${input.type}. Choose an input of the same kind.`;
  }
  return null;
}
