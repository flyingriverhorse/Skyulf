import type { Edge, Node } from '@xyflow/react';
import { registry } from '../../registry/NodeRegistry';
import type { NodeConfigModel } from '../../api/client';
import type { GraphValidationIssue } from '../useGraphStore';
import { convertGraphToPipelineConfig } from '../../utils/pipelineConverter';
import { findCycleIssues } from '../../utils/pipelineCycleValidation';
import { findPreprocessingBeforeSplitIssues } from '../../utils/pipelineLeakageValidation';

const prettifyDefinitionType = (definitionType: string): string =>
  definitionType.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

const getNodeLabel = (node: Node): string => {
  const definition = registry.get(node.data.definitionType as string);
  const data = (node.data ?? {}) as Record<string, unknown>;
  return (
    (typeof data.label === 'string' && data.label) ||
    (typeof data.title === 'string' && data.title) ||
    definition?.label ||
    (typeof node.data.definitionType === 'string'
      ? prettifyDefinitionType(node.data.definitionType as string)
      : node.id)
  );
};

/** Collects the blocking validation issues for a canvas graph. */
export function collectGraphValidationIssues(nodes: Node[], edges: Edge[]): GraphValidationIssue[] {
  const previewNodeIds = new Set(
    nodes.filter((node) => node.data.definitionType === 'data_preview').map((node) => node.id),
  );
  const activeNodes = nodes.filter((node) => !previewNodeIds.has(node.id));
  const activeEdges = edges.filter((edge) => !previewNodeIds.has(edge.source) && !previewNodeIds.has(edge.target));

  const issues: GraphValidationIssue[] = [];

  for (const node of activeNodes) collectNodeIssues(node, activeEdges, issues);
  const pipelineConfig = convertGraphToPipelineConfig(activeNodes, activeEdges);
  collectLeakageIssues(pipelineConfig.nodes, activeNodes, issues);
  collectCycleIssues(pipelineConfig.nodes, activeNodes, issues);
  return issues;
}

/** Keep configuration and required-connection issues in node order. */
function collectNodeIssues(node: Node, activeEdges: Edge[], issues: GraphValidationIssue[]): void {
  const definition = registry.get(node.data.definitionType as string);
  const label = getNodeLabel(node);

  if (!definition) {
    issues.push({
      nodeId: node.id,
      nodeLabel: label,
      category: 'configuration',
      message: `Refresh or re-add ${label} so the canvas knows how to validate it.`,
    });
    return;
  }

  const validation = definition.validate(node.data);
  if (!validation.isValid) {
    issues.push({
      nodeId: node.id,
      nodeLabel: label,
      category: 'configuration',
      message: `Fix the ${label} settings before running preview${validation.message ? `: ${validation.message}` : '.'}`,
      ...(validation.field ? { field: validation.field } : {}),
    });
  }

  if (definition.inputs.length > 0) {
    const hasInput = activeEdges.some((edge) => edge.target === node.id);
    if (!hasInput) {
      issues.push({
        nodeId: node.id,
        nodeLabel: label,
        category: 'connection',
        message: `Connect an upstream node to ${label} before running preview.`,
      });
    }
  }

  collectDatasetOutputIssue(node, label, activeEdges, issues);
}

/** Only configured datasets require a downstream executable node. */
function collectDatasetOutputIssue(node: Node, label: string, activeEdges: Edge[], issues: GraphValidationIssue[]): void {
  if (node.data.definitionType === 'dataset_node') {
    const hasDatasetId = typeof (node.data as { datasetId?: unknown }).datasetId === 'string' &&
      Boolean((node.data as { datasetId?: string }).datasetId);
    const hasOutput = activeEdges.some((edge) => edge.source === node.id);
    if (hasDatasetId && !hasOutput) {
      issues.push({
        nodeId: node.id,
        nodeLabel: label,
        category: 'connection',
        message: 'Connect a downstream node to this Dataset before running preview.',
      });
    }
  }
}

/** Translate pipeline leakage findings back to canvas labels. */
function collectLeakageIssues(pipelineNodes: NodeConfigModel[], activeNodes: Node[], issues: GraphValidationIssue[]): void {
  const leakageIssues = findPreprocessingBeforeSplitIssues(pipelineNodes);
  for (const issue of leakageIssues) {
    const node = activeNodes.find((candidate) => candidate.id === issue.nodeId);
    const splitter = activeNodes.find((candidate) => candidate.id === issue.splitterNodeId);
    if (!node || !splitter) continue;
    issues.push({
      nodeId: node.id,
      nodeLabel: getNodeLabel(node),
      category: 'leakage',
      message: `Move ${getNodeLabel(node)} after ${getNodeLabel(splitter)} so it only fits on training data.`,
    });
  }
}

/** Append cycle findings after configuration, connection, and leakage issues. */
function collectCycleIssues(pipelineNodes: NodeConfigModel[], activeNodes: Node[], issues: GraphValidationIssue[]): void {
  const cycleIssues = findCycleIssues(pipelineNodes);
  for (const cycle of cycleIssues) {
    const loopNodes = cycle.loopNodeIds
      .map((id) => activeNodes.find((candidate) => candidate.id === id))
      .filter((candidate): candidate is Node => Boolean(candidate));
    if (loopNodes.length === 0) continue;
    issues.push({
      nodeId: loopNodes[0]!.id,
      nodeLabel: getNodeLabel(loopNodes[0]!),
      category: 'cycle',
      message:
        `Cycle detected: ${loopNodes.map((loopNode) => getNodeLabel(loopNode)).join(' -> ')} ` +
        'feed back into each other. Remove one of these connections so the pipeline flows in one direction.',
    });
  }
}
