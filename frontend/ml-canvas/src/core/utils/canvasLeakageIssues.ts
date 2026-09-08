import type { Edge } from '@xyflow/react';
import type { NodeConfigModel } from '../api/client';
import type { CanvasLeakageIssue } from '../types/leakage';
import { findPreprocessingBeforeSplitIssues, TRAIN_TEST_SPLIT_STEP_TYPES } from './pipelineLeakageValidation';

type CanvasEdge = Pick<Edge, 'id' | 'source' | 'target'>;

/** Highlight only edges on a path from the offending learner to its row boundary. */
function pathEdges(source: string, target: string, edges: readonly CanvasEdge[]): string[] {
  const ancestors = new Set([target]);
  const pending = [target];
  while (pending.length) {
    const id = pending.pop()!;
    for (const edge of edges) {
      if (edge.target === id && !ancestors.has(edge.source)) {
        ancestors.add(edge.source);
        pending.push(edge.source);
      }
    }
  }
  const reachable = new Set([source]);
  const queue = [source];
  while (queue.length) {
    const id = queue.pop()!;
    if (id === target) continue;
    for (const edge of edges) {
      if (edge.source === id && ancestors.has(edge.target) && !reachable.has(edge.target)) {
        reachable.add(edge.target);
        queue.push(edge.target);
      }
    }
  }
  return edges.filter(edge => edge.source !== target
    && reachable.has(edge.source) && reachable.has(edge.target) && ancestors.has(edge.target))
    .map(edge => edge.id);
}

/** Derive presentation-only diagnostics without changing graph nodes or saved data. */
export function buildCanvasLeakageIssues(
  nodes: NodeConfigModel[],
  edges: readonly CanvasEdge[],
  labels: ReadonlyMap<string, string> = new Map(),
): CanvasLeakageIssue[] {
  const byId = new Map(nodes.map(node => [node.node_id, node]));
  const issues: CanvasLeakageIssue[] = findPreprocessingBeforeSplitIssues(nodes).map(issue => {
    const node = byId.get(issue.nodeId)!;
    const name = labels.get(issue.nodeId) ?? issue.stepType;
    const splitName = labels.get(issue.splitterNodeId) ?? 'TrainTestSplitter';
    const rules: unknown = node.params.transformations;
    const methods = Array.isArray(rules) ? rules.flatMap(rule =>
      rule && typeof rule === 'object' && 'method' in rule && typeof rule.method === 'string'
        ? [rule.method] : []) : [];
    const operation = methods.length ? ` (${[...new Set(methods)].join(', ')})` : '';
    return {
      id: `before-split:${issue.nodeId}:${issue.splitterNodeId}`,
      nodeId: issue.nodeId,
      edgeIds: pathEdges(issue.nodeId, issue.splitterNodeId, edges),
      severity: 'error',
      message: `${name}${operation} learns from all rows before ${splitName}, including held-out rows.`,
      suggestion: `Move ${name} after ${splitName}. Fit on train, then reuse the fitted parameters on test and validation.`,
    };
  });
  if (!nodes.some(node => TRAIN_TEST_SPLIT_STEP_TYPES.has(node.step_type))) {
    for (const node of nodes) {
      if (!['training', 'tuning'].includes(node.step_type)) continue;
      issues.push({
        id: `no-split:${node.node_id}`,
        nodeId: node.node_id,
        edgeIds: edges.filter(edge => edge.target === node.node_id).map(edge => edge.id),
        severity: 'warning',
        message: 'No outer train/test row split is configured for this model.',
        suggestion: 'Add a TrainTestSplitter before learned preprocessing, or use raw-data per-fold cross-validation. CV does not provide an independent final test set.',
      });
    }
  }
  return issues;
}
