import { describe, expect, it } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import { getLeakageErrorMessage, graphSemanticSignature } from './leakageFeedback';

const nodes: Node[] = [{ id: 'scale', position: { x: 0, y: 0 }, data: { definitionType: 'scaler', config: { method: 'standard' } } }];
const edges: Edge[] = [{ id: 'edge', source: 'dataset', target: 'scale', sourceHandle: 'data', targetHandle: 'in' }];

describe('graphSemanticSignature', () => {
  /** Moving or selecting the canvas must not invalidate a server safety notice. */
  it('ignores canvas presentation state', () => {
    expect(graphSemanticSignature(nodes.map(node => ({ ...node,
      position: { x: 420, y: 170 }, selected: true, dragging: true, measured: { width: 200, height: 100 },
    })), edges.map(edge => ({ ...edge, selected: true })))).toBe(graphSemanticSignature(nodes, edges));
  });

  /** Editing fitted preprocessing must dismiss safety feedback tied to an older submission. */
  it('changes when nested configuration changes', () => {
    expect(graphSemanticSignature([{ ...nodes[0]!, data: { ...nodes[0]!.data, config: { method: 'minmax' } } }], edges))
      .not.toBe(graphSemanticSignature(nodes, edges));
  });

  /** Changing a source or destination port changes the submitted data path. */
  it.each(['sourceHandle', 'targetHandle', 'source', 'target'] as const)('includes edge %s', field => {
    expect(graphSemanticSignature(nodes, [{ ...edges[0]!, [field]: 'changed' }]))
      .not.toBe(graphSemanticSignature(nodes, edges));
  });
});

describe('getLeakageErrorMessage', () => {
  const detail = "Data leakage risk: node 'scale' fits on unsplit data.";

  /** Backend text must survive both intercepted Error objects and raw structured HTTP failures. */
  it.each([
    new Error(detail), detail, { response: { data: { detail } } },
    { response: { data: { detail } }, message: 'Request failed with status code 400' },
    { detail: { message: detail } },
  ])('preserves explicit safety detail from %j', error => {
    expect(getLeakageErrorMessage(error)).toBe(detail);
  });

  /** Unsupported fold reconstruction is a safety failure even without the word leakage. */
  it('recognizes backend fold-refit safety failures', () => {
    const message = 'Per-fold preprocessing refit skipped: payload reconstruction failed; CV/tuning scores may be optimistically biased.';
    expect(getLeakageErrorMessage(new Error(message))).toBe(message);
  });

  /** Ordinary runtime errors must keep their established results-panel route. */
  it.each([new Error('Network unavailable'), new Error('Estimator fit failed'), 'Leakage report endpoint timed out', null, {}])('does not classify unrelated errors as leakage: %j', error => {
    expect(getLeakageErrorMessage(error)).toBeNull();
  });

  /** A failed preview payload carries its useful safety message in node results. */
  it('extracts safety errors from a failed preview response', () => {
    expect(getLeakageErrorMessage({ status: 'failed', node_results: { scale: { status: 'failed', error: detail } } })).toBe(detail);
  });

  /** A mixed failure must not lose unrelated model errors by being classified as safety-only. */
  it('retains ordinary failure routing for mixed node errors', () => {
    expect(getLeakageErrorMessage({ status: 'failed', node_results: {
      scale: { status: 'failed', error: detail }, model: { status: 'failed', error: 'Estimator fit failed' },
    } })).toBeNull();
  });
});
