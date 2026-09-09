import { beforeEach, describe, expect, it } from 'vitest';
import type { NodeConfigModel } from '../api/client';
import { buildCanvasLeakageIssues } from './canvasLeakageIssues';
import { applyRegistryLeakageFlags, getLeakageFlagsRevision, resetLeakageFlags, subscribeLeakageFlags } from './pipelineLeakageValidation';

const node = (node_id: string, step_type: string, inputs: string[] = [], params: Record<string, unknown> = {}): NodeConfigModel =>
  ({ node_id, step_type, inputs, params });
const edge = (id: string, source: string, target: string) => ({ id, source, target });

describe('canvas leakage presentation', () => {
  beforeEach(resetLeakageFlags);

  it('marks the offending path, not a sibling branch or downstream model', () => {
    const nodes = [
      node('load', 'data_loader'),
      node('power', 'GeneralTransformation', ['load'], { transformations: [{ column: 'x', method: 'yeo-johnson' }] }),
      node('fixed', 'SimpleTransformation', ['power']),
      node('split', 'TrainTestSplitter', ['fixed']),
      node('other', 'SimpleTransformation', ['power']),
      node('train', 'training', ['split']),
    ];
    const edges = [edge('input', 'load', 'power'), edge('first', 'power', 'fixed'),
      edge('second', 'fixed', 'split'), edge('sibling', 'power', 'other'), edge('output', 'split', 'train')];
    const original = structuredClone({ nodes, edges });
    const issues = buildCanvasLeakageIssues(nodes, edges, new Map([['power', 'Transformation'], ['split', 'Holdout']]));
    expect(issues).toHaveLength(1);
    expect(issues[0]).toMatchObject({ nodeId: 'power', edgeIds: ['first', 'second'], severity: 'error' });
    expect(issues[0]?.message).toContain('yeo-johnson');
    expect(issues[0]?.suggestion).toContain('after Holdout');
    expect({ nodes, edges }).toEqual(original);
  });

  it('clears markers after a parameter switches to a fixed operation', () => {
    const nodes = [node('load', 'data_loader'),
      node('power', 'GeneralTransformation', ['load'], { transformations: [{ column: 'x', method: 'yeo-johnson' }] }),
      node('split', 'TrainTestSplitter', ['power'])];
    expect(buildCanvasLeakageIssues(nodes, [])).toHaveLength(1);
    nodes[1]!.params = { transformations: [{ column: 'x', method: 'log' }] };
    expect(buildCanvasLeakageIssues(nodes, [])).toEqual([]);
  });

  it('keeps an offending node marked when its canvas connection is unavailable', () => {
    const nodes = [node('scale', 'StandardScaler'), node('split', 'TrainTestSplitter', ['scale'])];
    expect(buildCanvasLeakageIssues(nodes, [])[0]).toMatchObject({ nodeId: 'scale', edgeIds: [], severity: 'error' });
  });

  it('uses an advisory, not a blocking error, for a model without an outer split', () => {
    const nodes = [node('load', 'data_loader'), node('model', 'training', ['load'])];
    expect(buildCanvasLeakageIssues(nodes, [edge('train-input', 'load', 'model')]))
      .toEqual([expect.objectContaining({ nodeId: 'model', severity: 'warning', edgeIds: ['train-input'] })]);
  });

  it('does not warn on an ordinary preprocessing-only preview', () => {
    expect(buildCanvasLeakageIssues([node('load', 'data_loader'), node('scale', 'StandardScaler', ['load'])], [])).toEqual([]);
  });

  it('refreshes live presentation consumers when registry flags change', () => {
    const revisions: number[] = [];
    const unsubscribe = subscribeLeakageFlags(() => revisions.push(getLeakageFlagsRevision()));
    const nodes = [node('step', 'CustomLearner'), node('split', 'TrainTestSplitter', ['step'])];
    expect(buildCanvasLeakageIssues(nodes, [])).toEqual([]);
    applyRegistryLeakageFlags([{ id: 'CustomLearner', learns_from_data: true }, { id: 'TrainTestSplitter', is_splitter: true }]);
    expect(buildCanvasLeakageIssues(nodes, [])).toHaveLength(1);
    expect(revisions).toHaveLength(1);
    unsubscribe();
    resetLeakageFlags();
    expect(revisions).toHaveLength(1);
  });
});
