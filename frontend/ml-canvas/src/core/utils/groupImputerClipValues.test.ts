import { describe, expect, it } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import { ImputationNode } from '../../modules/nodes/processing/ImputationNode';
import { OutlierNode } from '../../modules/nodes/processing/OutlierNode';
import { TrainTestSplitNode } from '../../modules/nodes/modeling/TrainTestSplitNode';
import { convertGraphToPipelineConfig } from './pipelineConverter';
import { findPreprocessingBeforeSplitIssues } from './pipelineLeakageValidation';

/** Serialize one canvas node between a dataset and a split, as the run button does. */
function emitted(definition: string, config: Record<string, unknown>, beforeSplit = false) {
  const nodes: Node[] = [
    { id: 'load', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'd' } },
    { id: 'op', type: 'custom', position: { x: 0, y: 0 }, data: { ...config, definitionType: definition } },
    { id: 'split', type: 'custom', position: { x: 0, y: 0 }, data: { ...TrainTestSplitNode.getDefaultConfig(), target_column: 'target', definitionType: TrainTestSplitNode.type } },
  ];
  const edges: Edge[] = beforeSplit
    ? [{ id: 'a', source: 'load', target: 'op' }, { id: 'b', source: 'op', target: 'split' }]
    : [{ id: 'a', source: 'load', target: 'split' }, { id: 'b', source: 'split', target: 'op' }];
  const pipeline = JSON.parse(JSON.stringify(convertGraphToPipelineConfig(nodes, edges)));
  return { step: pipeline.nodes.find((node: { node_id: string }) => node.node_id === 'op'), nodes: pipeline.nodes };
}

describe('GroupImputer canvas route', () => {
  const group = { ...ImputationNode.getDefaultConfig(), method: 'group' as const, columns: ['employees'], group_by: 'industry', strategy: 'median' as const };

  it('sends only the group imputer parameters the backend understands', () => {
    // Inactive KNN/iterative/constant settings must not leak into the GroupImputer config.
    expect(emitted(ImputationNode.type, group).step).toMatchObject({
      step_type: 'GroupImputer',
      params: { columns: ['employees'], group_by: 'industry', strategy: 'median' },
    });
    expect(Object.keys(emitted(ImputationNode.type, group).step.params).sort()).toEqual(['columns', 'group_by', 'strategy']);
  });

  it('is gated as a learned step when placed before the split', () => {
    // Group statistics come from rows, so held-out rows must not shape them.
    expect(findPreprocessingBeforeSplitIssues(emitted(ImputationNode.type, group, true).nodes)).toHaveLength(1);
  });

  it.each([
    [{ group_by: '' }, 'group_by', 'Select the column that defines the groups'],
    [{ group_by: 'employees' }, 'group_by', 'The group column cannot also be filled'],
    [{ strategy: 'constant' as const }, 'strategy', 'Group imputation supports mean, median or most frequent'],
  ])('rejects %j before the run', (override, field, message) => {
    // Each mistake would otherwise fail on the server with a less direct message.
    expect(ImputationNode.validate({ ...group, ...override })).toEqual({ isValid: false, field, message });
  });

  it('accepts a complete group configuration', () => {
    // A valid group setup must not be blocked by the simple-imputer constant check.
    expect(ImputationNode.validate(group)).toEqual({ isValid: true });
  });
});

describe('ClipValues canvas route', () => {
  const clip = { ...OutlierNode.getDefaultConfig(), method: 'clip', columns: ['employees'], bounds: { employees: { lower: 0, upper: 500 }, unused: { lower: 1 } } };

  it('sends the bounds of the selected columns only', () => {
    // Deselected columns keep their UI settings but must not be clipped.
    expect(emitted(OutlierNode.type, clip).step).toEqual(expect.objectContaining({
      step_type: 'ClipValues',
      params: { bounds: { employees: { lower: 0, upper: 500 } } },
    }));
  });

  it('may run before the split because it learns nothing', () => {
    // Fixed business bounds are not computed from rows, unlike Winsorize.
    expect(findPreprocessingBeforeSplitIssues(emitted(OutlierNode.type, clip, true).nodes)).toEqual([]);
  });

  it('validates clip bounds like manual bounds', () => {
    // The same lower/upper rules apply; only the action differs.
    expect(OutlierNode.validate({ ...clip, bounds: { employees: { lower: 9, upper: 1 } } })).toMatchObject({ isValid: false, field: 'bounds.employees.upper' });
    expect(OutlierNode.validate(clip)).toEqual({ isValid: true });
  });

  it('previews as CLIP', () => {
    // The node body tells the user the rows are kept.
    expect(OutlierNode.bodyPreview?.({ method: 'clip', columns: ['employees'] })).toBe('CLIP · 1 col');
  });
});
