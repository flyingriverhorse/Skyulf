import { beforeEach, describe, expect, it } from 'vitest';
import type { NodeConfigModel } from '../api/client';
import { findPreprocessingBeforeSplitIssues, resetLeakageFlags } from './pipelineLeakageValidation';

const node = (
  node_id: string, step_type: string, inputs: string[] = [], params: Record<string, unknown> = {},
): NodeConfigModel => ({ node_id, step_type, inputs, params });

describe('canvas leakage boundary accuracy', () => {
  beforeEach(resetLeakageFlags);

  it('does not flag a learner already protected by the first row splitter', () => {
    const nodes = [
      node('load', 'data_loader'),
      node('first', 'TrainTestSplitter', ['load']),
      node('scale', 'StandardScaler', ['first']),
      node('second', 'Split', ['scale']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('does not borrow a target name from an unrelated sibling branch', () => {
    const nodes = [
      node('load', 'data_loader'),
      node('other_split', 'TrainTestSplitter', ['load'], { target_column: 'category' }),
      node('encode', 'OrdinalEncoder', ['load'], { columns: ['category'] }),
      node('split', 'TrainTestSplitter', ['encode'], { target_column: 'target' }),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([
      { nodeId: 'encode', stepType: 'OrdinalEncoder', splitterNodeId: 'split' },
    ]);
  });

  it('does not grant a target exemption when descendant targets conflict', () => {
    const nodes = [
      node('load', 'data_loader'),
      node('encode', 'OrdinalEncoder', ['load'], { columns: ['label_a'] }),
      node('first', 'TrainTestSplitter', ['encode'], { target_column: 'label_a' }),
      node('second', 'TrainTestSplitter', ['encode'], { target_column: 'label_b' }),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toHaveLength(1);
  });

  it('still flags a learned merge fed by an unprotected raw branch', () => {
    const nodes = [
      node('load', 'data_loader'),
      node('split_a', 'TrainTestSplitter', ['load']),
      node('fixed', 'SimpleTransformation', ['load']),
      node('scale', 'StandardScaler', ['split_a', 'fixed']),
      node('split_b', 'TrainTestSplitter', ['scale']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([
      { nodeId: 'scale', stepType: 'StandardScaler', splitterNodeId: 'split_b' },
    ]);
  });
});
