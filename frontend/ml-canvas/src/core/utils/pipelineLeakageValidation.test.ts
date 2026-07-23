import { describe, it, expect } from 'vitest';
import type { NodeConfigModel } from '../api/client';
import {
  findPreprocessingBeforeSplitIssues,
  formatLeakageIssueMessage,
} from './pipelineLeakageValidation';

const node = (node_id: string, step_type: string, inputs: string[] = []): NodeConfigModel => ({
  node_id,
  step_type,
  params: {},
  inputs,
});

describe('findPreprocessingBeforeSplitIssues', () => {
  it('flags a scaler wired before the TrainTestSplitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('scale', 'StandardScaler', ['load']),
      node('split', 'TrainTestSplitter', ['scale']),
      node('model', 'LogisticRegression', ['split']),
    ];
    const issues = findPreprocessingBeforeSplitIssues(nodes);
    expect(issues).toEqual([{ nodeId: 'scale', stepType: 'StandardScaler', splitterNodeId: 'split' }]);
  });

  it('allows the same node moved after the split', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('split', 'TrainTestSplitter', ['load']),
      node('scale', 'StandardScaler', ['split']),
      node('model', 'LogisticRegression', ['scale']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('does not flag anything when there is no splitter in the graph', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('impute', 'SimpleImputer', ['load']),
      node('scale', 'StandardScaler', ['impute']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('does not treat feature_target_split as a train/test boundary', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('encode', 'OneHotEncoder', ['load']),
      node('split_xy', 'feature_target_split', ['encode']),
      node('model', 'LogisticRegression', ['split_xy']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('allows stateless/rule-based nodes before the splitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('bounds', 'ManualBounds', ['load']),
      node('hash', 'HashEncoder', ['bounds']),
      node('split', 'TrainTestSplitter', ['hash']),
      node('model', 'LogisticRegression', ['split']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('flags an indirect ancestor reached through intermediate stateless nodes', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('impute', 'SimpleImputer', ['load']),
      node('clean', 'ValueReplacement', ['impute']),
      node('split', 'TrainTestSplitter', ['clean']),
      node('model', 'LogisticRegression', ['split']),
    ];
    const issues = findPreprocessingBeforeSplitIssues(nodes);
    expect(issues.map((i) => i.nodeId)).toEqual(['impute']);
  });
});

describe('formatLeakageIssueMessage', () => {
  it('names the offending node, its step type, and the splitter', () => {
    const message = formatLeakageIssueMessage({
      nodeId: 'scale',
      stepType: 'StandardScaler',
      splitterNodeId: 'split',
    });
    expect(message).toContain("'scale'");
    expect(message).toContain('StandardScaler');
    expect(message).toContain("'split'");
  });
});
