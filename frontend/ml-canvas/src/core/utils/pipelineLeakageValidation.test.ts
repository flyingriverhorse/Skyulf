import { describe, it, expect, afterEach } from 'vitest';
import type { NodeConfigModel } from '../api/client';
import {
  DATA_DEPENDENT_FIT_STEP_TYPES,
  TRAIN_TEST_SPLIT_STEP_TYPES,
  applyRegistryLeakageFlags,
  findPreprocessingBeforeSplitIssues,
  formatLeakageIssueMessage,
  resetLeakageFlags,
} from './pipelineLeakageValidation';

const node = (
  node_id: string,
  step_type: string,
  inputs: string[] = [],
  params: Record<string, unknown> = {},
): NodeConfigModel => ({
  node_id,
  step_type,
  params,
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
      node('split', 'TrainTestSplitter', ['bounds']),
      node('model', 'LogisticRegression', ['split']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it.each([
    // Keyless HashEncoder auto-detects its columns from the rows (gated);
    // an explicit column list makes it stateless, see the exemptions below.
    ['HashEncoder', { n_features: 8 }],
    // Keyless MissingIndicator auto-detects which columns carry missing
    // values (gated); an explicit list is exempt, see below.
    ['MissingIndicator', {}],
    // Param-less DropMissingColumns is the explicit/no-op mode (exempt);
    // its data-dependent mode is the positive missing-% threshold.
    ['DropMissingColumns', { missing_threshold: 50 }],
    ['Deduplicate', {}],
    ['Oversampling', {}],
    ['Undersampling', {}],
  ])('flags reclassified stateful node %s before the splitter', (stepType, params) => {
    const nodes = [
      node('load', 'DataLoader'),
      node('step', stepType, ['load'], params),
      node('split', 'TrainTestSplitter', ['step']),
      node('model', 'LogisticRegression', ['split']),
    ];
    const issues = findPreprocessingBeforeSplitIssues(nodes);
    expect(issues.map((i) => i.nodeId)).toEqual(['step']);
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

  it.each(['LabelEncoder', 'OrdinalEncoder'])(
    'allows target-only %s (no columns selected) before the splitter',
    (stepType) => {
      const nodes = [
        node('load', 'DataLoader'),
        node('encode_target', stepType, ['load'], {}),
        node('split', 'TrainTestSplitter', ['encode_target']),
        node('model', 'LogisticRegression', ['split']),
      ];
      expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
    },
  );

  it.each(['LabelEncoder', 'OrdinalEncoder'])(
    'still flags %s with explicit feature columns before the splitter',
    (stepType) => {
      const nodes = [
        node('load', 'DataLoader'),
        node('encode_features', stepType, ['load'], { columns: ['city', 'country'] }),
        node('split', 'TrainTestSplitter', ['encode_features']),
        node('model', 'LogisticRegression', ['split']),
      ];
      const issues = findPreprocessingBeforeSplitIssues(nodes);
      expect(issues.map((i) => i.nodeId)).toEqual(['encode_features']);
    },
  );

  it.each(['LabelEncoder', 'OrdinalEncoder'])(
    'allows %s with columns == [target_column] (explicit target pick) before the splitter',
    (stepType) => {
      const nodes = [
        node('load', 'DataLoader'),
        node('encode_target', stepType, ['load'], { columns: ['species'] }),
        node('split', 'TrainTestSplitter', ['encode_target'], { target_column: 'species' }),
        node('model', 'LogisticRegression', ['split']),
      ];
      expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
    },
  );

  it.each(['LabelEncoder', 'OrdinalEncoder'])(
    'still flags %s mixing the target column with real feature columns',
    (stepType) => {
      const nodes = [
        node('load', 'DataLoader'),
        node('encode_mixed', stepType, ['load'], { columns: ['species', 'city'] }),
        node('split', 'TrainTestSplitter', ['encode_mixed'], { target_column: 'species' }),
        node('model', 'LogisticRegression', ['split']),
      ];
      const issues = findPreprocessingBeforeSplitIssues(nodes);
      expect(issues.map((i) => i.nodeId)).toEqual(['encode_mixed']);
    },
  );

  it('allows an explicit named-column drop before the splitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('drop_id', 'DropMissingColumns', ['load'], {
        columns: ['passenger_id'],
        missing_threshold: 0,
      }),
      node('split', 'TrainTestSplitter', ['drop_id']),
      node('model', 'LogisticRegression', ['split']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('still flags a missing-percentage drop before the splitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('drop_sparse', 'DropMissingColumns', ['load'], { missing_threshold: 50 }),
      node('split', 'TrainTestSplitter', ['drop_sparse']),
      node('model', 'LogisticRegression', ['split']),
    ];
    const issues = findPreprocessingBeforeSplitIssues(nodes);
    expect(issues.map((i) => i.nodeId)).toEqual(['drop_sparse']);
  });

  it('allows a constant-strategy imputer before the splitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('fill', 'SimpleImputer', ['load'], { strategy: 'constant', fill_value: 0 }),
      node('split', 'TrainTestSplitter', ['fill']),
      node('model', 'LogisticRegression', ['split']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it.each(['mean', 'median', 'most_frequent'])(
    'still flags a %s-strategy imputer before the splitter',
    (strategy) => {
      const nodes = [
        node('load', 'DataLoader'),
        node('fill', 'SimpleImputer', ['load'], { strategy }),
        node('split', 'TrainTestSplitter', ['fill']),
        node('model', 'LogisticRegression', ['split']),
      ];
      const issues = findPreprocessingBeforeSplitIssues(nodes);
      expect(issues.map((i) => i.nodeId)).toEqual(['fill']);
    },
  );

  it('allows a MissingIndicator with explicitly named columns before the splitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('flags', 'MissingIndicator', ['load'], { columns: ['age', 'fare'] }),
      node('split', 'TrainTestSplitter', ['flags']),
      node('model', 'LogisticRegression', ['split']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('still flags a MissingIndicator with an empty column list before the splitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('flags', 'MissingIndicator', ['load'], { columns: [] }),
      node('split', 'TrainTestSplitter', ['flags']),
      node('model', 'LogisticRegression', ['split']),
    ];
    const issues = findPreprocessingBeforeSplitIssues(nodes);
    expect(issues.map((i) => i.nodeId)).toEqual(['flags']);
  });

  it('allows a HashEncoder with explicitly named columns before the splitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('hash', 'HashEncoder', ['load'], { columns: ['city'], n_features: 8 }),
      node('split', 'TrainTestSplitter', ['hash']),
      node('model', 'LogisticRegression', ['split']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('allows a HashEncoder with the empty-selection no-op before the splitter', () => {
    const nodes = [
      node('load', 'DataLoader'),
      node('hash', 'HashEncoder', ['load'], { columns: [] }),
      node('split', 'TrainTestSplitter', ['hash']),
      node('model', 'LogisticRegression', ['split']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
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

describe('applyRegistryLeakageFlags', () => {
  afterEach(() => {
    resetLeakageFlags();
  });

  it('overrides the bundled lists with registry-provided flags', () => {
    applyRegistryLeakageFlags([
      { id: 'StandardScaler', learns_from_data: true, is_splitter: false },
      // 'SimpleImputer' is in the bundled fallback but NOT flagged here.
      { id: 'SimpleImputer', learns_from_data: false, is_splitter: false },
      { id: 'BrandNewNode', learns_from_data: true, is_splitter: false },
      { id: 'TrainTestSplitter', learns_from_data: false, is_splitter: true, aliases: ['Split'] },
    ]);

    expect(DATA_DEPENDENT_FIT_STEP_TYPES.has('StandardScaler')).toBe(true);
    expect(DATA_DEPENDENT_FIT_STEP_TYPES.has('BrandNewNode')).toBe(true);
    expect(DATA_DEPENDENT_FIT_STEP_TYPES.has('SimpleImputer')).toBe(false);
    expect(TRAIN_TEST_SPLIT_STEP_TYPES.has('TrainTestSplitter')).toBe(true);
    // Alias spellings used by saved graphs stay gated.
    expect(TRAIN_TEST_SPLIT_STEP_TYPES.has('Split')).toBe(true);
  });

  it('makes the graph check follow the overridden lists', () => {
    applyRegistryLeakageFlags([
      { id: 'SimpleImputer', learns_from_data: false, is_splitter: false },
      { id: 'TrainTestSplitter', learns_from_data: false, is_splitter: true },
    ]);
    const nodes = [
      node('load', 'DataLoader'),
      node('impute', 'SimpleImputer', ['load']),
      node('split', 'TrainTestSplitter', ['impute']),
    ];
    expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual([]);
  });

  it('keeps the bundled fallback when the registry payload is empty', () => {
    applyRegistryLeakageFlags([]);
    expect(DATA_DEPENDENT_FIT_STEP_TYPES.has('StandardScaler')).toBe(true);
    expect(TRAIN_TEST_SPLIT_STEP_TYPES.has('TrainTestSplitter')).toBe(true);
  });

  it('resetLeakageFlags restores the bundled fallback', () => {
    applyRegistryLeakageFlags([
      { id: 'TrainTestSplitter', learns_from_data: false, is_splitter: true },
    ]);
    expect(DATA_DEPENDENT_FIT_STEP_TYPES.size).toBe(0);
    resetLeakageFlags();
    expect(DATA_DEPENDENT_FIT_STEP_TYPES.has('SimpleImputer')).toBe(true);
    expect(TRAIN_TEST_SPLIT_STEP_TYPES.has('Split')).toBe(true);
  });
});
