import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { NodeConfigModel } from '../api/client';
import {
  applyRegistryLeakageFlags,
  findPreprocessingBeforeSplitIssues,
  resetLeakageFlags,
} from './pipelineLeakageValidation';

const emptySelectionNoops = [
  'OneHotEncoder', 'DummyEncoder', 'TargetEncoder', 'WOEEncoder', 'PowerTransformer',
  'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'SimpleImputer',
  'KNNImputer', 'IterativeImputer', 'GeneralBinning', 'KBinsDiscretizer', 'CustomBinning',
  'IQR', 'ZScore', 'Winsorize', 'EllipticEnvelope',
];
const featureGenerationTypes = ['FeatureGeneration', 'FeatureMath', 'FeatureGenerationNode'];
const polynomialTypes = ['PolynomialFeatures', 'PolynomialFeaturesNode'];
const selectionIsAutomatic = [
  'MissingIndicator', 'VarianceThreshold', 'CorrelationThreshold', 'UnivariateSelection',
  'ModelBasedSelection', 'feature_selection', 'Oversampling', 'Undersampling',
];
const registeredTypes = [
  ...emptySelectionNoops, ...featureGenerationTypes, ...selectionIsAutomatic, ...polynomialTypes,
  'GeneralTransformation', 'Casting', 'LabelEncoder', 'OrdinalEncoder',
  'count_vectorizer', 'tfidf_vectorizer',
];

/** Exercise the real graph boundary rather than asserting classification constants. */
function expectBlocked(stepType: string, params: Record<string, unknown>, blocked: boolean): void {
  const nodes: NodeConfigModel[] = [
    { node_id: 'load', step_type: 'DataLoader', params: {}, inputs: [] },
    { node_id: 'operation', step_type: stepType, params, inputs: ['load'] },
    {
      node_id: 'split', step_type: 'TrainTestSplitter',
      params: { target_column: 'target' }, inputs: ['operation'],
    },
  ];
  expect(findPreprocessingBeforeSplitIssues(nodes)).toEqual(blocked
    ? [{ nodeId: 'operation', stepType, splitterNodeId: 'split' }]
    : []);
}

describe.each(['bundled', 'registry'])('operation-sensitive leakage with %s flags', (source) => {
  beforeEach(() => {
    resetLeakageFlags();
    if (source === 'registry') {
      applyRegistryLeakageFlags([
        ...registeredTypes.map(id => ({ id, learns_from_data: true })),
        { id: 'TrainTestSplitter', is_splitter: true, aliases: ['Split'] },
      ]);
    }
  });
  afterEach(resetLeakageFlags);

  /** Fixed formulas must remain usable even when the registry conservatively marks the node learned. */
  it.each(['log', 'sqrt', 'square_root', 'cube_root', 'reciprocal', 'square', 'exp', 'exponential'])(
    'allows the fixed %s transformation', (method) => {
      expectBlocked('GeneralTransformation', { transformations: [{ method, columns: ['value'] }] }, false);
    },
  );

  /** Every rule matters: appending a learned transform cannot hide behind a fixed first rule. */
  it.each(['yeo-johnson', 'box-cox'])('blocks %s alone and in mixed rules', (method) => {
    const learned = { method, columns: ['value'] };
    const fixed = { method: 'log', columns: ['other'] };
    for (const transformations of [[learned], [fixed, learned], [learned, fixed]]) {
      expectBlocked('GeneralTransformation', { transformations }, true);
    }
  });

  /** Unknown operations cannot accidentally gain a stateless exemption. */
  it.each([{ method: 'future_method' }, {}, null])('blocks an unknown transformation %j', (rule) => {
    expectBlocked('GeneralTransformation', { transformations: [rule] }, true);
  });

  /** An empty operation list performs no fit even when the node supports learned modes. */
  it.each([{}, { transformations: [] }, { transformations: null }])('allows empty transforms %j', (params) => {
    expectBlocked('GeneralTransformation', params, false);
  });

  /** All saved registration aliases must distinguish formulas from fitted group tables. */
  it.each(featureGenerationTypes)('classifies each operation for %s', (stepType) => {
    for (const operation_type of ['arithmetic', 'ratio', 'similarity', 'datetime_extract']) {
      expectBlocked(stepType, { operations: [{ operation_type }] }, false);
    }
    expectBlocked(stepType, { operations: [{}] }, false);
    expectBlocked(stepType, { operations: [] }, false);
    expectBlocked(stepType, {}, false);
    expectBlocked(stepType, { operations: [{ operation_type: 'group_agg' }] }, true);
    expectBlocked(stepType, {
      operations: [{ operation_type: 'ratio' }, { operation_type: 'group_agg' }],
    }, true);
    expectBlocked(stepType, { operations: [{ operation_type: 'unknown' }] }, true);
  });

  /** Only automatic column discovery learns; selecting columns explicitly preserves fixed math. */
  it.each(polynomialTypes)('distinguishes automatic column selection from fixed math for %s', (stepType) => {
    for (const params of [
      {},
      { columns: [] },
      { columns: ['value'] },
      { auto_detect: false },
      { auto_detect: false, columns: [] },
      { auto_detect: true, columns: ['value'] },
    ]) {
      expectBlocked(stepType, params, false);
    }
    expectBlocked(stepType, { auto_detect: true }, true);
    expectBlocked(stepType, { auto_detect: true, columns: [] }, true);
    // Null is not a valid calculator selection; malformed imported configs gain no exemption.
    expectBlocked(stepType, { auto_detect: true, columns: null }, true);
  });

  /** The separate interaction node does not inherit Polynomial's automatic-selection behavior. */
  it('keeps FeatureInteraction fixed before the split', () => {
    expectBlocked('FeatureInteraction', { columns: ['value'], degree: 2 }, false);
  });

  /** Explicit empty selections are real no-ops for these calculators only. */
  it.each(emptySelectionNoops)('allows an empty %s selection', (stepType) => {
    expectBlocked(stepType, { columns: [] }, false);
  });

  /** Empty selectors still inspect training values and must remain behind the split. */
  it.each(selectionIsAutomatic)('keeps an empty %s selection gated', (stepType) => {
    expectBlocked(stepType, { columns: [] }, true);
  });

  /** Ordinal's implicit categorical feature discovery is different from explicit target-only mode. */
  it.each([{}, { columns: null }])('blocks ordinal auto-detection %j', (params) => {
    expectBlocked('OrdinalEncoder', params, true);
  });

  /** Vocabulary nodes require explicit features; absent or exclusively target columns do no work. */
  it.each(['count_vectorizer', 'tfidf_vectorizer'])('resolves text selection for %s', (stepType) => {
    for (const params of [{}, { columns: null }, { columns: [] }, { columns: ['target'] }]) {
      expectBlocked(stepType, params, false);
    }
    expectBlocked(stepType, { columns: ['target', 'target'] }, false);
    expectBlocked(stepType, { columns: ['text'] }, true);
    expectBlocked(stepType, { columns: ['target', 'text'] }, true);
    expectBlocked(stepType, { columns: ['local_target'], target_column: 'local_target' }, true);
    expectBlocked(stepType, { columns: ['target'], target_column: 'local_target' }, false);
  });

  /** Label defaults never auto-detect feature columns, while explicit target picks remain safe. */
  it.each([{}, { columns: null }, { columns: [] }, { columns: ['target'] }])(
    'allows label target-only mode %j', (params) => {
      expectBlocked('LabelEncoder', params, false);
    },
  );

  /** The shared label exception must not exempt actual feature encodings. */
  it.each(['LabelEncoder', 'OrdinalEncoder'])('distinguishes explicit columns for %s', (stepType) => {
    expectBlocked(stepType, { columns: [] }, false);
    expectBlocked(stepType, { columns: ['target'] }, false);
    expectBlocked(stepType, { columns: ['feature'] }, true);
    expectBlocked(stepType, { columns: ['target', 'feature'] }, true);
  });

  /** Fixed edges are insufficient for safety if custom binning still discovers columns from values. */
  it.each([{}, { columns: null }, { columns: ['value'] }, { columns: [] }])(
    'classifies custom binning selection %j', (params) => {
      expectBlocked('CustomBinning', { ...params, bins: [0, 10, 20] }, !Array.isArray(params.columns));
    },
  );

  /** Categorical casts freeze a vocabulary; all other casts are fixed conversions. */
  it.each([
    { params: {}, blocked: false },
    { params: { column_types: { city: 'string', value: 'float' } }, blocked: false },
    { params: { column_types: { city: 'category' } }, blocked: true },
    { params: { column_types: { city: 'CATEGORICAL' } }, blocked: true },
    { params: { columns: ['city'], target_type: 'category' }, blocked: true },
    { params: { columns: ['__proto__'], target_type: 'category' }, blocked: true },
    { params: { columns: [], target_type: 'category' }, blocked: false },
    { params: { columns: [], column_types: { city: 'category' } }, blocked: true },
    {
      params: { column_types: { city: 'category' }, columns: ['city'], target_type: 'float' },
      blocked: false,
    },
    {
      params: { column_types: { city: 'float' }, columns: ['city'], target_type: 'categorical' },
      blocked: true,
    },
    {
      params: {
        column_types: { city: 'category', country: 'category' },
        columns: ['city'], target_type: 'float',
      },
      blocked: true,
    },
  ])('resolves casting overrides before checking vocabulary learning: %j', ({ params, blocked }) => {
    expectBlocked('Casting', params, blocked);
  });
});
