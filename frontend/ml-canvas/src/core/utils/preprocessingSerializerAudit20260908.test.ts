import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import type { PipelineConfigModel } from '../api/client';
import registryFixture from '../../../../../skyulf-core/tests/test_cases/leakage/registry_nodes.json';
import { ImputationNode } from '../../modules/nodes/processing/ImputationNode';
import { EncodingNode } from '../../modules/nodes/processing/EncodingNode';
import { TransformationNode } from '../../modules/nodes/processing/TransformationNode';
import { FeatureGenerationNode } from '../../modules/nodes/processing/FeatureGenerationNode';
import { PolynomialFeaturesNode } from '../../modules/nodes/processing/PolynomialFeaturesNode';
import { CastTypeNode } from '../../modules/nodes/processing/CastTypeNode';
import { TrainTestSplitNode } from '../../modules/nodes/modeling/TrainTestSplitNode';
import { FeatureTargetSplitNode } from '../../modules/nodes/modeling/FeatureTargetSplitNode';
import { convertGraphToPipelineConfig } from './pipelineConverter';
import {
  applyRegistryLeakageFlags,
  findPreprocessingBeforeSplitIssues,
  resetLeakageFlags,
} from './pipelineLeakageValidation';

interface CanvasRoute {
  definition: string;
  emitted: string;
  overrides?: Record<string, unknown>;
}

// Core IDs are not all canvas definition IDs. Record actual converter routes,
// rather than treating unsupported core-only nodes as successful UI executions.
const routes: Record<string, CanvasRoute> = {
  AliasReplacement: { definition: 'AliasReplacement', emitted: 'AliasReplacement' },
  Casting: { definition: 'casting', emitted: 'Casting' },
  DateFeatures: { definition: 'TimeSeriesNode', emitted: 'DateFeatures', overrides: { method: 'date' } },
  Deduplicate: { definition: 'deduplicate', emitted: 'Deduplicate' },
  DropMissingColumns: { definition: 'drop_missing_columns', emitted: 'DropMissingColumns' },
  DropMissingRows: { definition: 'drop_missing_rows', emitted: 'DropMissingRows' },
  DummyEncoder: { definition: 'encoding', emitted: 'DummyEncoder', overrides: { method: 'dummy' } },
  EllipticEnvelope: { definition: 'outlier', emitted: 'EllipticEnvelope', overrides: { method: 'elliptic_envelope' } },
  FeatureGenerationNode: { definition: 'FeatureGenerationNode', emitted: 'FeatureMath' },
  FeatureInteraction: { definition: 'FeatureInteractionNode', emitted: 'FeatureInteraction' },
  FeatureMath: { definition: 'FeatureGenerationNode', emitted: 'FeatureMath' },
  GeneralBinning: { definition: 'BinningNode', emitted: 'GeneralBinning' },
  GeneralTransformation: { definition: 'TransformationNode', emitted: 'GeneralTransformation' },
  HashEncoder: { definition: 'encoding', emitted: 'HashEncoder', overrides: { method: 'hash' } },
  IQR: { definition: 'outlier', emitted: 'IQR', overrides: { method: 'iqr' } },
  InvalidValueReplacement: { definition: 'InvalidValueReplacement', emitted: 'InvalidValueReplacement' },
  IterativeImputer: { definition: 'imputation_node', emitted: 'IterativeImputer', overrides: { method: 'iterative' } },
  KNNImputer: { definition: 'imputation_node', emitted: 'KNNImputer', overrides: { method: 'knn' } },
  LabelEncoder: { definition: 'encoding', emitted: 'LabelEncoder', overrides: { method: 'label' } },
  LagFeatures: { definition: 'TimeSeriesNode', emitted: 'LagFeatures', overrides: { method: 'lag' } },
  MaxAbsScaler: { definition: 'scale_numeric_features', emitted: 'MaxAbsScaler', overrides: { method: 'maxabs' } },
  MinMaxScaler: { definition: 'scale_numeric_features', emitted: 'MinMaxScaler', overrides: { method: 'minmax' } },
  MissingIndicator: { definition: 'MissingIndicator', emitted: 'MissingIndicator' },
  OneHotEncoder: { definition: 'encoding', emitted: 'OneHotEncoder', overrides: { method: 'onehot' } },
  OrdinalEncoder: { definition: 'encoding', emitted: 'OrdinalEncoder', overrides: { method: 'ordinal' } },
  Oversampling: { definition: 'ResamplingNode', emitted: 'Oversampling', overrides: { type: 'oversampling' } },
  PolynomialFeatures: { definition: 'PolynomialFeaturesNode', emitted: 'PolynomialFeatures' },
  PolynomialFeaturesNode: { definition: 'PolynomialFeaturesNode', emitted: 'PolynomialFeatures' },
  RobustScaler: { definition: 'scale_numeric_features', emitted: 'RobustScaler', overrides: { method: 'robust' } },
  RollingAggregate: { definition: 'TimeSeriesNode', emitted: 'RollingAggregate', overrides: { method: 'rolling' } },
  SimpleImputer: { definition: 'imputation_node', emitted: 'SimpleImputer', overrides: { method: 'simple' } },
  StandardScaler: { definition: 'scale_numeric_features', emitted: 'StandardScaler', overrides: { method: 'standard' } },
  TargetEncoder: { definition: 'encoding', emitted: 'TargetEncoder', overrides: { method: 'target' } },
  TextCleaning: { definition: 'TextCleaning', emitted: 'TextCleaning' },
  TrainTestSplitter: { definition: 'TrainTestSplitter', emitted: 'TrainTestSplitter' },
  Undersampling: { definition: 'ResamplingNode', emitted: 'Undersampling', overrides: { type: 'undersampling' } },
  ValueReplacement: { definition: 'ValueReplacement', emitted: 'ValueReplacement' },
  WOEEncoder: { definition: 'encoding', emitted: 'WOEEncoder', overrides: { method: 'woe' } },
  Winsorize: { definition: 'outlier', emitted: 'Winsorize', overrides: { method: 'winsorize' } },
  ZScore: { definition: 'outlier', emitted: 'ZScore', overrides: { method: 'zscore' } },
  count_vectorizer: { definition: 'count_vectorizer', emitted: 'count_vectorizer' },
  feature_selection: { definition: 'feature_selection', emitted: 'feature_selection' },
  feature_target_split: { definition: 'feature_target_split', emitted: 'feature_target_split' },
  hashing_vectorizer: { definition: 'hashing_vectorizer', emitted: 'hashing_vectorizer' },
  sentence_embedder: { definition: 'sentence_embedder', emitted: 'sentence_embedder' },
  tfidf_vectorizer: { definition: 'tfidf_vectorizer', emitted: 'tfidf_vectorizer' },
  tokenizer: { definition: 'tokenizer', emitted: 'tokenizer' },
};

/** Construct a real canvas shape, not an already-converted backend step. */
function canvasNode(id: string, definition: string, config: Record<string, unknown> = {}): Node {
  return { id, type: 'custom', position: { x: 0, y: 0 }, data: { ...config, definitionType: definition } };
}

/** Include the JSON wire round trip so undefined fields disappear as they do on submission. */
function serialize(definition: string, config: Record<string, unknown>): PipelineConfigModel {
  const nodes = [
    canvasNode('load', 'dataset_node', { datasetId: 'audit-dataset' }),
    canvasNode('operation', definition, config),
    canvasNode('split', TrainTestSplitNode.type, {
      ...TrainTestSplitNode.getDefaultConfig(), target_column: 'target',
    }),
    canvasNode('model', 'training', { model_type: 'logistic_regression', target_column: 'target' }),
  ];
  const edges: Edge[] = [
    { id: 'a', source: 'load', target: 'operation' },
    { id: 'b', source: 'operation', target: 'split' },
    { id: 'c', source: 'split', target: 'model' },
  ];
  return JSON.parse(JSON.stringify(convertGraphToPipelineConfig(nodes, edges))) as PipelineConfigModel;
}

const registeredCases = [
  ...Object.entries(registryFixture.learned_transformers).map(([id, params]) => ({ id, params, learned: true })),
  ...Object.entries(registryFixture.stateless_transformers).map(([id, params]) => ({ id, params, learned: false })),
  ...registryFixture.splitters.map(id => ({ id, params: {}, learned: false })),
];

describe.each(['bundled', 'registry'])('preprocessing serializer audit with %s metadata', (source) => {
  beforeEach(() => {
    resetLeakageFlags();
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    if (source === 'registry') {
      applyRegistryLeakageFlags([
        ...Object.keys(registryFixture.learned_transformers).map(id => ({ id, learns_from_data: true })),
        ...Object.keys(registryFixture.stateless_transformers).map(id => ({ id, learns_from_data: false })),
        ...registryFixture.splitters.map(id => ({ id, is_splitter: true })),
      ]);
    }
  });
  afterEach(() => {
    resetLeakageFlags();
    vi.restoreAllMocks();
  });

  /** The inventory must account for every registration without silently dropping core-only IDs. */
  it('covers all 62 registered IDs', () => {
    expect(new Set(registeredCases.map(testCase => testCase.id)).size).toBe(62);
  });

  /** Inspect emitted steps and gate decisions, including unsupported canvas spellings explicitly. */
  it.each(registeredCases)('serializes the registry example for $id', ({ id, params, learned }) => {
    const route = routes[id];
    const config: Record<string, unknown> = { ...params, ...route?.overrides };
    if (id === 'GeneralTransformation' && Array.isArray(config.transformations)) {
      config.transformations = config.transformations.map((rule: Record<string, unknown>) => {
        const { column, method, ...options } = rule;
        return { columns: [column], method, params: options };
      });
    }
    const pipeline = serialize(route?.definition ?? id, config);
    const operation = pipeline.nodes.find(node => node.node_id === 'operation');
    expect(operation?.step_type).toBe(route?.emitted ?? 'Unknown');
    if (!route) {
      // The backend rejects Unknown; this is not successful execution of a core-only node.
      expect(operation?.step_type).toBe('Unknown');
      return;
    }
    const polynomial = id === 'PolynomialFeatures' || id === 'PolynomialFeaturesNode';
    if (polynomial) expect(operation?.params).not.toHaveProperty('auto_detect');
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toHaveLength(learned && !polynomial ? 1 : 0);
  });

  /** Valid UI power rules remain learned after their selected columns are flattened. */
  it.each(['yeo-johnson', 'box-cox'] as const)('blocks serialized %s rules with standardize=false', (method) => {
    const config = {
      ...TransformationNode.getDefaultConfig(),
      transformations: [{ columns: ['x', 'other'], method, params: { standardize: false } }],
    };
    expect(TransformationNode.validate(config).isValid).toBe(true);
    const pipeline = serialize(TransformationNode.type, config);
    expect(pipeline.nodes.find(node => node.node_id === 'operation')?.params.transformations).toEqual([
      { column: 'x', method, standardize: false },
      { column: 'other', method, standardize: false },
    ]);
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toHaveLength(1);
  });

  /** Fixed math stays available, but a later learned rule cannot hide behind it. */
  it('distinguishes fixed and mixed transformation lists after serialization', () => {
    const fixed = { columns: ['x'], method: 'log' as const, params: {} };
    const learned = { columns: ['other'], method: 'yeo-johnson' as const, params: {} };
    expect(findPreprocessingBeforeSplitIssues(serialize(TransformationNode.type, {
      transformations: [fixed],
    }).nodes)).toEqual([]);
    expect(findPreprocessingBeforeSplitIssues(serialize(TransformationNode.type, {
      transformations: [fixed, learned],
    }).nodes)).toHaveLength(1);
  });

  /** The validator examines the final flattened method, including imported option overrides. */
  it('does not miss a learned method overriding the outer transformation rule', () => {
    const pipeline = serialize(TransformationNode.type, {
      transformations: [{ columns: ['x'], method: 'log', params: { method: 'yeo-johnson' } }],
    });
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toHaveLength(1);
  });

  /** A UI-valid group aggregation remains learned under its FeatureMath backend alias. */
  it('blocks a mixed feature-generation list emitted by the real definition', () => {
    const config = {
      ...FeatureGenerationNode.getDefaultConfig(),
      operations: [
        { operation_type: 'arithmetic' as const, method: 'add', input_columns: ['x'], constants: [1] },
        { operation_type: 'group_agg' as const, method: 'mean', input_columns: ['group'], secondary_columns: ['x'] },
      ],
    };
    expect(FeatureGenerationNode.validate(config).isValid).toBe(true);
    const pipeline = serialize(FeatureGenerationNode.type, config);
    expect(pipeline.nodes.find(node => node.node_id === 'operation')?.step_type).toBe('FeatureMath');
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toHaveLength(1);
  });

  /** Categorical casts preserve their vocabulary-learning configuration across the wire. */
  it.each(['category', 'float', 'string'])('classifies the actual %s cast payload', (targetType) => {
    const config = { ...CastTypeNode.getDefaultConfig(), column_types: { x: targetType } };
    expect(CastTypeNode.validate(config).isValid).toBe(true);
    const pipeline = serialize(CastTypeNode.type, config);
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toHaveLength(targetType === 'category' ? 1 : 0);
  });

  /** Encoder-local target hints cannot turn a real feature encoding into a target-only exemption. */
  it.each(['label', 'ordinal'] as const)('retains authoritative target context for UI %s encoding', (method) => {
    const config = { ...EncodingNode.getDefaultConfig(), method, columns: ['feature'], target_column: 'feature' };
    expect(EncodingNode.validate(config).isValid).toBe(true);
    expect(findPreprocessingBeforeSplitIssues(serialize(EncodingNode.type, config).nodes)).toHaveLength(1);
    expect(findPreprocessingBeforeSplitIssues(serialize(EncodingNode.type, {
      ...config, columns: ['target'],
    }).nodes)).toEqual([]);
  });

  /** Default empty Label/Ordinal selections remain explicit arrays, not feature auto-detection. */
  it.each(['label', 'ordinal'] as const)('preserves the real empty %s selection', (method) => {
    const config = { ...EncodingNode.getDefaultConfig(), method };
    expect(EncodingNode.validate(config).isValid).toBe(true);
    const pipeline = serialize(EncodingNode.type, config);
    expect(pipeline.nodes.find(node => node.node_id === 'operation')?.params.columns).toEqual([]);
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toEqual([]);
  });

  /** X/y column separation must not suppress a learned transform's downstream row-split diagnostic. */
  it('does not confuse Feature-Target Split with a row boundary', () => {
    const config = { ...FeatureTargetSplitNode.getDefaultConfig(), target_column: 'target' };
    const pipeline = serialize(FeatureTargetSplitNode.type, config);
    const xy = pipeline.nodes.find(node => node.node_id === 'operation')!;
    pipeline.nodes.push({ node_id: 'scale', step_type: 'StandardScaler', params: { columns: ['x'] }, inputs: [xy.node_id] });
    pipeline.nodes.find(node => node.node_id === 'split')!.inputs = ['scale'];
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toHaveLength(1);
  });

  /** The Polynomial editor's real default requires explicit columns and emits no automatic discovery. */
  it('keeps valid Polynomial UI configurations fixed', () => {
    const config = { ...PolynomialFeaturesNode.getDefaultConfig(), columns: ['x'] };
    expect(PolynomialFeaturesNode.validate(config).isValid).toBe(true);
    const pipeline = serialize(PolynomialFeaturesNode.type, config);
    expect(pipeline.nodes.find(node => node.node_id === 'operation')?.params).not.toHaveProperty('auto_detect');
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toEqual([]);
  });

  /** Mean/KNN/iterative UI modes preserve selected columns and remain gated before a row split. */
  it.each(['simple', 'knn', 'iterative'] as const)('blocks the real %s imputation payload', (method) => {
    const config = { ...ImputationNode.getDefaultConfig(), method, columns: ['x'] };
    expect(ImputationNode.validate(config).isValid).toBe(true);
    expect(findPreprocessingBeforeSplitIssues(serialize(ImputationNode.type, config).nodes)).toHaveLength(1);
  });

  /** Constant mode preserves its configured fill and admission; core/backend tests establish data independence. */
  it('preserves a UI-valid constant imputer and permits it before the row split', () => {
    const config = { ...ImputationNode.getDefaultConfig(), columns: ['x'], strategy: 'constant' as const };
    expect(ImputationNode.validate(config).isValid).toBe(true);
    const pipeline = serialize(ImputationNode.type, config);
    expect(pipeline.nodes.find(node => node.node_id === 'operation')?.params).toEqual({
      columns: ['x'], strategy: 'constant', fill_value: 0,
    });
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toEqual([]);
  });
});
