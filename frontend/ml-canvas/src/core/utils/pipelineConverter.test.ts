import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { convertGraphToPipelineConfig } from './pipelineConverter';

const node = (id: string, definitionType: string, extra: Record<string, unknown> = {}): Node => ({
  id,
  type: 'custom',
  position: { x: 0, y: 0 },
  data: { definitionType, ...extra },
});

const edge = (source: string, target: string): Edge => ({
  id: `${source}-${target}`,
  source,
  target,
});

describe('convertGraphToPipelineConfig', () => {
  it('produces an empty-but-valid config for an empty graph', () => {
    const cfg = convertGraphToPipelineConfig([], []);
    expect(cfg.nodes).toEqual([]);
    expect(cfg.metadata?.dataset_source_id).toBeUndefined();
    expect(cfg.pipeline_id).toMatch(/^preview_/);
  });

  it('emits a DataLoader step for a dataset_node and threads the dataset id into metadata', () => {
    const nodes = [node('ds', 'dataset_node', { datasetId: 'abc-123' })];
    const cfg = convertGraphToPipelineConfig(nodes, []);
    expect(cfg.metadata?.dataset_source_id).toBe('abc-123');
    // Single-node graphs without a downstream terminal still emit the loader.
    expect(cfg.nodes).toHaveLength(1);
    expect(cfg.nodes[0]?.step_type).toBe('data_loader');
    expect(cfg.nodes[0]?.params).toMatchObject({ dataset_id: 'abc-123' });
  });

  it('topologically orders nodes from dataset → preprocess → terminal', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('imp', 'imputation_node', { method: 'simple', strategy: 'mean', columns: ['x'] }),
      node('train', 'basic_training', {
        target_column: 'y',
        model_type: 'random_forest_classifier',
        hyperparameters: {},
        cv_enabled: false,
        cv_folds: 5,
        cv_type: 'kfold',
        cv_shuffle: true,
        cv_random_state: 0,
      }),
    ];
    const edges = [edge('ds', 'imp'), edge('imp', 'train')];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ids = cfg.nodes.map((n) => n.node_id);
    expect(ids).toEqual(['ds', 'imp', 'train']);
    expect(cfg.nodes[1]?.step_type).toBe('SimpleImputer');
    expect(cfg.nodes[2]?.step_type).toBe('basic_training');
    expect(cfg.nodes[2]?.inputs).toEqual(['imp']);
  });

  it('routes imputation method to the right step_type and forwards key params', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('knn', 'imputation_node', {
        method: 'knn',
        columns: ['x'],
        n_neighbors: 7,
        weights: 'distance',
      }),
    ];
    const edges = [edge('ds', 'knn')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const knn = cfg.nodes.find((n) => n.node_id === 'knn');
    expect(knn?.step_type).toBe('KNNImputer');
    expect(knn?.params).toMatchObject({ n_neighbors: 7, weights: 'distance' });
  });

  it('attaches _merge_strategy as an underscore-prefixed param when not "last_wins"', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('imp', 'imputation_node', {
        method: 'simple',
        strategy: 'mean',
        merge_strategy: 'first_wins',
      }),
    ];
    const edges = [edge('ds', 'imp')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const imp = cfg.nodes.find((n) => n.node_id === 'imp');
    expect(imp?.params._merge_strategy).toBe('first_wins');
  });

  it('omits _merge_strategy when it equals the engine default "last_wins"', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('imp', 'imputation_node', {
        method: 'simple',
        strategy: 'mean',
        merge_strategy: 'last_wins',
      }),
    ];
    const edges = [edge('ds', 'imp')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const imp = cfg.nodes.find((n) => n.node_id === 'imp');
    expect(imp?.params).not.toHaveProperty('_merge_strategy');
  });

  it('forwards execution_mode through basic_training params', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('train', 'basic_training', {
        target_column: 'y',
        model_type: 'logistic_regression',
        hyperparameters: { C: 1 },
        cv_enabled: true,
        cv_folds: 3,
        cv_type: 'kfold',
        cv_shuffle: true,
        cv_random_state: 42,
        execution_mode: 'parallel',
      }),
    ];
    const edges = [edge('ds', 'train')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const train = cfg.nodes.find((n) => n.node_id === 'train');
    expect(train?.step_type).toBe('basic_training');
    expect(train?.params).toMatchObject({
      target_column: 'y',
      model_type: 'logistic_regression',
      execution_mode: 'parallel',
    });
  });

  it('emits a data_preview step with empty params', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('preview', 'data_preview'),
    ];
    const edges = [edge('ds', 'preview')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const preview = cfg.nodes.find((n) => n.node_id === 'preview');
    expect(preview?.step_type).toBe('data_preview');
    expect(preview?.params).toEqual({});
  });

  it('prunes dead-end branches that don\'t feed a terminal', () => {
    // Two parallel chains: one ends at a training terminal, the other dangles.
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('imp', 'imputation_node', { method: 'simple', strategy: 'mean' }),
      node('train', 'basic_training', {
        target_column: 'y',
        model_type: 'rf',
        hyperparameters: {},
        cv_enabled: false,
        cv_folds: 5,
        cv_type: 'kfold',
        cv_shuffle: true,
        cv_random_state: 0,
      }),
      node('dangling', 'imputation_node', { method: 'simple', strategy: 'median' }),
    ];
    const edges = [edge('ds', 'imp'), edge('imp', 'train'), edge('ds', 'dangling')];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ids = cfg.nodes.map((n) => n.node_id);
    expect(ids).toContain('train');
    expect(ids).toContain('imp');
    expect(ids).toContain('ds');
    // Pruning kicks in via the terminalTypes seed walk and removes the
    // disconnected branch.
    expect(ids).not.toContain('dangling');
  });
});
