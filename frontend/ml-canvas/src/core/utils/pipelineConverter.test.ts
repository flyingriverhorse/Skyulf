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
      node('train', 'classification', {
        run_mode: 'basic',
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
    expect(cfg.nodes[2]?.step_type).toBe('training');
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

  it('forwards the configured random_state for the iterative (MICE) imputer', () => {
    // Regression test: the converter previously hardcoded random_state: 0
    // for the IterativeImputer step regardless of the UI's random_state
    // field (ImputationNode.tsx), so the backend's random_state fix had no
    // user-visible effect until this was corrected here too.
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('iter', 'imputation_node', {
        method: 'iterative',
        columns: ['x'],
        max_iter: 15,
        estimator: 'BayesianRidge',
        random_state: 99,
      }),
    ];
    const edges = [edge('ds', 'iter')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const iter = cfg.nodes.find((n) => n.node_id === 'iter');
    expect(iter?.step_type).toBe('IterativeImputer');
    expect(iter?.params).toMatchObject({ max_iter: 15, estimator: 'BayesianRidge', random_state: 99 });
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

  it('forwards execution_mode through Classification node params (canonical training/fixed)', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('train', 'classification', {
        run_mode: 'basic',
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
    expect(train?.step_type).toBe('training');
    expect(train?.params).toMatchObject({
      run_mode: 'fixed',
      target_column: 'y',
      model_type: 'logistic_regression',
      execution_mode: 'parallel',
    });
  });

  it('emits a canonical training/tuned step for a Classification node with run_mode=advanced', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('tune', 'classification', {
        run_mode: 'advanced',
        target_column: 'y',
        model_type: 'logistic_regression',
        search_space: { C: [0.1, 1, 10] },
        search_strategy: 'random',
        metric: 'accuracy',
        n_trials: 5,
        cv_enabled: true,
        cv_folds: 3,
        cv_type: 'kfold',
        cv_shuffle: true,
        cv_random_state: 42,
      }),
    ];
    const edges = [edge('ds', 'tune')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const tune = cfg.nodes.find((n) => n.node_id === 'tune');
    expect(tune?.step_type).toBe('training');
    expect(tune?.params).toMatchObject({
      run_mode: 'tuned',
      target_column: 'y',
      algorithm: 'logistic_regression',
    });
    expect((tune?.params as { tuning_config?: Record<string, unknown> })?.tuning_config).toMatchObject({
      strategy: 'random',
      metric: 'accuracy',
      n_trials: 5,
      search_space: { C: [0.1, 1, 10] },
    });
  });

  it('emits a canonical training/tuned step for a Regression node with run_mode=advanced', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('tune', 'regression', {
        run_mode: 'advanced',
        target_column: 'y',
        model_type: 'random_forest_regressor',
        search_space: { n_estimators: [50, 100] },
        search_strategy: 'grid',
        metric: 'r2',
        n_trials: 8,
        cv_enabled: true,
        cv_folds: 5,
        cv_type: 'kfold',
        cv_shuffle: true,
        cv_random_state: 42,
      }),
    ];
    const edges = [edge('ds', 'tune')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const tune = cfg.nodes.find((n) => n.node_id === 'tune');
    expect(tune?.step_type).toBe('training');
    expect(tune?.params).toMatchObject({
      run_mode: 'tuned',
      target_column: 'y',
      algorithm: 'random_forest_regressor',
    });
    expect((tune?.params as { tuning_config?: Record<string, unknown> })?.tuning_config).toMatchObject({
      strategy: 'grid',
      metric: 'r2',
      n_trials: 8,
    });
  });

  it('emits a canonical training/fixed step for a TextClassification node with run_mode=basic', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('train', 'text_classification', {
        run_mode: 'basic',
        target_column: 'y',
        model_type: 'logistic_regression',
        hyperparameters: { C: 1 },
        cv_enabled: true,
        cv_folds: 3,
        cv_type: 'kfold',
        cv_shuffle: true,
        cv_random_state: 42,
      }),
    ];
    const edges = [edge('ds', 'train')];
    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const train = cfg.nodes.find((n) => n.node_id === 'train');
    expect(train?.step_type).toBe('training');
    expect(train?.params).toMatchObject({
      run_mode: 'fixed',
      target_column: 'y',
      model_type: 'logistic_regression',
      hyperparameters: { C: 1 },
    });
  });

  // Phase 3 Part B (plan §0.6): the generic TrainingNode and the 3
  // task-scoped nodes (Classification/Regression/Text Classification) all
  // share the same run_mode-keyed dispatch — only the model dropdown they
  // expose differs, the submitted step_type/params must be identical.
  it.each(['training', 'classification', 'regression', 'text_classification'])(
    '%s definitionType in basic run_mode submits a training step with run_mode=fixed',
    (definitionType) => {
      const nodes = [
        node('ds', 'dataset_node', { datasetId: 'd1' }),
        node('train', definitionType, {
          run_mode: 'basic',
          target_column: 'y',
          model_type: 'logistic_regression',
          hyperparameters: { C: 1 },
          cv_enabled: true,
          cv_folds: 3,
          cv_type: 'kfold',
          cv_shuffle: true,
          cv_random_state: 42,
        }),
      ];
      const edges = [edge('ds', 'train')];
      const cfg = convertGraphToPipelineConfig(nodes, edges);
      const train = cfg.nodes.find((n) => n.node_id === 'train');
      expect(train?.step_type).toBe('training');
      expect(train?.params).toMatchObject({
        run_mode: 'fixed',
        target_column: 'y',
        model_type: 'logistic_regression',
        hyperparameters: { C: 1 },
      });
    },
  );

  it.each(['training', 'classification', 'regression', 'text_classification'])(
    '%s definitionType in advanced run_mode submits a training step with run_mode=tuned',
    (definitionType) => {
      const nodes = [
        node('ds', 'dataset_node', { datasetId: 'd1' }),
        node('train', definitionType, {
          run_mode: 'advanced',
          target_column: 'y',
          model_type: 'logistic_regression',
          search_space: { C: [0.1, 1, 10] },
          search_strategy: 'random',
          metric: 'accuracy',
          n_trials: 5,
          cv_enabled: true,
          cv_folds: 3,
          cv_type: 'kfold',
          cv_shuffle: true,
          cv_random_state: 42,
        }),
      ];
      const edges = [edge('ds', 'train')];
      const cfg = convertGraphToPipelineConfig(nodes, edges);
      const train = cfg.nodes.find((n) => n.node_id === 'train');
      expect(train?.step_type).toBe('training');
      expect(train?.params).toMatchObject({
        run_mode: 'tuned',
        target_column: 'y',
        algorithm: 'logistic_regression',
      });
      expect((train?.params as { tuning_config?: Record<string, unknown> })?.tuning_config).toMatchObject({
        strategy: 'random',
        metric: 'accuracy',
        n_trials: 5,
        search_space: { C: [0.1, 1, 10] },
      });
    },
  );

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

  it('keeps dangling preprocessing leaves alongside training terminals', () => {
    // Two parallel chains: one ends at a training terminal, the other is a
    // dangling preprocessing leaf with no consumer. Both should survive so
    // Run Preview can show a tab per branch (training data + dangling
    // preprocessing output side-by-side).
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('imp', 'imputation_node', { method: 'simple', strategy: 'mean' }),
      node('train', 'classification', {
        run_mode: 'basic',
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
    // Dangling leaves are now intentionally kept so the backend can render
    // a preview tab for them. (Previously they were pruned, which silently
    // hid them from Run Preview when a training node was also present.)
    expect(ids).toContain('dangling');
  });

  it('keeps every parallel preview branch even when their chain lengths differ', () => {
    // Regression: previously the depth heuristic only kept the deepest leaves,
    // so a Run Preview without a training/preview terminal silently dropped
    // shorter branches and the UI showed only one path's data.
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      // Branch A: ds → impA (depth 1, leaf)
      node('impA', 'imputation_node', { method: 'simple', strategy: 'mean' }),
      // Branch B: ds → impB → scaleB (depth 2, leaf)
      node('impB', 'imputation_node', { method: 'simple', strategy: 'median' }),
      node('scaleB', 'StandardScaler', { columns: ['x'] }),
    ];
    const edges = [
      edge('ds', 'impA'),
      edge('ds', 'impB'),
      edge('impB', 'scaleB'),
    ];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ids = cfg.nodes.map((n) => n.node_id);
    // Both leaves (impA and scaleB) and their ancestors must survive so the
    // backend can split the preview into one tab per branch.
    expect(ids).toContain('impA');
    expect(ids).toContain('impB');
    expect(ids).toContain('scaleB');
    expect(ids).toContain('ds');
  });

  it('keeps dangling preview branches when a training terminal also exists', () => {
    // Regression: when the canvas mixes a training pipeline with a separate
    // preprocessing-only branch, the seed-from-terminals walk previously
    // pruned the dangling branch out of the request — Run Preview only
    // showed the training-fed data and the other branch silently vanished.
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      // Training branch: ds → impA → train
      node('impA', 'imputation_node', { method: 'simple', strategy: 'mean' }),
      node('train', 'classification', {
        run_mode: 'basic',
        target_column: 'y',
        model_type: 'rf',
        hyperparameters: {},
        cv_enabled: false,
        cv_folds: 5,
        cv_type: 'kfold',
        cv_shuffle: true,
        cv_random_state: 0,
      }),
      // Dangling preprocessing branch: ds → scaleB (no training downstream)
      node('scaleB', 'StandardScaler', { columns: ['x'] }),
    ];
    const edges = [
      edge('ds', 'impA'),
      edge('impA', 'train'),
      edge('ds', 'scaleB'),
    ];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ids = cfg.nodes.map((n) => n.node_id);
    // Both the training leaf and the dangling preprocessing leaf must
    // survive so the backend can produce a tab for each.
    expect(ids).toContain('train');
    expect(ids).toContain('impA');
    expect(ids).toContain('scaleB');
    expect(ids).toContain('ds');
  });
});

describe('convertGraphToPipelineConfig — ensemble wiring (Phase 2)', () => {
  it('derives base learners from connected model nodes and overrides the in-node selection', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      // Two model nodes feed the ensemble purely as base-learner specs.
      node('rf', 'classification', {
        target_column: 'y',
        model_type: 'random_forest_classifier',
        hyperparameters: { n_estimators: 200 },
      }),
      node('lr', 'classification', {
        target_column: 'y',
        model_type: 'logistic_regression',
        hyperparameters: { C: 0.5 },
      }),
      node('ens', 'EnsembleNode', {
        task: 'classification',
        target_column: 'y',
        model_type: 'voting_classifier',
        run_mode: 'basic',
        voting: 'soft',
        // In-node selection that should be OVERRIDDEN by the wired models.
        base_estimators: ['decision_tree'],
        base_estimator_params: {},
      }),
    ];
    const edges = [
      edge('ds', 'rf'),
      edge('ds', 'lr'),
      edge('ds', 'ens'),
      edge('rf', 'ens'),
      edge('lr', 'ens'),
    ];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ens = cfg.nodes.find((n) => n.node_id === 'ens');
    const hp = ens?.params.hyperparameters as Record<string, unknown>;
    expect(hp.base_estimators).toEqual(['random_forest', 'logistic_regression']);
    expect(hp.base_estimator_params).toMatchObject({
      random_forest: { n_estimators: 200 },
      logistic_regression: { C: 0.5 },
    });
    // Model-spec sources are stripped from the data inputs; only the dataset edge remains.
    expect(ens?.inputs).toEqual(['ds']);
    // The wired model nodes are spec providers, not standalone trainers.
    const ids = cfg.nodes.map((n) => n.node_id);
    expect(ids).not.toContain('rf');
    expect(ids).not.toContain('lr');
    expect(ids).toContain('ens');
  });

  it('falls back to the in-node base selection when no model nodes are wired', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('ens', 'EnsembleNode', {
        task: 'classification',
        target_column: 'y',
        model_type: 'voting_classifier',
        run_mode: 'basic',
        base_estimators: ['random_forest', 'svc'],
        base_estimator_params: { svc: { C: 2 } },
      }),
    ];
    const edges = [edge('ds', 'ens')];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ens = cfg.nodes.find((n) => n.node_id === 'ens');
    const hp = ens?.params.hyperparameters as Record<string, unknown>;
    expect(hp.base_estimators).toEqual(['random_forest', 'svc']);
    expect(hp.base_estimator_params).toMatchObject({ svc: { C: 2 } });
    expect(ens?.inputs).toEqual(['ds']);
  });

  it('threads wired base learners into advanced tuning_config', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('gb', 'regression', {
        target_column: 'y',
        model_type: 'gradient_boosting_regressor',
        hyperparameters: { learning_rate: 0.1 },
      }),
      node('ens', 'EnsembleNode', {
        task: 'regression',
        target_column: 'y',
        model_type: 'voting_regressor',
        run_mode: 'advanced',
        search_strategy: 'random',
        n_trials: 20,
        base_estimators: ['ridge'],
      }),
    ];
    const edges = [edge('ds', 'gb'), edge('ds', 'ens'), edge('gb', 'ens')];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ens = cfg.nodes.find((n) => n.node_id === 'ens');
    expect(ens?.step_type).toBe('training');
    expect(ens?.params.run_mode).toBe('tuned');
    const tuning = ens?.params.tuning_config as Record<string, unknown>;
    expect(tuning.base_estimators).toEqual(['gradient_boosting']);
    expect(tuning.base_estimator_params).toMatchObject({
      gradient_boosting: { learning_rate: 0.1 },
    });
    expect(ens?.inputs).toEqual(['ds']);
    expect(cfg.nodes.map((n) => n.node_id)).not.toContain('gb');
  });

  it('inherits the data source from wired models when the ensemble has no direct dataset edge', () => {
    // Common flow: split → model → ensemble, with NO direct split → ensemble edge.
    // The ensemble must still resolve a training dataset by inheriting the data
    // its base-learner models consume (base learners are spec-only).
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('split', 'TrainTestSplitter', {}),
      node('rf', 'classification', {
        target_column: 'y',
        model_type: 'random_forest_classifier',
        hyperparameters: {},
      }),
      node('lr', 'classification', {
        target_column: 'y',
        model_type: 'logistic_regression',
        hyperparameters: {},
      }),
      node('ens', 'EnsembleNode', {
        task: 'classification',
        target_column: 'y',
        model_type: 'voting_classifier',
        run_mode: 'basic',
        base_estimators: [],
      }),
    ];
    const edges = [
      edge('ds', 'split'),
      edge('split', 'rf'),
      edge('split', 'lr'),
      edge('rf', 'ens'),
      edge('lr', 'ens'),
    ];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ens = cfg.nodes.find((n) => n.node_id === 'ens');
    // Ensemble inherits the split as its data source (no direct edge existed).
    expect(ens?.inputs).toEqual(['split']);
    const hp = ens?.params.hyperparameters as Record<string, unknown>;
    expect(hp.base_estimators).toEqual(['random_forest', 'logistic_regression']);
    // Spec-only model nodes are dropped; the split is retained as the data root.
    const ids = cfg.nodes.map((n) => n.node_id);
    expect(ids).not.toContain('rf');
    expect(ids).not.toContain('lr');
    expect(ids).toContain('split');
    expect(ids).toContain('ens');
  });

  it('skips unsupported model types (no ensemble base key) gracefully', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      // calibrated_classifier is a meta-model with no ensemble base-key mapping
      // → skipped as a base learner.
      node('cal', 'classification', {
        target_column: 'y',
        model_type: 'calibrated_classifier',
        hyperparameters: {},
      }),
      node('rf', 'classification', {
        target_column: 'y',
        model_type: 'random_forest_classifier',
        hyperparameters: {},
      }),
      node('ens', 'EnsembleNode', {
        task: 'classification',
        target_column: 'y',
        model_type: 'voting_classifier',
        run_mode: 'basic',
        base_estimators: ['decision_tree'],
      }),
    ];
    const edges = [
      edge('ds', 'cal'),
      edge('ds', 'rf'),
      edge('ds', 'ens'),
      edge('cal', 'ens'),
      edge('rf', 'ens'),
    ];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ens = cfg.nodes.find((n) => n.node_id === 'ens');
    const hp = ens?.params.hyperparameters as Record<string, unknown>;
    // Only the supported model contributes a base key; the meta-model is dropped.
    expect(hp.base_estimators).toEqual(['random_forest']);
    // Both model sources are still excluded from the data inputs.
    expect(ens?.inputs).toEqual(['ds']);
  });

  it('emits training step_type with fixed run_mode for basic EnsembleNode', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('ens', 'EnsembleNode', {
        task: 'classification',
        target_column: 'y',
        model_type: 'voting_classifier',
        run_mode: 'basic',
        base_estimators: ['random_forest'],
        base_estimator_params: {},
        cv_enabled: false,
        cv_folds: 5,
        cv_type: 'kfold',
        cv_shuffle: true,
        cv_random_state: 0,
        voting: 'soft',
        execution_mode: 'parallel',
        n_jobs: -1,
      }),
    ];
    const edges = [edge('ds', 'ens')];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ens = cfg.nodes.find((n) => n.node_id === 'ens');
    expect(ens?.step_type).toBe('training');
    expect(ens?.params.run_mode).toBe('fixed');
  });

  it('emits training step_type with tuned run_mode for advanced EnsembleNode', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('ens', 'EnsembleNode', {
        task: 'classification',
        target_column: 'y',
        model_type: 'voting_classifier',
        run_mode: 'advanced',
        search_strategy: 'random',
        n_trials: 20,
        base_estimators: ['random_forest'],
        base_estimator_params: {},
        voting: 'soft',
        execution_mode: 'parallel',
        n_jobs: -1,
      }),
    ];
    const edges = [edge('ds', 'ens')];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const ens = cfg.nodes.find((n) => n.node_id === 'ens');
    expect(ens?.step_type).toBe('training');
    expect(ens?.params.run_mode).toBe('tuned');
  });
});

describe('convertGraphToPipelineConfig — SegmentationNode', () => {
  it('emits training step_type with fixed run_mode for SegmentationNode', () => {
    const nodes = [
      node('ds', 'dataset_node', { datasetId: 'd1' }),
      node('seg', 'SegmentationNode', {
        model_type: 'kmeans',
        hyperparameters: { n_clusters: 3 },
        cv_enabled: false,
        execution_mode: 'parallel',
        reference_column: undefined,
      }),
    ];
    const edges = [edge('ds', 'seg')];

    const cfg = convertGraphToPipelineConfig(nodes, edges);
    const seg = cfg.nodes.find((n) => n.node_id === 'seg');
    expect(seg?.step_type).toBe('training');
    expect(seg?.params.run_mode).toBe('fixed');
  });
});
