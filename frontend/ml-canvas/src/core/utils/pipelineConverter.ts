import { Node, Edge } from '@xyflow/react';
import { PipelineConfigModel, NodeConfigModel } from '../api/client';
import { v4 as uuidv4 } from 'uuid';
import { StepType as BackendStepType } from '../constants/stepTypes';
import { getMergeStrategy } from '../types/nodeData';
import { registry } from '../registry/NodeRegistry';

export const convertGraphToPipelineConfig = (nodes: Node[], edges: Edge[]): PipelineConfigModel => {
    const sortedNodes: NodeConfigModel[] = [];
    const visited = new Set<string>();
    const queue: string[] = [];

    // Collect ALL dataset nodes so disconnected subgraphs are included
    const datasetNodes = nodes.filter(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNodes[0]?.data.datasetId as string;

    for (const ds of datasetNodes) {
      queue.push(ds.id);
    }

    while (queue.length > 0) {
      const nodeId = queue.shift();
      if (!nodeId) continue;
      if (visited.has(nodeId)) continue;
      visited.add(nodeId);

      const node = nodes.find(n => n.id === nodeId);
      if (!node) continue;

      let stepType = 'unknown';
          let params: Record<string, unknown> = {};
      const incomingEdges = edges.filter(e => e.target === nodeId);
      const inputs = incomingEdges.map(e => e.source);

      if (node.data.definitionType === 'dataset_node') {
        stepType = BackendStepType.DATA_LOADER;
        params = { 
            dataset_id: node.data.datasetId,
        };
      } else if (node.data.definitionType === 'imputation_node') {
          const method = node.data.method || 'simple';
          if (method === 'knn') {
              stepType = 'KNNImputer';
              params = {
                  columns: node.data.columns,
                  n_neighbors: node.data.n_neighbors,
                  weights: node.data.weights
              };
          } else if (method === 'iterative') {
              stepType = 'IterativeImputer';
              params = {
                  columns: node.data.columns,
                  max_iter: node.data.max_iter,
                  estimator: node.data.estimator,
                  random_state: 0
              };
          } else {
              stepType = 'SimpleImputer';
              params = {
                  columns: node.data.columns,
                  strategy: node.data.strategy,
                  fill_value: node.data.fill_value
              };
          }
      } else if (node.data.definitionType === 'simple_imputer') {
          stepType = 'SimpleImputer';
          params = node.data || {};
      } else if (node.data.definitionType === 'drop_missing_columns' || node.data.definitionType === 'DropMissingColumns') {
          stepType = 'DropMissingColumns';
          params = {
            columns: node.data.columns || [],
            missing_threshold: node.data.missing_threshold
          };
      } else if (node.data.definitionType === 'drop_missing_rows' || node.data.definitionType === 'DropMissingRows') {
          stepType = 'DropMissingRows';
          params = {
            missing_threshold: node.data.missing_threshold,
            drop_if_any_missing: node.data.drop_if_any_missing
          };
      } else if (node.data.definitionType === 'deduplicate' || node.data.definitionType === 'Deduplicate') {
          stepType = 'Deduplicate';
          params = {
            subset: node.data.subset,
            keep: node.data.keep
          };
      } else if (node.data.definitionType === 'casting' || node.data.definitionType === 'Casting') {
          stepType = 'Casting';
          params = {
            column_types: node.data.column_types
          };
      } else if (node.data.definitionType === 'MissingIndicator' || node.data.definitionType === 'missing_indicator') {
          stepType = 'MissingIndicator';
          params = {
            columns: node.data.columns,
            flag_suffix: node.data.flag_suffix
          };
      } else if (node.data.definitionType === 'scale_numeric_features') {
                    const config: Record<string, unknown> =
                        (node.data && typeof node.data === 'object') ? (node.data as Record<string, unknown>) : {};
                    const method = config.method || 'standard';
          if (method === 'minmax') stepType = 'MinMaxScaler';
          else if (method === 'maxabs') stepType = 'MaxAbsScaler';
          else if (method === 'robust') stepType = 'RobustScaler';
          else stepType = 'StandardScaler';
          params = config;
      } else if (node.data.definitionType === 'encoding') {
          const method = node.data.method;
          if (method === 'onehot') stepType = 'OneHotEncoder';
          else if (method === 'dummy') {
              stepType = 'DummyEncoder';
              params = { ...node.data, drop_first: true };
          }
          else if (method === 'label') stepType = 'LabelEncoder';
          else if (method === 'ordinal') stepType = 'OrdinalEncoder';
          else if (method === 'target') stepType = 'TargetEncoder';
          else if (method === 'hash') stepType = 'HashEncoder';
          else stepType = 'OneHotEncoder'; // Default
          
          if (stepType !== 'DummyEncoder') {
              params = node.data;
          }
      } else if (node.data.definitionType === 'TrainTestSplitter') {
          stepType = 'TrainTestSplitter';
          params = node.data || {};
      } else if (node.data.definitionType === 'label_encoding') {
          stepType = 'LabelEncoder';
          params = node.data || {};
      } else if (node.data.definitionType === 'feature_target_split') {
          stepType = 'feature_target_split';
          params = node.data || {};
      }  else if (node.data.definitionType === 'feature_selection') {
          stepType = 'feature_selection';
          params = node.data || {};
      } else if (node.data.definitionType === 'outlier') {
          const method = node.data.method || 'iqr';
          if (method === 'iqr') stepType = 'IQR';
          else if (method === 'zscore') stepType = 'ZScore';
          else if (method === 'winsorize') stepType = 'Winsorize';
          else if (method === 'elliptic_envelope') stepType = 'EllipticEnvelope';
          else stepType = 'IQR';
          params = node.data;
      } else if (node.data.definitionType === 'TransformationNode') {
          stepType = 'GeneralTransformation';
          
          // Flatten transformations: { columns: ['a', 'b'], method: 'log' } -> [{ column: 'a', method: 'log' }, { column: 'b', method: 'log' }]
          const rawTransformations = (node.data.transformations || []) as unknown[];
          const flattenedTransformations = [];
          
          for (const r of rawTransformations) {
              const rule = r as Record<string, unknown>;
              if (rule.columns && Array.isArray(rule.columns)) {
                  for (const col of (rule.columns as string[])) {
                      flattenedTransformations.push({
                          column: col,
                          method: rule.method,
                          ...(rule.params as Record<string, unknown>)
                      });
                  }
              }
          }
          
          params = { transformations: flattenedTransformations };
      } else if (node.data.definitionType === 'BinningNode') {
          stepType = 'GeneralBinning';
          params = {
              columns: node.data.columns,
              strategy: node.data.strategy,
              n_bins: node.data.n_bins,
              label_format: node.data.label_format,
              output_suffix: node.data.output_suffix,
              drop_original: node.data.drop_original,
              custom_bins: node.data.custom_bins, // For custom strategy
              custom_labels: node.data.custom_labels // For custom strategy
          };
      } else if (node.data.definitionType === 'ResamplingNode') {
          const type = node.data.type || 'oversampling';
          if (type === 'oversampling') {
              stepType = 'Oversampling';
          } else {
              stepType = 'Undersampling';
          }
          params = node.data;
      } else if (node.data.definitionType === 'FeatureGenerationNode') {
          stepType = 'FeatureMath';
          params = {
              operations: node.data.operations
          };
      } else if (node.data.definitionType === 'PolynomialFeaturesNode') {
          stepType = 'PolynomialFeatures';
          params = {
              columns: node.data.columns,
              degree: node.data.degree,
              interaction_only: node.data.interaction_only,
              include_bias: node.data.include_bias,
              output_prefix: node.data.output_prefix,
              include_input_features: node.data.include_input_features
          };
      } else if (node.data.definitionType === 'TextCleaning') {
          stepType = 'TextCleaning';
          params = node.data;
      } else if (node.data.definitionType === 'ValueReplacement') {
          stepType = 'ValueReplacement';
          params = node.data;
      } else if (node.data.definitionType === 'AliasReplacement') {
          stepType = 'AliasReplacement';
          params = node.data;
      } else if (node.data.definitionType === 'InvalidValueReplacement') {
          stepType = 'InvalidValueReplacement';
          params = node.data;
      } else if (node.data.definitionType === 'model_training' || node.data.definitionType === BackendStepType.BASIC_TRAINING) {
          stepType = BackendStepType.BASIC_TRAINING;
          params = {
              target_column: node.data.target_column,
              model_type: node.data.model_type,
              hyperparameters: node.data.hyperparameters,
              cv_enabled: node.data.cv_enabled,
              cv_folds: node.data.cv_folds,
              cv_type: node.data.cv_type,
              cv_shuffle: node.data.cv_shuffle,
              cv_random_state: node.data.cv_random_state,
              cv_time_column: node.data.cv_time_column,
              execution_mode: node.data.execution_mode
          };
      } else if (node.data.definitionType === 'hyperparameter_tuning' || node.data.definitionType === BackendStepType.ADVANCED_TUNING) {
          stepType = BackendStepType.ADVANCED_TUNING;
          params = {
              target_column: node.data.target_column,
              algorithm: node.data.model_type,
              execution_mode: node.data.execution_mode,
              tuning_config: {
                  strategy: node.data.search_strategy,                    strategy_params: node.data.strategy_params,
                    metric: node.data.metric,
                  n_trials: node.data.n_trials,
                  search_space: node.data.search_space,
                  cv_enabled: node.data.cv_enabled,
                  cv_folds: node.data.cv_folds,
                  cv_type: node.data.cv_type,
                  cv_shuffle: node.data.cv_shuffle,
                  cv_random_state: node.data.cv_random_state,
                  cv_time_column: node.data.cv_time_column,
                  random_state: node.data.random_state
              }
          };
      } else if (node.data.definitionType === 'data_preview') {
          stepType = 'data_preview';
          params = {};
      } else {
          console.warn(`Unknown node type: ${node.data.definitionType}`);
          // Don't throw, just skip or use generic
          stepType = 'Unknown';
              params = (node.data && typeof node.data === 'object') ? (node.data as Record<string, unknown>) : {};
      }

      sortedNodes.push({
        node_id: node.id,
        step_type: stepType,
        params: (() => {
          // Attach per-node merge strategy (last_wins default) so the engine
          // can switch column-overlap semantics when the user requests it.
          // Kept under an underscore key to keep it separate from
          // step-specific params.
          const strat = getMergeStrategy(node.data);
          // Resolve the canvas-displayed label so the backend can use it
          // as the suffix for branch tabs (matches what the user sees on
          // the canvas — e.g. "Encoding" rather than the raw step type
          // "LabelEncoder").
          const defType = node.data.definitionType as string | undefined;
          const userLabel = (node.data.label as string | undefined) || (node.data.title as string | undefined);
          const registryLabel = defType ? registry.get(defType)?.label : undefined;
          const displayName = userLabel || registryLabel;
          const merged: Record<string, unknown> = { ...params };
          if (strat && strat !== 'last_wins') merged._merge_strategy = strat;
          if (displayName) merged._display_name = displayName;
          return merged;
        })(),
        inputs: inputs
      });

      const outgoingEdges = edges.filter(e => e.source === nodeId);
      outgoingEdges.forEach(e => queue.push(e.target));
    }

    // Prune dead-end branches: reverse-walk from terminal/seed nodes,
    // keep only ancestors. When no explicit terminals exist, infer from
    // graph leaves so parallel preview branches survive.
    const terminalTypes = new Set([
      BackendStepType.BASIC_TRAINING,
      BackendStepType.ADVANCED_TUNING,
      'data_preview',
    ]);
    let seeds = sortedNodes.filter(n => terminalTypes.has(n.step_type));

    // Always treat data leaves (no downstream consumer) as additional seeds.
    // This keeps preview-only branches alive when the canvas mixes a training
    // pipeline with one or more dangling preprocessing chains — without this,
    // Run Preview would silently drop the dangling branches and only show the
    // training-fed ones in the results tabs.
    if (sortedNodes.length > 1) {
      const consumed = new Set<string>();
      for (const node of sortedNodes) {
        for (const id of node.inputs) consumed.add(id);
      }
      const leaves = sortedNodes.filter(n => !consumed.has(n.node_id));
      // De-dupe: a node already in `seeds` (e.g. a training terminal) won't
      // be added twice because the reverse-BFS short-circuits on `reachable`.
      if (seeds.length === 0 && leaves.length > 1) {
        seeds = leaves;
      } else if (seeds.length > 0) {
        const seedIds = new Set(seeds.map(n => n.node_id));
        for (const leaf of leaves) {
          if (!seedIds.has(leaf.node_id)) {
            seeds.push(leaf);
            seedIds.add(leaf.node_id);
          }
        }
      }
    }

    let prunedNodes = sortedNodes;
    if (seeds.length > 0) {
      const reachable = new Set<string>();
      const reverseQueue: string[] = seeds.map(n => n.node_id);
      while (reverseQueue.length > 0) {
        const nid = reverseQueue.shift()!;
        if (reachable.has(nid)) continue;
        reachable.add(nid);
        const cfg = sortedNodes.find(n => n.node_id === nid);
        if (cfg?.inputs) {
          for (const inputId of cfg.inputs) {
            reverseQueue.push(inputId);
          }
        }
      }
      prunedNodes = sortedNodes.filter(n => reachable.has(n.node_id));
    }

    return {
      pipeline_id: `preview_${uuidv4()}`,
      nodes: prunedNodes,
      metadata: { dataset_source_id: datasetId }
    };
};

