import { Node, Edge } from '@xyflow/react';
import { PipelineConfigModel, NodeConfigModel } from '../api/client';

export const convertGraphToPipelineConfig = (nodes: Node[], edges: Edge[]): PipelineConfigModel => {
    const sortedNodes: NodeConfigModel[] = [];
    const visited = new Set<string>();
    const queue: string[] = [];

    const startNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = startNode?.data.datasetId as string;

    if (startNode) {
      queue.push(startNode.id);
    }

    while (queue.length > 0) {
      const nodeId = queue.shift()!;
      if (visited.has(nodeId)) continue;
      visited.add(nodeId);

      const node = nodes.find(n => n.id === nodeId);
      if (!node) continue;

      let stepType = 'unknown';
      let params: any = {};
      const incomingEdges = edges.filter(e => e.target === nodeId);
      const inputs = incomingEdges.map(e => e.source);

      if (node.data.definitionType === 'dataset_node') {
        stepType = 'data_loader';
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
          const config = node.data as any || {};
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
          const rawTransformations = (node.data.transformations || []) as any[];
          const flattenedTransformations = [];
          
          for (const rule of rawTransformations) {
              if (rule.columns && Array.isArray(rule.columns)) {
                  for (const col of rule.columns) {
                      flattenedTransformations.push({
                          column: col,
                          method: rule.method,
                          ...rule.params
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
      } else if (node.data.definitionType === 'model_training') {
          stepType = 'model_training';
          params = {
              target_column: node.data.target_column,
              model_type: node.data.model_type,
              hyperparameters: node.data.hyperparameters,
              cv_enabled: node.data.cv_enabled,
              cv_folds: node.data.cv_folds,
              cv_type: node.data.cv_type,
              cv_shuffle: node.data.cv_shuffle,
              cv_random_state: node.data.cv_random_state
          };
      } else if (node.data.definitionType === 'hyperparameter_tuning') {
          stepType = 'model_tuning';
          params = {
              target_column: node.data.target_column,
              algorithm: node.data.model_type,
              tuning_config: {
                  strategy: node.data.search_strategy,
                  metric: node.data.metric,
                  n_trials: node.data.n_trials,
                  search_space: node.data.search_space,
                  cv_enabled: node.data.cv_enabled,
                  cv_folds: node.data.cv_folds,
                  cv_type: node.data.cv_type,
                  cv_shuffle: node.data.cv_shuffle,
                  cv_random_state: node.data.cv_random_state,
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
          params = node.data;
      }

      sortedNodes.push({
        node_id: node.id,
        step_type: stepType,
        params: params,
        inputs: inputs
      });

      const outgoingEdges = edges.filter(e => e.source === nodeId);
      outgoingEdges.forEach(e => queue.push(e.target));
    }

    return {
      pipeline_id: `preview_${Date.now()}`,
      nodes: sortedNodes,
      metadata: { dataset_source_id: datasetId }
    };
};
