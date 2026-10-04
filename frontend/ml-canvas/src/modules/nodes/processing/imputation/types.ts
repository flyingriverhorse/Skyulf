export interface ImputationConfig {
  columns: string[];
  method: 'simple' | 'knn' | 'iterative' | 'group';

  // Simple Imputer
  strategy: 'mean' | 'median' | 'most_frequent' | 'constant';
  fill_value?: string | number | undefined;

  // Group Imputer: fill each row from the statistic of its own group.
  group_by?: string | undefined;

  // KNN Imputer
  n_neighbors?: number | undefined;
  weights?: 'uniform' | 'distance' | undefined;

  // Iterative Imputer
  max_iter?: number | undefined;
  estimator?: 'bayesian_ridge' | 'decision_tree' | 'extra_trees' | 'knn' | undefined;
  random_state?: number | undefined;
}

export interface ImputationSettingsProps {
  config: ImputationConfig;
  onChange: (config: ImputationConfig) => void;
}
