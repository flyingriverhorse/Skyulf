/** Evaluation data for a single train/test/val split. */
export interface EvaluationSplit {
  y_true: (string | number)[];
  y_pred: (string | number)[];
  y_proba?: {
    classes: (string | number)[];
    labels?: (string | number)[];
    values: number[][];
  };
  metrics?: Record<string, number>;
}

/** Per-cluster size/percentage/mean-feature-value ("centroid") stats. */
export interface ClusterCentroid {
  cluster_id: number;
  size: number;
  percentage: number;
  center: Record<string, number>;
  /** Auto-generated human-readable label (e.g. "High petal_length, Low
   * petal_width") describing what numerically distinguishes this cluster.
   * Not a real-world name — just a description of the centroid. */
  profile?: string;
}

/** Cluster-size/centroid summary for a clustering split (no ground-truth target). */
export interface ClusteringSummary {
  n_clusters: number;
  cluster_sizes: Record<string, number>;
  centroids: ClusterCentroid[];
  /** Optional: if a "reference column" (e.g. a known label like species
   * name) was set aside during training, this is cluster_id -> {label:
   * row_count}, letting the user see e.g. "Cluster 0 is 92% setosa". */
  reference_crosstab?: Record<string, Record<string, number>> | null;
  reference_column?: string | null;
}

/** Raw split payload for an unsupervised clustering job — no y_true/y_pred,
 * only the predicted cluster label per row plus the centroid summary. */
export interface ClusteringSplit {
  labels: number[];
  clustering?: ClusteringSummary;
  metrics?: Record<string, number>;
}

/** Top-level evaluation response from the backend. */
export type EvaluationData =
  | { problem_type: 'classification' | 'regression'; splits: Record<string, EvaluationSplit> }
  | { problem_type: 'clustering'; splits: Record<string, ClusteringSplit> };

/** Convenience alias: the y_proba shape used by every classification chart helper. */
export type YProba = NonNullable<EvaluationSplit['y_proba']>;

/** A single sampled row's SHAP explanation, as produced by
 * `skyulf.modeling._explainability.compute_shap_explanation`. */
export interface ShapSample {
  base_value: number;
  feature_values: Record<string, number>;
  shap_values: Record<string, number>;
}

/** Global SHAP feature-interaction summary (tree models only). Mean(|interaction
 * value|) matrix over the top-K features by interaction strength; `matrix[i][j]`
 * lines up with `feature_names[i]` / `feature_names[j]`. */
export interface ShapInteractionData {
  feature_names: string[];
  matrix: number[][];
}

/** The `shap_explanation` metric stored on a completed training job. */
export interface ShapExplanationData {
  feature_names: string[];
  mean_abs_importance: Record<string, number>;
  samples: ShapSample[];
  interactions?: ShapInteractionData | null;
}

