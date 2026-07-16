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

/** Top-level evaluation response from the backend. */
export interface EvaluationData {
  problem_type: 'classification' | 'regression';
  splits: Record<string, EvaluationSplit>;
}

/** Convenience alias: the y_proba shape used by every classification chart helper. */
export type YProba = NonNullable<EvaluationSplit['y_proba']>;

/** A single sampled row's SHAP explanation, as produced by
 * `skyulf.modeling._explainability.compute_shap_explanation`. */
export interface ShapSample {
  base_value: number;
  feature_values: Record<string, number>;
  shap_values: Record<string, number>;
}

/** The `shap_explanation` metric stored on a completed training job. */
export interface ShapExplanationData {
  feature_names: string[];
  mean_abs_importance: Record<string, number>;
  samples: ShapSample[];
}

