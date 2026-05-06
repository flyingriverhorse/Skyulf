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
