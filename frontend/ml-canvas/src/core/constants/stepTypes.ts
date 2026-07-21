// Matching backend/ml_pipeline/constants.py
export enum StepType {
  DATA_LOADER = 'data_loader',
  FEATURE_ENGINEERING = 'feature_engineering',
  // Every training-family canvas node (the generic node plus the
  // task-scoped Classification/Regression/Text Classification/Segmentation/
  // Ensemble nodes) submits this same canonical step_type, discriminated by
  // a `run_mode: 'fixed' | 'tuned'` param — the backend doesn't need to know
  // which canvas node produced the job.
  TRAINING = 'training',
  CLASSIFICATION = 'classification',
  REGRESSION = 'regression',
  TEXT_CLASSIFICATION = 'text_classification',
}
