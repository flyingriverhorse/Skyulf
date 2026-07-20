// Matching backend/ml_pipeline/constants.py
export enum StepType {
  DATA_LOADER = 'data_loader',
  FEATURE_ENGINEERING = 'feature_engineering',
  BASIC_TRAINING = 'basic_training',
  ADVANCED_TUNING = 'advanced_tuning',
  // Backend Phase 2b superset ("training", run_mode: fixed|tuned). Used only
  // as the canvas `definitionType` for the unified `TrainingNode` (Phase 3);
  // it is never submitted as a job `job_type` — the node still submits
  // BASIC_TRAINING/ADVANCED_TUNING depending on its `run_mode`, same as
  // before, so the backend dispatcher doesn't need to understand this value.
  TRAINING = 'training',
}

// Additional legacy or frontend-specific types can be added here if needed,
// but the goal is to align specifically with the backend execution engine.
