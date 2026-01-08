// Matching backend/ml_pipeline/constants.py
export enum StepType {
  DATA_LOADER = 'data_loader',
  FEATURE_ENGINEERING = 'feature_engineering',
  BASIC_TRAINING = 'basic_training',
  ADVANCED_TUNING = 'advanced_tuning',
}

// Additional legacy or frontend-specific types can be added here if needed,
// but the goal is to align specifically with the backend execution engine.
