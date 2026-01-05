from enum import Enum

class StepType(str, Enum):
    DATA_LOADER = "data_loader"
    FEATURE_ENGINEERING = "feature_engineering"
    BASIC_TRAINING = "basic_training"
    ADVANCED_TUNING = "advanced_tuning"
    
    # Legacy/Aliases (kept for backward compatibility if needed, though we refactored)
    # MODEL_TRAINING = "model_training" 
    # MODEL_TUNING = "model_tuning"
