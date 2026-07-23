from enum import StrEnum


class StepType(StrEnum):
    DATA_LOADER = "data_loader"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
