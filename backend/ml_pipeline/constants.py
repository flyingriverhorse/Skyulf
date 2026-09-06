"""Step-type constants shared across the pipeline execution layer."""

from enum import StrEnum


class StepType(StrEnum):
    """The structural step types, as the ``step_type`` strings they carry.

    Transformer nodes are dispatched under their own registered step-type
    strings instead, which is why engine code compares against both this enum
    and raw strings.
    """

    DATA_LOADER = "data_loader"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
