from enum import StrEnum


class StepType(StrEnum):
    DATA_LOADER = "data_loader"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"

    # Legacy/Aliases — retained (not removed) so already-saved pipeline JSON /
    # job rows that still reference these values keep loading and executing
    # unchanged. New pipelines only ever write ``TRAINING`` + a ``run_mode``
    # param; the dispatcher normalizes these two old values to that shape on
    # the way in (see ``PipelineEngine._resolve_run_mode``). Do not delete —
    # same precedent as the (now-removed) MODEL_TRAINING/MODEL_TUNING aliases.
    BASIC_TRAINING = "basic_training"
    ADVANCED_TUNING = "advanced_tuning"
