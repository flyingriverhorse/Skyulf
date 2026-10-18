"""Validate the internal removal of CV-only metadata before raw-row inference."""

from ..modeling._tuning.schemas import TuningResult
from ..pipeline._pipeline import _TuningColumnDropApplier
from .fitted_pipeline import FittedPipelineArtifact


def tuning_metadata_columns(artifact: FittedPipelineArtifact) -> tuple[str, ...]:
    """Admit only the exact saved no-op removal of columns absent from inference inputs.

    Temporal/group tuning adds one internal record before the user recipe. Its
    columns must agree with the fitted tuner and be declared CV metadata, never
    raw or transformed model features. No callbacks are executed here.
    """
    records = artifact.pipeline.feature_engineer.fitted_steps
    drops = [record for record in records if record.get("type") == "TuningColumnDrop"]
    result = _saved_tuning_result(artifact)
    columns = getattr(result, "excluded_feature_columns", [])
    if not drops and not columns:
        return ()
    if type(result) is not TuningResult or type(columns) is not list or not columns:
        raise ValueError("Tuning metadata removal requires a fitted tuning result.")
    _validate_columns(artifact, columns)
    if len(drops) != 1 or records[0] is not drops[0]:
        raise ValueError("Tuning metadata removal must be the first and only internal drop.")
    _validate_record(drops[0], columns)
    return tuple(columns)


def _saved_tuning_result(artifact: FittedPipelineArtifact) -> TuningResult | None:
    """Find only the exact tuner wrapper emitted by the built-in fit path."""
    model = getattr(artifact.pipeline.model_estimator, "model", None)
    if type(model) is tuple and len(model) == 2 and type(model[1]) is TuningResult:
        return model[1]
    return None


def _validate_columns(artifact: FittedPipelineArtifact, columns: list) -> None:
    """Bind nonempty unique column names to the configured CV metadata and saved schemas."""
    if any(type(column) is not str or not column for column in columns):
        raise ValueError("Tuning metadata columns must be nonempty strings.")
    modeling = artifact.pipeline.config.get("modeling", {})
    allowed = {modeling.get("cv_time_column"), modeling.get("cv_group_column")} - {None}
    features = set(artifact.manifest.input_columns) | set(artifact.manifest.feature_columns)
    if (
        modeling.get("type") != "hyperparameter_tuner"
        or len(set(columns)) != len(columns)
        or not set(columns) <= allowed
        or set(columns) & features
    ):
        raise ValueError("Tuning metadata removal differs from the configured CV-only columns.")


def _validate_record(record: dict, columns: list[str]) -> None:
    """Require the precise stateless built-in applier and saved column list."""
    if (
        set(record) != {"name", "type", "applier", "artifact"}
        or record["name"] != "tuning_time_columns"
        or type(record["applier"]) is not _TuningColumnDropApplier
        or vars(record["applier"])
        or record["artifact"] != {"columns": columns}
    ):
        raise ValueError("Tuning metadata removal has an altered internal fitted record.")
