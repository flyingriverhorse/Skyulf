"""Shared fitted schema checks and delegation to preprocessing-owned validators."""

from typing import Any

from ..core.portable_state import _pack
from ..pipeline.seal import artifact_digest
from .fitted_pipeline import FittedPipelineArtifact


def check_fitted_schemas(artifact: FittedPipelineArtifact) -> None:
    """Bind column order and dtype metadata to the saved inference schemas."""
    schemas = artifact.pipeline._inference_schemas
    manifest = artifact.manifest
    if schemas is None:
        raise ValueError("Missing fitted inference schemas.")
    actual = tuple(
        (schema.columns, tuple(schema.dtypes.get(name, "unknown") for name in schema.columns))
        for schema in schemas
    )
    expected = (
        (manifest.input_columns, manifest.input_dtypes),
        (manifest.feature_columns, manifest.feature_dtypes),
    )
    if actual != expected:
        raise ValueError("Manifest and fitted input/output schemas disagree.")


def resolve_fitted_step(
    record: dict, config: dict, *, require_portable: bool = True
) -> tuple[Any, dict, bool]:
    """Delegate strict worker validation or explicitly abstain for local-only state."""
    if config["name"] != record["name"] or config["transformer"] != record["type"]:
        raise ValueError("Configured step disagrees with fitted name/type.")
    owner = type(record["applier"])
    validate = getattr(owner, "validate_fitted_state", None)
    resolve = getattr(owner, "resolve_fitted_config", None)
    if callable(validate) and callable(resolve):
        try:
            state = validate(record["artifact"])
        except (TypeError, ValueError):
            if require_portable:
                raise
            return _local_record(record, config)
        params = resolve(record.get("params", {}), state)
        recipe = resolve(config.get("params", {}), state)
        if _pack(params) != _pack(recipe):
            raise ValueError("Recipe configuration disagrees with fitted parameters.")
        return state, params, True
    if require_portable:
        raise ValueError("Required fitted validation hooks are missing.")
    return _local_record(record, config)


def _local_record(record: dict, config: dict) -> tuple[Any, dict, bool]:
    """Compare unresolved recipes, accounting for target context injected by training."""
    params = dict(record.get("params", {}))
    recipe = config.get("params", {})
    if "target_column" not in recipe:
        params.pop("target_column", None)
    if artifact_digest(params) != artifact_digest(recipe):
        raise ValueError("Recipe configuration disagrees with fitted parameters.")
    return record["artifact"], params, False
