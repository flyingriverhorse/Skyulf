"""Bind validated portable FE records to known implementations and fresh runtime context."""

from typing import Any

from ..core.execution import ExecutionOptions, FrameSpec
from ..core.portable_pipeline import decode_pipeline, encode_pipeline
from ..core.portable_state import DEFAULT_MAX_STATE_BYTES
from ._spark import _preflight
from .imputation.simple import SimpleImputerApplier
from .scaling.standard import StandardScalerApplier

_APPLIERS = {"SimpleImputer": SimpleImputerApplier, "StandardScaler": StandardScalerApplier}


def export_feature_state(engineer: Any, *, options: ExecutionOptions | None = None) -> bytes:
    """Reject incomplete or unsupported execution histories before serializing learned state."""
    if not getattr(engineer, "_portable_fitted", bool(engineer.fitted_steps)):
        raise ValueError("FeatureEngineer must be successfully fitted before exporting state.")
    if len(engineer.fitted_steps) != len(engineer.steps_config):
        raise ValueError("Unsupported or incomplete portable pipeline step history.")
    records = []
    for record, config in zip(engineer.fitted_steps, engineer.steps_config, strict=True):
        node = record["type"]
        if node not in _APPLIERS or type(record["applier"]) is not _APPLIERS[node]:
            raise ValueError("Unsupported portable pipeline node or custom applier.")
        records.append({**record, "params": record.get("params", config.get("params", {}))})
    budget = options if options is not None else getattr(engineer, "execution_options", None)
    return encode_pipeline(records, max_bytes=_budget(budget))


def restore_feature_state(
    payload: bytes, options: ExecutionOptions | None, spec: FrameSpec | None
) -> tuple[list[dict], list[dict]]:
    """Rebind explicit Spark keys and instantiate only known node appliers after validation."""
    if options is not None and not isinstance(options, ExecutionOptions):
        raise TypeError("execution_options must be ExecutionOptions.")
    if spec is not None and not isinstance(spec, FrameSpec):
        raise TypeError("frame_spec must be FrameSpec.")
    spark = options is not None and options.engine == "spark"
    if spark and spec is None:
        raise ValueError("Spark requires frame_spec when loading portable state.")
    if not spark and spec is not None:
        raise ValueError("frame_spec is currently supported only for Spark execution.")
    records = decode_pipeline(payload, max_bytes=_budget(options))
    if spark:
        assert spec is not None
        _preflight(records, spec, training=False)
    configs = [
        {"name": step["name"], "transformer": step["type"], "params": step["params"]}
        for step in records
    ]
    for step in records:
        step["applier"] = _APPLIERS[step["type"]]()
    return configs, records


def _budget(options: ExecutionOptions | None) -> int:
    """Apply one total byte limit independently of the destination runtime."""
    return options.state_max_bytes if options is not None else DEFAULT_MAX_STATE_BYTES
