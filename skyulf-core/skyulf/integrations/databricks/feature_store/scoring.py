"""Strict native lookup admission for immutable Skyulf registry packages.

The training_snapshot policy requires exclusive ownership of feature tables:
no concurrent feature writes are allowed during scoring and publication. Native
FE has no versionAsOf argument; guards detect drift but are not a table lock.
"""

from typing import Any

from ...mlflow.shared._nullable_transport import transport_spec
from . import snapshots
from .runtime import require_feature_engineering


def feature_binding(artifact: Any) -> dict[str, Any] | None:
    """Decode only the immutable lookup envelope attached by a verified loader."""
    raw = getattr(artifact, "feature_lookup_json", None)
    if raw is None:
        return None
    from .lifecycle_config import parse_feature_binding  # noqa: PLC0415

    return parse_feature_binding(raw)


def _input_columns(artifact: Any) -> list[tuple[str, str]]:
    """Read the raw input schema of either supported fitted artifact kind."""
    manifest = artifact.manifest
    if hasattr(manifest, "input_columns"):
        return list(zip(manifest.input_columns, manifest.input_dtypes, strict=True))
    return [(column.name, column.dtype) for column in manifest.input_schema]


def feature_source_columns(artifact: Any, keys: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Select direct inputs and lookup controls without preselecting fetched features."""
    inputs = _input_columns(artifact)
    binding = feature_binding(artifact)
    if binding is None:
        return tuple(dict.fromkeys((*keys, *(name for name, _ in inputs))))
    if transport_spec(inputs) is not None:
        raise ValueError("Native feature lookup does not support nullable primitive transport.")
    from .lifecycle_config import deserialize_feature_spec  # noqa: PLC0415

    spec = deserialize_feature_spec(binding["lookup_spec"])
    features = set(spec.feature_names)
    direct = [name for name, _ in inputs if name not in features]
    controls = [
        name for lookup in spec.lookups for name in (*lookup.lookup_key, *lookup.timestamp_columns)
    ]
    return tuple(dict.fromkeys((*keys, *direct, *controls)))


def validate_feature_source(frame: Any, artifact: Any) -> None:
    """Reject silent feature overrides and missing lookup controls before projection."""
    binding = feature_binding(artifact)
    if binding is None:
        return
    from .lifecycle_config import deserialize_feature_spec  # noqa: PLC0415
    from .runtime import validate_feature_input  # noqa: PLC0415

    spec = deserialize_feature_spec(binding["lookup_spec"])
    names = {name.casefold() for name in (*frame.columns, *spec.feature_names)}
    if frame.isStreaming or "prediction" in names:
        raise ValueError(
            "Native feature scoring requires batch input without reserved prediction columns."
        )
    validate_feature_input(frame, spec, training=False, allow_feature_overrides=False)
    missing = set(feature_source_columns(artifact)).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing native feature scoring source inputs: {sorted(missing)}.")


def _current_feature_table(spark: Any, name: str) -> dict[str, Any]:
    """Read current Delta identity and version without collecting feature values."""
    return snapshots.snapshot_table(spark, name)


def _validate_binding_snapshot(spark: Any, binding: dict[str, Any]) -> None:
    """Reject feature-only changes rather than silently returning a stale base-source no-op."""
    snapshots.validate_snapshots(spark, binding["lookup_evidence"]["feature_tables"])


def validate_feature_snapshot(spark: Any, artifact: Any) -> dict[str, Any]:
    """Check the training snapshot before selection and return detached receipt evidence."""
    binding = feature_binding(artifact)
    if binding is None:
        return {}
    _validate_binding_snapshot(spark, binding)
    return {"feature_lookup": binding}


def validate_feature_receipt(spark: Any, receipt: dict[str, Any]) -> None:
    """Recheck feature identity and versions immediately before an atomic publication."""
    binding = receipt.get("feature_lookup")
    if binding is not None:
        from .lifecycle_config import parse_feature_binding  # noqa: PLC0415

        _validate_binding_snapshot(spark, parse_feature_binding(binding))


def validate_feature_continuation(
    previous: dict | None, current: dict[str, Any], *, rebuilding: bool = False
) -> None:
    """Require an explicit rebuild when the published feature lineage changes."""
    if (
        not rebuilding
        and previous is not None
        and previous.get("feature_lookup") != current.get("feature_lookup")
    ):
        raise ValueError(
            "Feature lookup binding changed; explicitly rebuild the prediction target."
        )


def _validate_feature_stores(tracking_uri: str | None, registry_uri: str | None) -> None:
    """Require the native SDK's global download context to match explicit model stores."""
    from ...mlflow.shared._client import require_mlflow  # noqa: PLC0415

    mlflow = require_mlflow()
    if (
        not tracking_uri
        or not registry_uri
        or mlflow.get_tracking_uri() != tracking_uri
        or mlflow.get_registry_uri() != registry_uri
    ):
        raise ValueError(
            "Native feature scoring requires active MLflow tracking and registry URIs "
            "to match the explicitly configured stores."
        )


def native_feature_score(
    frame: Any,
    *,
    model_uri: str,
    result_type: Any,
    env_manager: str,
    prediction_batch_rows: int,
    tracking_uri: str | None,
    registry_uri: str | None,
) -> Any:
    """Invoke native batch lookup using one immutable release and exact named outputs."""
    from ...mlflow.spark._spark_output import (  # noqa: PLC0415
        SPARK_BATCH_ROWS_PARAM,
        SPARK_OUTPUT_PARAM,
    )

    _validate_feature_stores(tracking_uri, registry_uri)
    client = require_feature_engineering().FeatureEngineeringClient(model_registry_uri=registry_uri)
    return client.score_batch(
        model_uri=model_uri,
        df=frame,
        result_type=result_type,
        env_manager=env_manager,
        params={SPARK_OUTPUT_PARAM: True, SPARK_BATCH_ROWS_PARAM: prediction_batch_rows},
    )
