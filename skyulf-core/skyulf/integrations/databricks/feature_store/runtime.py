"""Optional Feature Engineering calls without replacing the Skyulf lifecycle."""

import importlib
from collections.abc import Callable
from typing import Any

from .config import FeatureLookupSpec, FeatureTrainingSpec, _uc_name


def _sdk() -> Any:
    """Load the runtime-only dependency when an uninjected operation needs it."""
    try:
        return importlib.import_module("databricks.feature_engineering")
    except ModuleNotFoundError as exc:
        if exc.name not in ("databricks", "databricks.feature_engineering"):
            raise
        raise ImportError(
            "Unity Catalog feature lookups require skyulf-core[feature-store] "
            "on a compatible Databricks runtime."
        ) from exc


def _client(client: Any) -> Any:
    """Keep caller-injected clients and their selected workspace unchanged."""
    return _sdk().FeatureEngineeringClient() if client is None else client


def _lookup(spec: FeatureLookupSpec, factory: Callable[..., Any]) -> Any:
    """Translate an explicit lookup without materializing or casting data."""
    options: dict[str, Any] = {
        "table_name": spec.table_name,
        "lookup_key": list(spec.lookup_key),
        "feature_names": list(spec.feature_names),
    }
    if spec.timestamp_lookup_key is not None:
        options["timestamp_lookup_key"] = spec.timestamp_lookup_key
    if spec.lookback_window is not None:
        options["lookback_window"] = spec.lookback_window
    return factory(**options)


def _validate_input(
    df: Any,
    spec: FeatureTrainingSpec,
    *,
    training: bool,
    allow_feature_overrides: bool,
) -> None:
    """Validate the source schema before any optional SDK or remote operation."""
    if not isinstance(spec, FeatureTrainingSpec):
        raise TypeError("spec must be FeatureTrainingSpec.")
    if type(allow_feature_overrides) is not bool:
        raise TypeError("allow_feature_overrides must be bool.")
    columns = list(df.columns)
    if len({name.casefold() for name in columns}) != len(columns):
        raise ValueError("Input columns must have distinct names.")
    required = _required_columns(spec, training=training)
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Missing feature lookup input columns: {sorted(missing)}.")
    _validate_time_types(dict(df.dtypes), spec)
    _validate_overrides(columns, spec, allow_feature_overrides)


def _validate_overrides(
    columns: list[str], spec: FeatureTrainingSpec, allow_feature_overrides: bool
) -> None:
    """Make supplied feature values an explicit opt-in to SDK override behavior."""
    features = {name.casefold() for name in spec.feature_names}
    supplied = {name for name in columns if name.casefold() in features}
    if supplied and not allow_feature_overrides:
        raise ValueError(
            f"Input features would override table lookups: {sorted(supplied)}. "
            "Pass allow_feature_overrides=True to use Databricks override semantics."
        )


def _required_columns(spec: FeatureTrainingSpec, *, training: bool) -> set[str]:
    """Require labels and exclusions only when constructing the training set."""
    columns = {
        name for lookup in spec.lookups for name in (*lookup.lookup_key, *lookup.timestamp_columns)
    }
    if training:
        columns.update(set(spec.exclude_columns).difference(spec.feature_names))
        if spec.label is not None:
            columns.add(spec.label)
    return columns


def _validate_time_types(dtypes: dict[str, str], spec: FeatureTrainingSpec) -> None:
    """Reject temporal dtype drift instead of silently casting time semantics."""
    for lookup in spec.lookups:
        for name in lookup.timestamp_columns:
            if dtypes.get(name) != lookup.timestamp_type:
                raise ValueError(
                    f"Timestamp lookup column {name!r} must have Spark type "
                    f"{lookup.timestamp_type!r}; got {dtypes.get(name)!r}."
                )


def create_feature_training_set(
    df: Any,
    spec: FeatureTrainingSpec,
    *,
    client: Any = None,
    lookup_factory: Callable[..., Any] | None = None,
    allow_feature_overrides: bool = False,
) -> Any:
    """Create a native TrainingSet retaining point-in-time lookup metadata.

    Train from the returned object's ``load_df()`` and retain that same object
    for ``log_feature_model``. Apply fitted preprocessing inside the model,
    because SDK feature lookup does not replay external frame transformations.
    This helper does not collect Spark data, fit models, or publish tables.
    """
    _validate_input(df, spec, training=True, allow_feature_overrides=allow_feature_overrides)
    factory = _sdk().FeatureLookup if lookup_factory is None else lookup_factory
    lookups = [_lookup(lookup, factory) for lookup in spec.lookups]
    return _client(client).create_training_set(
        df=df,
        feature_lookups=lookups,
        label=spec.label,
        exclude_columns=list(spec.exclude_columns),
    )


def _validate_log_destination(artifact_path: str, options: dict[str, Any]) -> None:
    """Require run-relative artifacts and Unity Catalog registration names."""
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError("artifact_path must be a nonempty run-relative path.")
    parts = artifact_path.split("/")
    if any(part in ("", ".", "..") for part in parts) or "\\" in artifact_path:
        raise ValueError("artifact_path must be a run-relative path without traversal.")
    if ":" in artifact_path:
        raise ValueError("artifact_path must be a run-relative path.")
    if options.get("registered_model_name") is not None:
        _uc_name(options["registered_model_name"])


def log_feature_model(
    model: Any,
    *,
    training_set: Any,
    flavor: Any,
    artifact_path: str,
    client: Any = None,
    **model_options: Any,
) -> Any:
    """Package a compatible MLflow flavor with its native training-set lineage.

    The caller owns the active MLflow run and must have trained on this set's
    ``load_df()`` output. Named signatures and artifact/environment options are
    passed unchanged. This generic adapter does not package a Skyulf fitted
    artifact or issue a Spark partition-safety certificate; use of the existing
    Skyulf lifecycle requires its dedicated packaging and validation steps.
    """
    _validate_log_destination(artifact_path, model_options)
    if not callable(getattr(training_set, "load_df", None)):
        raise TypeError("training_set must be a native TrainingSet with load_df().")
    if not callable(getattr(flavor, "save_model", None)):
        raise TypeError("flavor must support MLflow save_model().")
    return _client(client).log_model(
        model=model,
        training_set=training_set,
        flavor=flavor,
        artifact_path=artifact_path,
        **model_options,
    )


def score_feature_model(
    model_uri: str,
    df: Any,
    spec: FeatureTrainingSpec,
    *,
    result_type: Any,
    client: Any = None,
    allow_feature_overrides: bool = False,
    **score_options: Any,
) -> Any:
    """Score a feature-packaged model through automatic SDK feature lookup.

    Supply the same lookup contract used for training and the model's Spark
    output type explicitly, including field names for structured predictions.
    Existing input features are rejected unless their SDK override behavior is
    explicitly enabled. The SDK checks model metadata and table key/type
    compatibility. This call does not certify Skyulf partition safety.
    """
    if not isinstance(model_uri, str) or not model_uri.startswith(("models:/", "runs:/")):
        raise ValueError("model_uri must reference a feature-packaged models:/ or runs:/ model.")
    if result_type is None or (isinstance(result_type, str) and not result_type.strip()):
        raise ValueError("result_type must explicitly describe the model output schema.")
    _validate_input(df, spec, training=False, allow_feature_overrides=allow_feature_overrides)
    return _client(client).score_batch(
        model_uri=model_uri, df=df, result_type=result_type, **score_options
    )
