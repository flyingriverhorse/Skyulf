"""Check public native lookup properties and saved SDK YAML without SDK imports."""

import hashlib
from pathlib import Path
from typing import Any

from ...databricks.feature_store.config import FeatureTrainingSpec

SPARK_MLFLOW_TYPES = {
    "smallint": "integer",
    "int": "integer",
    "bigint": "long",
    "float": "float",
    "double": "double",
    "boolean": "boolean",
    "string": "string",
    "date": "datetime",
    "timestamp": "datetime",
    "binary": "binary",
}


def _expected_features(spec: FeatureTrainingSpec) -> dict[str, tuple[Any, ...]]:
    """Bind every explicit feature to its table and ordered temporal lookup keys."""
    return {
        name: (lookup.table_name, name, lookup.lookup_key, lookup.timestamp_columns)
        for lookup in spec.lookups
        for name in lookup.feature_names
    }


def _check_tables(tables: dict[str, Any], spec: FeatureTrainingSpec) -> None:
    """Require identical table membership and lookback durations."""
    expected = {
        lookup.table_name: None
        if lookup.lookback_window is None
        else lookup.lookback_window.total_seconds()
        for lookup in spec.lookups
    }
    if tables != expected:
        raise ValueError(
            "Native feature lookup tables or lookback windows differ from lookup_spec."
        )


def _check_timestamps(dtypes: dict[str, Any], spec: FeatureTrainingSpec) -> None:
    """Require the saved temporal key type instead of allowing implicit date casts."""
    for lookup in spec.lookups:
        if (
            lookup.timestamp_lookup_key is not None
            and dtypes.get(lookup.timestamp_lookup_key) != lookup.timestamp_type
        ):
            raise ValueError("Native feature lookup timestamp type differs from lookup_spec.")


def validate_native_training_set(
    training_set: Any, spec: FeatureTrainingSpec, inputs: dict[str, str]
) -> None:
    """Inspect the original SDK TrainingSet without materializing or replacing it."""
    native = getattr(training_set, "feature_spec", None)
    if native is None or native.function_infos:
        raise ValueError("A native explicit-lookup TrainingSet is required.")
    expected_outputs = list(inputs) + ([] if spec.label is None else [spec.label])
    if sorted(training_set.get_output_columns()) != sorted(expected_outputs):
        raise ValueError("Native TrainingSet output columns differ from model inputs and label.")
    _check_timestamps(
        {column.output_name: column.data_type for column in native.column_infos}, spec
    )
    _check_columns(*_native_columns(native), spec, inputs)
    _check_tables({row.table_name: row.lookback_window for row in native.table_infos}, spec)


def _native_columns(native: Any) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Project public SDK column properties into the validated lookup contract."""
    features = {}
    included = {}
    excluded = []
    for column in native.column_infos:
        if column.include:
            included[column.output_name] = SPARK_MLFLOW_TYPES.get(column.data_type)
        else:
            excluded.append(column.output_name)
        info = column.info
        if hasattr(info, "table_name"):
            if info.default_value_str is not None:
                raise ValueError("Feature lookup defaults are not supported by lookup_spec.")
            features[column.output_name] = (
                info.table_name,
                info.feature_name,
                tuple(info.lookup_key),
                tuple(info.timestamp_lookup_key),
            )
    return included, excluded, features


def _check_columns(
    included: dict[str, Any],
    excluded: list[str],
    features: dict[str, Any],
    spec: FeatureTrainingSpec,
    inputs: dict[str, str],
) -> None:
    """Exclude metadata and labels while preserving the fitted named input order."""
    if included != inputs or not set(excluded).issubset(spec.exclude_columns):
        raise ValueError(
            "Native feature lookup input names/types or excluded columns differ from lookup_spec."
        )
    if features != _expected_features(spec):
        raise ValueError("Native feature lookup columns differ from lookup_spec.")


def feature_spec_entries(value: Any) -> list[tuple[str, dict[str, Any]]]:
    """Reject ambiguous or malformed name-keyed SDK records."""
    if not isinstance(value, list):
        raise ValueError("Feature spec must contain lists of named records.")
    entries = []
    for row in value:
        if not isinstance(row, dict) or len(row) != 1:
            raise ValueError("Feature spec records must contain one name.")
        name, info = next(iter(row.items()))
        if not isinstance(name, str) or not isinstance(info, dict):
            raise ValueError("Feature spec records must map names to attributes.")
        entries.append((name, info))
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError("Feature spec records must have unique names.")
    return entries


def _yaml_columns(value: Any) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Extract only the supported ordinary source and table-lookup semantics."""
    included, excluded, features = {}, [], {}
    for name, data in feature_spec_entries(value):
        if type(data.get("include", True)) is not bool:
            raise ValueError("Feature spec include must be boolean.")
        if data.get("include", True):
            included[name] = SPARK_MLFLOW_TYPES.get(data.get("data_type"))
        else:
            excluded.append(name)
        source = data.get("source")
        if source == "feature_store":
            if data.get("default_value") is not None:
                raise ValueError("Feature lookup defaults are not supported.")
            features[name] = (
                data.get("table_name"),
                data.get("feature_name"),
                tuple(data.get("lookup_key", [])),
                tuple(data.get("timestamp_lookup_key", [])),
            )
        elif source != "training_data":
            raise ValueError("Feature spec contains unsupported column sources.")
    return included, excluded, features


def validate_saved_feature_spec(
    path: Path, spec: FeatureTrainingSpec, inputs: dict[str, str]
) -> str:
    """Bind actual SDK scoring instructions to the declared immutable contract."""
    import yaml  # noqa: PLC0415 - optional MLflow dependency

    if not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("Feature spec is missing or exceeds 64 KiB.")
    payload = path.read_bytes()
    try:
        saved = yaml.safe_load(payload)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError("Invalid saved feature spec YAML.") from exc
    if not isinstance(saved, dict) or saved.get("input_functions") or saved.get("features"):
        raise ValueError("Only explicit table lookup feature specs are supported.")
    _check_columns(*_yaml_columns(saved.get("input_columns")), spec, inputs)
    _check_timestamps(
        {
            name: info.get("data_type")
            for name, info in feature_spec_entries(saved.get("input_columns"))
        },
        spec,
    )
    tables = {
        name: data.get("lookback_window")
        for name, data in feature_spec_entries(saved.get("input_tables"))
    }
    _check_tables(tables, spec)
    return hashlib.sha256(payload).hexdigest()
