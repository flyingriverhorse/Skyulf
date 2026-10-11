"""Reconstruct native feature observations from verified scoring receipts in Spark."""

from dataclasses import replace
from functools import partial
from typing import Any

from .lifecycle_config import deserialize_feature_spec, parse_feature_binding
from .snapshots import validate_snapshots
from .training import create_checked_training_set


def bind_observation_reader(reader: Any, config: Any, artifact: Any) -> Any:
    """Route feature-backed observations only through the admitted distributed batch reader."""
    value = getattr(artifact, "feature_lookup_json", None)
    if value is None:
        return reader
    if config.execution_engine != "spark" or config.serving_endpoint:
        raise ValueError("Native feature monitoring requires Spark batch observations.")
    return partial(reader, feature_binding=parse_feature_binding(value))


def _feature_contracts(binding: dict) -> dict[str, dict]:
    """Compare per-feature lookups even when a model set has coalesced their declarations."""
    return {
        name: {key: value for key, value in item.items() if key != "feature_names"}
        for item in binding["lookup_spec"]["lookups"]
        for name in item["feature_names"]
    }


def _receipt_binding(expected: dict, receipt: dict) -> None:
    """Require the same component lineage inside a single-model or complete-set receipt."""
    if receipt.get("feature_lookup") is None:
        raise ValueError("Scoring receipt is missing native feature evidence.")
    actual = parse_feature_binding(receipt["feature_lookup"])
    contracts = _feature_contracts(actual)
    if any(contracts.get(name) != value for name, value in _feature_contracts(expected).items()):
        raise ValueError("Scoring receipt feature lookup differs from the monitored model.")
    tables = {row["table_name"]: row for row in actual["lookup_evidence"]["feature_tables"]}
    if any(
        tables.get(row["table_name"]) != row
        for row in expected["lookup_evidence"]["feature_tables"]
    ):
        raise ValueError("Scoring receipt feature snapshot differs from the monitored model.")


def enrich_observation(
    spark: Any,
    source: Any,
    keys: tuple[str, ...],
    features: tuple[str, ...],
    binding: dict | None,
    receipt: dict | None,
) -> Any:
    """Fetch the same historical inputs for scored record keys, with no label requirement."""
    if binding is None:
        return source
    binding = parse_feature_binding(binding)
    if receipt is not None:
        _receipt_binding(binding, receipt)
    lookup = replace(
        deserialize_feature_spec(binding["lookup_spec"]), label=None, exclude_columns=()
    )
    supplied = {name.casefold() for name in source.columns}
    if supplied.intersection(name.casefold() for name in lookup.feature_names):
        raise ValueError("Monitoring source would override native feature lookups.")
    controls = tuple(
        name for item in lookup.lookups for name in (*item.lookup_key, *item.timestamp_columns)
    )
    direct = tuple(name for name in features if name not in lookup.feature_names)
    source = source.select(*dict.fromkeys((*keys, *direct, *controls)))
    return create_checked_training_set(source, lookup, binding, spark=spark).load_df()


def validate_observation_snapshot(spark: Any, binding: dict | None) -> None:
    """Recheck table history after distributed key and population probes execute."""
    if binding is not None:
        validate_snapshots(spark, binding["lookup_evidence"]["feature_tables"])
