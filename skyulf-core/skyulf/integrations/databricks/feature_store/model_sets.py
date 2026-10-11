"""Combine compatible native lookup lineage into one immutable model-set package."""

from dataclasses import replace
from typing import Any

from .config import FeatureLookupSpec, FeatureTrainingSpec
from .lifecycle_config import (
    binding_json,
    deserialize_feature_spec,
    parse_feature_binding,
    serialize_feature_spec,
)
from .scoring import (
    feature_binding,
    feature_source_columns,
    validate_feature_snapshot,
    validate_feature_source,
)
from .training import create_checked_training_set


def _merge_lookup_features(
    outputs: dict[str, FeatureLookupSpec], spec: FeatureTrainingSpec
) -> None:
    """Require every shared output to resolve through identical table and time semantics."""
    for lookup in spec.lookups:
        for name in lookup.feature_names:
            single = replace(lookup, feature_names=(name,))
            previous = outputs.setdefault(name, single)
            if previous != single:
                raise ValueError(f"Model-set feature {name!r} has incompatible component lookups.")


def _merge_snapshots(records: dict[str, dict], binding: dict[str, Any]) -> None:
    """Keep one matching training version for every table used anywhere in the set."""
    for table in binding["lookup_evidence"]["feature_tables"]:
        previous = records.setdefault(table["table_name"], table)
        if previous != table:
            raise ValueError("Model-set components use incompatible feature table snapshots.")


def _ordered_lookups(
    features: dict[str, FeatureLookupSpec], columns: tuple[str, ...]
) -> tuple[FeatureLookupSpec, ...]:
    """Keep fitted input order while merging adjacent outputs with the same lookup."""
    if set(features).difference(columns):
        raise ValueError("Model-set lookup features must occur in its fitted input schema.")
    result: list[FeatureLookupSpec] = []
    for name in columns:
        lookup = features.get(name)
        if lookup is None:
            continue
        if result and replace(result[-1], feature_names=(name,)) == lookup:
            result[-1] = replace(result[-1], feature_names=(*result[-1].feature_names, name))
        else:
            result.append(lookup)
    return tuple(result)


def union_feature_binding(
    components: dict[str, Any], input_columns: tuple[str, ...]
) -> dict[str, Any] | None:
    """Build one compatible feature contract without changing any component's input meaning."""
    features: dict[str, FeatureLookupSpec] = {}
    records: dict[str, dict] = {}
    direct: set[str] = set()
    for artifact in components.values():
        binding = feature_binding(artifact)
        fetched: tuple[str, ...] = ()
        if binding is not None:
            spec = deserialize_feature_spec(binding["lookup_spec"])
            fetched = spec.feature_names
            _merge_lookup_features(features, spec)
            _merge_snapshots(records, binding)
        direct.update(set(artifact.manifest.input_columns).difference(fetched))
    if not features:
        return None
    if direct.intersection(features):
        raise ValueError("A model-set feature is a direct input to another component.")
    lookups = _ordered_lookups(features, input_columns)
    controls = tuple(
        dict.fromkeys(
            name for lookup in lookups for name in (*lookup.lookup_key, *lookup.timestamp_columns)
        )
    )
    spec = FeatureTrainingSpec(
        lookups=lookups,
        label=None,
        exclude_columns=tuple(name for name in controls if name not in input_columns),
    )
    return parse_feature_binding(
        {
            "version": 1,
            "lookup_spec": serialize_feature_spec(spec),
            "lookup_evidence": {
                "policy": "training_snapshot",
                "feature_tables": list(records.values()),
            },
        }
    )


def validate_component_bindings(branches: tuple, components: dict[str, Any]) -> None:
    """Bind verified registry lookup metadata to each completed branch's saved training spec."""
    for branch in branches:
        artifact = components[branch.name]
        actual = feature_binding(artifact)
        expected_json = getattr(branch.spec, "feature_binding_json", None)
        expected = None if expected_json is None else parse_feature_binding(expected_json)
        if getattr(branch.spec, "feature_lookup_json", None) is not None and expected is None:
            raise ValueError("Declared component feature lookup is missing its training snapshot.")
        if actual != expected:
            raise ValueError("Registered component feature binding differs from its training plan.")
        if actual is not None and artifact.manifest.input_columns != branch.spec.input_columns:
            raise ValueError("Registered feature component inputs differ from its training plan.")


def model_set_training_set(spark: Any, source: Any, artifact: Any, binding: dict[str, Any]) -> Any:
    """Recreate one native union TrainingSet from the already pinned common source."""
    bound = replace(artifact, feature_lookup_json=binding_json(binding))
    validate_feature_source(source, bound)
    selected = source.select(*feature_source_columns(bound))
    return create_checked_training_set(
        selected,
        deserialize_feature_spec(binding["lookup_spec"]),
        binding,
        spark=spark,
    )


def enrich_approval_source(spark: Any, source: Any, artifact: Any) -> Any:
    """Fetch saved lookup inputs while retaining identifiers for a bounded functional probe."""
    binding = feature_binding(artifact)
    if binding is None:
        return source
    validate_feature_source(source, artifact)
    validate_feature_snapshot(spark, artifact)
    lookup = replace(deserialize_feature_spec(binding["lookup_spec"]), exclude_columns=())
    selected = source.select(*feature_source_columns(artifact))
    return create_checked_training_set(selected, lookup, binding, spark=spark).load_df()
