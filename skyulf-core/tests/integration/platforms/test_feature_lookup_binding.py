"""Pin immutable native feature lineage before training or scoring can use it."""

import json
from copy import deepcopy
from datetime import timedelta

import pytest

from skyulf.integrations.databricks.feature_store.config import (
    FeatureLookupSpec,
    FeatureTrainingSpec,
)
from skyulf.integrations.databricks.feature_store.lifecycle_config import (
    binding_json,
    deserialize_feature_spec,
    parse_feature_binding,
    parse_lookup_config,
    serialize_feature_spec,
)


def _spec():
    """Use a historical feature and separate metadata from actual model inputs."""
    return FeatureTrainingSpec(
        lookups=(
            FeatureLookupSpec(
                table_name="workspace.features.history",
                lookup_key=("id",),
                feature_names=("amount",),
                timestamp_lookup_key="at",
                lookback_window=timedelta(days=30),
            ),
        ),
        label="target",
        exclude_columns=("id", "at"),
    )


def _binding():
    """Represent one immutable table identity and the exact lookup contract."""
    return {
        "version": 1,
        "lookup_spec": serialize_feature_spec(_spec()),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": [
                {"table_name": "workspace.features.history", "table_id": "table-one", "version": 2}
            ],
        },
    }


def test_binding_roundtrip_and_detachment():
    """Receipts and artifact metadata must preserve the same independent lookup identity."""
    value = _binding()
    frozen = binding_json(value)
    decoded = parse_feature_binding(frozen)
    value["lookup_evidence"]["feature_tables"][0]["version"] = 99
    assert decoded["lookup_evidence"]["feature_tables"][0]["version"] == 2
    assert deserialize_feature_spec(decoded["lookup_spec"]) == _spec()
    assert binding_json(decoded) == frozen


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.update(version=True),
        lambda v: v.update(extra="ignored"),
        lambda v: v["lookup_evidence"].update(policy="latest"),
        lambda v: v["lookup_evidence"]["feature_tables"][0].update(version=True),
        lambda v: v["lookup_evidence"]["feature_tables"][0].update(table_id=""),
        lambda v: v["lookup_evidence"]["feature_tables"][0].update(
            table_name="workspace.features.other"
        ),
        lambda v: v["lookup_evidence"]["feature_tables"].append(
            deepcopy(v["lookup_evidence"]["feature_tables"][0])
        ),
        lambda v: v["lookup_spec"]["lookups"][0].update(lookup_key="id"),
        lambda v: v["lookup_spec"]["lookups"][0].update(lookback_seconds=float("inf")),
    ],
)
def test_binding_rejects_ambiguous_contracts(mutation):
    """Malformed metadata cannot weaken snapshot or temporal lookup rules."""
    value = _binding()
    mutation(value)
    with pytest.raises((ValueError, TypeError)):
        parse_feature_binding(value)


def test_binding_rejects_duplicate_json_keys():
    """A repeated field cannot replace the published version silently."""
    source = json.dumps(_binding()).replace('"version": 1', '"version": 1, "version": 1')
    with pytest.raises(ValueError, match="Duplicate"):
        parse_feature_binding(source)


def test_workflow_lookup_derives_label_and_exclusions():
    """Users declare lookups once while lifecycle metadata stays outside model inputs."""
    settings = {"lookups": serialize_feature_spec(_spec())["lookups"]}
    result = parse_lookup_config(
        settings, label="target", input_columns=("amount",), exclude_columns=("id", "at")
    )
    assert result == _spec()
    assert parse_lookup_config(None, label="target", input_columns=("amount",)) is None


def test_workflow_lookup_rejects_unconsumed_feature():
    """Lookup configuration must match the frozen pipeline input contract."""
    with pytest.raises(ValueError, match="input_columns"):
        parse_lookup_config(
            {"lookups": serialize_feature_spec(_spec())["lookups"]},
            label="target",
            input_columns=("another_feature",),
        )
