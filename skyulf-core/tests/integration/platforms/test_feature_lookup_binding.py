"""Pin immutable native feature lineage before training or scoring can use it."""

import json
from copy import deepcopy
from datetime import timedelta

import pytest

from skyulf.integrations.databricks.feature_store.config import (
    FeatureLookupSpec,
    FeatureTrainingSpec,
    validate_uc_name,
)
from skyulf.integrations.databricks.feature_store.lifecycle_config import (
    bind_lookup_tables,
    binding_json,
    deserialize_feature_spec,
    parse_feature_binding,
    parse_lookup_config,
    serialize_feature_spec,
)

UC_NAME_MESSAGE = (
    "Unity Catalog names must be catalog.schema.name; each part a simple identifier "
    "(letters, digits, underscore; not starting with a digit)."
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


@pytest.mark.parametrize(
    "lookup",
    [
        {"table_name": "{catalog}.{input_schema}.bad name"},
        {"table_name": "a.b.c.d"},
        {"table_name": "{catalog}.history"},
        {"table_name": "", "lookup_key": ["id"], "feature_names": ["amount"]},
    ],
)
def test_unbound_lookup_table_error_names_the_setting_to_complete(lookup):
    """A partly filled lookup must name its indexed field and the one valid name shape."""
    lookups = [{"table_name": "{catalog}.{input_schema}.history"}, lookup]
    original = deepcopy(lookups)
    bindings = {"catalog": "workspace", "input_schema": "features"}
    with pytest.raises(ValueError) as error:
        bind_lookup_tables({"lookups": lookups}, bindings)
    assert str(error.value) == f"feature_lookup.lookups[1].table_name: {UC_NAME_MESSAGE}"
    assert lookups == original  # editable YAML values stay untouched


@pytest.mark.parametrize(
    "stub",
    [
        {"table_name": "", "lookup_key": [], "feature_names": []},
        {"table_name": "  "},
    ],
)
def test_unfilled_generated_lookup_stub_reports_one_actionable_message(stub):
    """The include_online_feature_lookup stub must say what to fill, not misreport a name syntax."""
    with pytest.raises(ValueError) as error:
        bind_lookup_tables({"lookups": [stub]}, {"catalog": "workspace"})
    assert str(error.value) == (
        "feature_lookup.lookups[0] is not filled in: complete feature_lookup and "
        "online_features in config/training.yml, or regenerate with "
        "include_online_feature_lookup=false."
    )


def test_filled_lookup_binding_is_unchanged():
    """A completed lookup must keep binding placeholders exactly as before the stub check."""
    lookup = {"table_name": "{catalog}.f.t", "lookup_key": ["id"], "feature_names": ["x"]}
    bound = bind_lookup_tables({"lookups": [lookup]}, {"catalog": "workspace"})
    assert bound == {"lookups": [{**lookup, "table_name": "workspace.f.t"}]}


@pytest.mark.parametrize(
    "name", ["", "a", "a.b", "a.b.c.d", "1a.b.c", "a.b.bad name", "a.b.c\n", "a..c", None]
)
def test_uc_name_errors_use_one_message(name):
    """Every invalid UC name must fail with the same complete rule, never a two-step hint."""
    with pytest.raises(ValueError) as error:
        validate_uc_name(name)
    assert str(error.value) == UC_NAME_MESSAGE


def test_uc_name_accepts_three_simple_identifiers():
    """Valid catalog.schema.name values must keep passing the narrowed validator."""
    validate_uc_name("_c1.schema_2.Table3")
