"""Feature groups have explicit table ownership and unambiguous join contracts."""

from copy import deepcopy

import pytest

from skyulf.integrations.databricks.features.config import parse_feature_config


@pytest.fixture
def feature_config():
    """Use two data domains independent of the number of trained models."""
    return {
        "version": 1,
        "base_table": "workspace.demo.observations",
        "output_table": "workspace.demo.merged",
        "keys": ["company_id"],
        "timestamp": "observed_at",
        "groups": {
            "company": {
                "source_table": "workspace.demo.raw_company",
                "output_table": "workspace.demo.company_features",
                "transform": "src/features/groups/company.py:compute_features",
                "columns": ["size"],
            },
            "activity": {
                "source_table": "workspace.demo.raw_activity",
                "output_table": "workspace.demo.activity_features",
                "transform": "src/features/groups/activity.py:compute_features",
                "columns": ["amount"],
                "lookup": "asof",
            },
        },
    }


def test_ready_table_needs_no_feature_job():
    """Existing projects must keep their ready-table workflow without extra tasks."""
    assert parse_feature_config({"version": 1, "groups": {}}) is None


def test_config_is_immutable_and_domains_share_keys(feature_config):
    """A model count must never determine feature group layout or mutate user settings."""
    before = deepcopy(feature_config)
    plan = parse_feature_config(feature_config)
    assert plan is not None
    assert plan.record_keys == ("company_id", "observed_at")
    assert [group.lookup for group in plan.groups] == ["exact", "asof"]
    assert feature_config == before


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("keys", [], "keys"),
        ("timestamp", "company_id", "timestamp"),
        ("output_table", "workspace.demo.observations", "table"),
        ("output_table", "observations", "catalog.schema.table"),
        ("surprise", True, "Unknown"),
        ("version", True, "version"),
    ],
)
def test_invalid_contract_is_rejected(feature_config, field, value, reason):
    """Fail before starting Spark or overwriting a source table."""
    feature_config[field] = value
    with pytest.raises(ValueError, match=reason):
        parse_feature_config(feature_config)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("columns", ["size"], "overlap"),
        ("columns", ["company_id"], "key"),
        ("transform", "../outside.py:run", "transform"),
        ("transform", "src/features/groups/../preprocessing.py:run", "transform"),
        ("transform", "src/features/preprocessing.py:run", "transform"),
        ("transform", "src/feature_groups/company.py:run", "transform"),
        ("lookup", "inner", "lookup"),
        ("allow_missing", "false", "boolean"),
        ("output_table", "workspace.demo.raw_company", "table"),
    ],
)
def test_invalid_group_is_rejected(feature_config, field, value, reason):
    """Ambiguous feature names, paths and table ownership cannot enter a job."""
    feature_config["groups"]["activity"][field] = value
    with pytest.raises(ValueError, match=reason):
        parse_feature_config(feature_config)


def _base_only(**changes):
    """A raw source whose labels and keys are derived by one project function."""
    config = {
        "version": 1,
        "base_table": "workspace.demo.raw_observations",
        "base_transform": "src/features/groups/base.py:prepare",
        "output_table": "workspace.demo.observations",
        "keys": ["entity_id", "observed_on"],
        "timestamp": "observed_date",
        "groups": {},
    }
    return config | changes


def test_base_transform_needs_no_feature_groups():
    """Deriving the target and key from one raw table must not need a dummy group."""
    plan = parse_feature_config(_base_only())
    assert plan is not None
    assert plan.groups == ()
    assert plan.base_transform == "src/features/groups/base.py:prepare"


def test_base_transform_can_be_combined_with_groups(feature_config):
    """A derived base still joins regular feature groups on its keys."""
    plan = parse_feature_config(
        feature_config | {"base_transform": "src/features/groups/base.py:run"}
    )
    assert plan is not None and len(plan.groups) == 2


@pytest.mark.parametrize(
    "value", ["src/features/base.py:prepare", "../base.py:prepare", "base.py", 3]
)
def test_base_transform_path_is_a_project_function(value):
    """Only a function in src/features/<module>.py may run on the pinned base."""
    with pytest.raises(ValueError, match="base_transform"):
        parse_feature_config(_base_only(base_transform=value))


def test_groups_and_base_transform_both_empty_stay_disabled():
    """Without groups or a base transform the project keeps its ready table."""
    with pytest.raises(ValueError, match="Disabled features"):
        parse_feature_config({"version": 1, "groups": {}, "base_table": "a.b.c"})


def _placeholder_config(feature_config):
    """Rewrite every table to the same target placeholders training.yml accepts."""
    feature_config["base_table"] = "{catalog}.{input_schema}.observations"
    feature_config["output_table"] = "{catalog}.{output_schema}.merged{resource_suffix}"
    for name, group in feature_config["groups"].items():
        group["source_table"] = "{catalog}.{input_schema}.raw_" + name
        group["output_table"] = "{catalog}.{output_schema}." + name + "_features{resource_suffix}"
    feature_config["source_snapshot_schema"] = "{catalog}.{metadata_schema}"
    return feature_config


def test_target_placeholders_bind_per_target(feature_config):
    """One features.yml must serve test, syst, prod and personal targets without collisions."""
    from skyulf.integrations.databricks.features.config import bind_feature_plan

    plan = parse_feature_config(_placeholder_config(feature_config))
    assert plan is not None
    bindings = {
        "catalog": "prod",
        "input_schema": "raw",
        "output_schema": "gold",
        "metadata_schema": "ml",
        "resource_suffix": "_dev_7",
    }
    bound = bind_feature_plan(plan, bindings)
    assert bound.base_table == "prod.raw.observations"
    assert bound.output_table == "prod.gold.merged_dev_7"
    assert [g.source_table for g in bound.groups] == [
        "prod.raw.raw_company",
        "prod.raw.raw_activity",
    ]
    assert bound.groups[0].output_table == "prod.gold.company_features_dev_7"
    assert bound.snapshot_schema == "prod.ml"
    assert plan.base_table == "{catalog}.{input_schema}.observations"  # parsing never binds


def test_plain_names_need_no_bindings(feature_config):
    """Existing literal three-part names keep working without any feature-job parameter."""
    from skyulf.integrations.databricks.features.config import bind_feature_plan

    plan = parse_feature_config(feature_config)
    assert plan is not None
    assert bind_feature_plan(plan, {}) == plan


def test_unknown_placeholder_is_rejected_before_deployment(feature_config):
    """A typo such as {catalg} must fail locally instead of reading a literal brace name."""
    feature_config["base_table"] = "{catalg}.demo.observations"
    with pytest.raises(ValueError):
        parse_feature_config(feature_config)


def test_missing_binding_names_the_fix(feature_config):
    """An old feature graph without bindings must say how to repair it."""
    from skyulf.integrations.databricks.features.config import bind_feature_plan

    plan = parse_feature_config(_placeholder_config(feature_config))
    assert plan is not None
    with pytest.raises(ValueError, match="refresh_feature_graph"):
        bind_feature_plan(plan, {"catalog": "prod"})
