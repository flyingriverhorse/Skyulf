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
