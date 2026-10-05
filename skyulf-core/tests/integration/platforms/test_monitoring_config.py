"""Monitoring enrollment keeps namespaces, bounds and model selection explicit."""

import json

import pytest


def entry(**changes):
    """Use two different source namespaces without assuming the registry schema."""
    return {
        "environment": "test",
        "project": "risk",
        "model_name": "models.risk.score",
        "model_version": "2",
        "source_table": "features.risk.inputs",
        "prediction_table": "outputs.risk.predictions",
        **changes,
    }


def test_namespace_and_version_are_separate_from_monitor_identity():
    """Upgrading a model preserves its inventory history while catalog changes do not collide."""
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    original = MonitorConfig.from_dict(entry())
    upgraded = MonitorConfig.from_dict(entry(model_version="3"))
    other = MonitorConfig.from_dict(entry(model_name="other.risk.score"))
    assert original.monitor_id == upgraded.monitor_id
    assert original.monitor_id != other.monitor_id
    assert original.max_rows == 10000


@pytest.mark.parametrize(
    "changes",
    [
        {"model_name": "score"},
        {"source_table": "a.b.c; DROP TABLE x"},
        {"max_rows": True},
        {"max_rows": 0},
        {"max_bytes": -1},
        {"expected_interval_hours": float("nan")},
        {"model_version": "champion"},
        {"model_alias": "champion"},
        {"surprise": True},
    ],
)
def test_invalid_enrollment_is_rejected_before_cloud_access(changes):
    """Typos and ambiguous model selections must not silently create a monitor."""
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    with pytest.raises((ValueError, TypeError)):
        MonitorConfig.from_dict(entry(**changes))


def test_alias_selection_and_absent_labels_are_explicit():
    """A named alias is allowed only without a competing concrete version."""
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    config = MonitorConfig.from_dict(entry(model_version=None, model_alias="champion"))
    assert config.model_alias == "champion"
    assert config.label_table is None


def test_store_namespace_requires_two_valid_identifiers():
    """All central object names are validated before interpolating SQL identifiers."""
    from skyulf.integrations.databricks.monitoring_config import store_namespace

    assert store_namespace("operations", "monitoring") == "operations.monitoring"
    with pytest.raises(ValueError):
        store_namespace("operations.other", "monitoring")


def test_model_set_component_requires_complete_parent_identity():
    """A component projection must bind to one concrete parent release and branch."""
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    with pytest.raises(ValueError):
        MonitorConfig.from_dict(entry(model_set_name="models.risk.set"))
    config = MonitorConfig.from_dict(
        entry(
            model_set_name="models.risk.set",
            model_set_version="3",
            model_set_branch="amount",
        )
    )
    assert config.model_set_branch == "amount"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("{}", {}),
        (
            '{"psi": 0.3, "ks_statistic": 0.2, "wasserstein": 0.4, "kl_divergence": 1}',
            {"psi": 0.3, "ks_statistic": 0.2, "wasserstein": 0.4, "kl_divergence": 1},
        ),
    ],
)
def test_bundle_threshold_policy_accepts_defaults_and_all_core_metrics(raw, expected):
    """Bundle overrides retain each metric's value without inventing new defaults."""
    from skyulf.integrations.databricks.monitoring_config import parse_drift_thresholds

    assert parse_drift_thresholds(raw) == expected


def performance_policy():
    """Use an explicit version-bound policy for enrollment assertions."""
    return {
        "mode": "report",
        "metric": "f1_weighted",
        "direction": "higher",
        "baseline": {"kind": "training_holdout", "model_version": "2"},
        "tolerance": 0.05,
        "tolerance_mode": "absolute",
        "window_hours": 24,
        "label_delay_hours": 6,
        "minimum_labeled_rows": 20,
        "minimum_label_coverage": 0.8,
        "consecutive_windows": 3,
    }


def test_absent_performance_policy_preserves_legacy_payload_digest():
    """Upgrading old enrollment must not change inventory identity or report replay keys."""
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    config = MonitorConfig.from_dict(entry())
    assert "performance_policy" not in config.payload()
    assert config.payload()["thresholds"] is None


def test_active_performance_policy_requires_actual_outcome_source():
    """Performance requests need real labels and a known UTC availability column."""
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    with pytest.raises(ValueError, match="label_table"):
        MonitorConfig.from_dict(entry(performance_policy=performance_policy()))
    config = MonitorConfig.from_dict(
        entry(
            label_table="labels.risk.actuals",
            result_available_at_column="available_at",
            performance_policy=performance_policy(),
        )
    )
    assert config.payload()["performance_policy"] == performance_policy()


def test_enrolled_performance_policy_is_detached_from_caller_mutation():
    """A later edit of caller settings cannot change a frozen inventory digest."""
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    policy = performance_policy()
    config = MonitorConfig.from_dict(
        entry(
            label_table="labels.risk.actuals",
            result_available_at_column="available_at",
            performance_policy=policy,
        )
    )
    policy["baseline"]["model_version"] = "99"
    assert config.payload()["performance_policy"]["baseline"]["model_version"] == "2"


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"model": {"mode": "off"}}',
        '{"models.risk.model": {"mode": "off", "surprise": true}}',
        '{"models.risk.model": "off"}',
        '{"models.risk.model": {}}',
        '{"models.risk.model": null}',
        '{"models.risk.model": {"mode": "report"}}',
        '{"models.risk.model": {"mode": "off"}, "models.risk.model": {"mode": "off"}}',
        '{"models.risk.model": {"mode": "off", "mode": "off"}}',
    ],
)
def test_invalid_per_model_performance_mapping_is_rejected(raw):
    """Typos or malformed policies must fail before scoring writes predictions."""
    from skyulf.integrations.databricks.monitoring_config import parse_performance_policies

    with pytest.raises(ValueError):
        parse_performance_policies(raw)


def test_per_model_performance_mapping_preserves_exact_keys():
    """Only an explicitly named component may inherit its own performance settings."""
    from skyulf.integrations.databricks.monitoring_config import parse_performance_policies

    policies = {"models.risk.model": performance_policy(), "models.risk.other": {"mode": "off"}}
    assert parse_performance_policies(json.dumps(policies)) == policies
