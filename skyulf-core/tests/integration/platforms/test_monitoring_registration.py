"""Independent model projects enroll their actual scoring target in central Delta."""

import json
from unittest.mock import Mock

import pytest
from test_monitoring_config import performance_policy


@pytest.fixture(autouse=True)
def existing_monitoring_store(monkeypatch):
    """Isolate configuration assertions from the separately tested shared-store bootstrap."""
    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    monkeypatch.setattr(registration, "ensure_monitoring_store", Mock())


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_development_mode_never_registers_in_shared_monitoring(enabled):
    """Ephemeral dev models must not create or pause central inventory entries."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        monitoring_destination,
        register_deployed_monitor,
        register_scoring_monitor,
    )

    values = {
        "monitoring_deployment_mode": "development",
        "monitoring_enabled": enabled,
        "monitoring_catalog": "workspace",
        "monitoring_schema": "shared_monitoring",
    }
    assert monitoring_destination(values) is None
    assert register_scoring_monitor(None, {}, values, {}) is None
    assert register_deployed_monitor(None, {}, values, {}, activation_started_ms=1) is None


@pytest.mark.parametrize("version", ["2", "7"])
def test_automatic_enrollment_preserves_custom_drift_thresholds(monkeypatch, version):
    """Activation and scoring must use the same custom policy across model upgrades."""
    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    enroll = Mock()
    monkeypatch.setattr(registration, "enroll_monitor", enroll)
    values = settings(monitoring_drift_thresholds='{"psi": 0.35, "ks_statistic": 0.15}')
    registration.register_deployed_monitor(
        Mock(),
        workflow(),
        values,
        {
            "action": "train",
            "result": {
                "alias_change": {
                    "kind": "promotion",
                    "model_name": "models.risk.model",
                    "new_version": version,
                }
            },
        },
        activation_started_ms=1000,
    )
    result = registration.register_scoring_monitor(
        Mock(),
        workflow(),
        values,
        {"result": {"selected_model_name": "models.risk.model", "selected_model_version": version}},
    )
    assert result is not None
    assert result["config"]["thresholds"] == {"psi": 0.35, "ks_statistic": 0.15}
    assert len(enroll.call_args_list) == 2
    assert all(
        call.args[2].thresholds == {"psi": 0.35, "ks_statistic": 0.15}
        for call in enroll.call_args_list
    )


@pytest.mark.parametrize(
    "raw",
    [
        "invalid",
        "[]",
        "null",
        '{"unknown": 0.2}',
        '{"psi": true}',
        '{"psi": "0.2"}',
        '{"psi": 0}',
        '{"psi": -1}',
        '{"psi": NaN}',
        '{"psi": Infinity}',
        '{"psi": 1e400}',
    ],
)
def test_invalid_thresholds_block_scoring_before_inference(monkeypatch, raw):
    """Invalid policies cannot be discovered after writing model predictions."""
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings(monitoring_drift_thresholds=raw)
    monkeypatch.setattr(job_runtime, "read_notebook_config", lambda values: workflow())
    run = Mock(side_effect=AssertionError("Invalid thresholds reached inference"))
    monkeypatch.setattr(job_runtime, "run_bundle_action", run)
    with pytest.raises(ValueError, match="threshold"):
        job_runtime.run_score_notebook(Mock(), dbutils, exit_notebook=False)
    run.assert_not_called()


def settings(**changes):
    """Keep monitoring storage separate from all model project namespaces."""
    return {
        "monitoring_enabled": "true",
        "monitoring_catalog": "ops",
        "monitoring_schema": "monitoring",
        "monitoring_environment": "prod",
        "monitoring_project": "risk",
        **changes,
    }


def workflow(**changes):
    """Represent already-resolved model and scoring table bindings."""
    return {
        "model_name": "models.risk.model",
        "score_source_table": "features.risk.source",
        "prediction_table": "outputs.risk.predictions",
        "model_change_mode": "full_rebuild",
        "max_rows": 500,
        "max_input_mb": 4,
        **changes,
    }


def test_enrollment_selects_only_exact_model_performance_policy():
    """A policy for another model cannot activate performance on this one."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        build_monitor_enrollment_config,
    )

    policies = {
        "models.risk.model": performance_policy(),
        "models.risk.other": {"mode": "off"},
    }
    values = settings(
        monitoring_label_table="labels.risk.actuals",
        monitoring_result_available_at_column="available_at",
        monitoring_performance_policies=json.dumps(policies),
    )
    selected = build_monitor_enrollment_config(workflow(), values, "2")
    other = build_monitor_enrollment_config(
        workflow(model_name="models.risk.third", training_layout="model_competition"), values, "2"
    )
    assert selected.performance_policy == performance_policy()
    assert "performance_policy" not in other.payload()


def test_scoring_preflight_rejects_invalid_policy_for_other_model():
    """Malformed Bundle settings must fail before inference even for an unselected model."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        validate_monitoring_settings,
    )

    values = settings(monitoring_performance_policies='{"models.risk.other": {"mode": "report"}}')
    with pytest.raises(ValueError):
        validate_monitoring_settings(values, workflow())


def test_single_model_preflight_rejects_unmatched_policy_key():
    """A mistyped model name must not silently leave the intended policy disabled."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        validate_monitoring_settings,
    )

    values = settings(monitoring_performance_policies='{"models.risk.other": {"mode": "off"}}')
    with pytest.raises(ValueError, match="model"):
        validate_monitoring_settings(values, workflow(training_layout="single_model"))


def test_single_model_preflight_rejects_extra_policy_key():
    """A valid policy must not hide a second mistyped name in a one-model project."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        validate_monitoring_settings,
    )

    values = settings(
        monitoring_performance_policies=json.dumps(
            {"models.risk.model": {"mode": "off"}, "models.risk.modle": {"mode": "off"}}
        )
    )
    with pytest.raises(ValueError, match="model"):
        validate_monitoring_settings(values, workflow(training_layout="single_model"))


def test_registration_pins_scored_version_not_mutable_alias(monkeypatch):
    """Alias movement after scoring cannot enroll a different model or logical output view."""
    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    enrolled = []
    monkeypatch.setattr(
        registration, "enroll_monitor", lambda spark, ns, cfg, **kwargs: enrolled.append((ns, cfg))
    )
    payload = {
        "result": {
            "selected_model_name": "models.risk.model",
            "selected_model_version": "7",
            "noop": True,
        }
    }
    result = registration.register_scoring_monitor(Mock(), workflow(), settings(), payload)
    namespace, config = enrolled[0]
    assert namespace == "ops.monitoring"
    assert config.model_version == "7" and config.model_alias is None
    assert config.prediction_table == "outputs.risk.predictions_v7"
    assert config.source_table == "features.risk.source"
    assert config.max_bytes == 4 * 1024 * 1024
    assert result is not None
    assert result["monitor_id"] == config.monitor_id


def test_default_opt_out_and_pending_recovery_do_not_register(monkeypatch):
    """Unconfigured projects and unsuccessful scoring branches cannot create inventory rows."""
    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    enroll = Mock(side_effect=AssertionError("Unexpected registration"))
    monkeypatch.setattr(registration, "enroll_monitor", enroll)
    assert registration.register_scoring_monitor(Mock(), workflow(), {}, {}) is None
    assert (
        registration.register_scoring_monitor(
            Mock(), workflow(), settings(), {"recovery_required": True}
        )
        is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"monitoring_catalog": ""},
        {"monitoring_schema": ""},
        {"monitoring_enabled": "maybe"},
        {"monitoring_expected_interval_hours": "nan"},
        {"monitoring_label_table": "labels.prod.truth"},
    ],
)
def test_invalid_monitoring_settings_fail_before_scoring(changes):
    """Explicit invalid monitoring settings cannot fail only after predictions are committed."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        validate_monitoring_settings,
    )

    with pytest.raises(ValueError):
        validate_monitoring_settings(settings(**changes), workflow())


def test_explicit_disable_keeps_enrollment_history(monkeypatch):
    """A configured destination plus disabled flag must pause an existing monitor."""
    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    saved = []
    monkeypatch.setattr(
        registration, "enroll_monitor", lambda spark, ns, cfg, **kwargs: saved.append(cfg)
    )
    registration.register_scoring_monitor(
        Mock(),
        workflow(),
        settings(monitoring_enabled="false"),
        {"result": {"selected_model_name": "models.risk.model", "selected_model_version": "2"}},
    )
    assert saved[0].enabled is False


def test_configured_storage_enables_monitoring_without_a_flag(monkeypatch):
    """A model using the shared store must not silently enroll as disabled by default."""
    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    values = settings()
    values.pop("monitoring_enabled")
    enroll = Mock()
    monkeypatch.setattr(registration, "enroll_monitor", enroll)
    registration.register_scoring_monitor(
        Mock(),
        workflow(),
        values,
        {"result": {"selected_model_name": "models.risk.model", "selected_model_version": "2"}},
    )
    assert enroll.call_args.args[2].enabled is True


def test_wrong_scored_model_cannot_replace_enrollment(monkeypatch):
    """The producer must not register an outcome from an unrelated model project."""
    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    monkeypatch.setattr(registration, "enroll_monitor", Mock(side_effect=AssertionError))
    with pytest.raises(ValueError, match="scored model"):
        registration.register_scoring_monitor(
            Mock(),
            workflow(),
            settings(),
            {
                "result": {
                    "selected_model_name": "models.other.model",
                    "selected_model_version": "2",
                }
            },
        )


def test_model_set_monitoring_requires_component_outputs():
    """Combined business outputs cannot substitute for individual model predictions."""
    from skyulf.integrations.databricks.model_sets.monitoring_model_set import (
        validate_set_monitoring,
    )

    with pytest.raises(ValueError, match="component"):
        validate_set_monitoring(settings(), {"publication": {"mode": "combined_only"}})


def test_score_notebook_registers_actual_result(monkeypatch):
    """Successful scoring must automatically enroll even when no new prediction rows were needed."""
    import json

    from skyulf.integrations.databricks.jobs.shared import job_runtime
    from skyulf.integrations.databricks.lifecycle.local_workflow import BundleActionResult
    from skyulf.integrations.databricks.scoring.incremental.local_incremental import (
        IncrementalBatchResult,
    )

    values = settings()
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = values
    monkeypatch.setattr(job_runtime, "read_notebook_config", lambda values: workflow())
    outcome = IncrementalBatchResult(0, 0, 0, 0, None, None, True, "models.risk.model", "7")
    monkeypatch.setattr(
        job_runtime,
        "run_bundle_action",
        lambda *args, **kwargs: BundleActionResult("score", outcome, False, {}),
    )
    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    saved = []
    monkeypatch.setattr(
        registration, "enroll_monitor", lambda spark, ns, cfg, **kwargs: saved.append(cfg)
    )
    output = json.loads(job_runtime.run_score_notebook(Mock(), dbutils, exit_notebook=False))
    assert output["monitoring"]["monitor_id"] == saved[0].monitor_id
    assert saved[0].prediction_table == "outputs.risk.predictions_v7"


def test_invalid_destination_blocks_score_notebook_before_inference(monkeypatch):
    """A missing monitoring location must not be discovered after a scoring commit."""
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings(monitoring_schema="")
    monkeypatch.setattr(job_runtime, "read_notebook_config", lambda values: workflow())
    monkeypatch.setattr(
        job_runtime, "run_bundle_action", Mock(side_effect=AssertionError("Scoring ran"))
    )
    with pytest.raises(ValueError, match="monitoring_schema"):
        job_runtime.run_score_notebook(Mock(), dbutils, exit_notebook=False)


def test_recovery_notebook_registers_completed_branch(monkeypatch):
    """A recovered batch must enroll the same pinned version after recovery succeeds."""
    import json

    from skyulf.integrations.databricks.observability.monitoring import monitoring_registration
    from skyulf.integrations.databricks.scoring.incremental import scoring_recovery

    monkeypatch.setattr(scoring_recovery, "_enabled_config", lambda _: (settings(), workflow()))
    monkeypatch.setattr(scoring_recovery, "_recovery_needed", lambda _: True)
    monkeypatch.setattr(scoring_recovery, "validate_recovery_request", lambda _: {})
    monkeypatch.setattr(
        scoring_recovery,
        "_recover",
        lambda *args: {
            "result": {"selected_model_name": "models.risk.model", "selected_model_version": "7"}
        },
    )
    saved = []
    monkeypatch.setattr(
        monitoring_registration,
        "enroll_monitor",
        lambda spark, ns, cfg, **kwargs: saved.append(cfg),
    )
    output = json.loads(
        scoring_recovery.run_cdf_recovery_notebook(Mock(), Mock(), exit_notebook=False)
    )
    assert output["monitoring"]["monitor_id"] == saved[0].monitor_id


def test_inventory_conflict_retry_is_bounded_and_does_not_swallow_permission_errors(monkeypatch):
    """Cross-repo write conflicts retry, while access failures remain visible immediately."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store

    monkeypatch.setattr(store.time, "sleep", lambda _: None)
    spark = Mock()
    spark.sql.side_effect = [RuntimeError("[DELTA_CONCURRENT_APPEND]"), Mock()]
    store.merge_with_retry(spark, "MERGE")
    assert spark.sql.call_count == 2
    spark.sql.reset_mock()
    spark.sql.side_effect = PermissionError("Denied")
    with pytest.raises(PermissionError):
        store.merge_with_retry(spark, "MERGE")
    assert spark.sql.call_count == 1
    spark.sql.reset_mock()
    spark.sql.side_effect = RuntimeError("[DELTA_CONCURRENT_APPEND]")
    with pytest.raises(RuntimeError):
        store.merge_with_retry(spark, "MERGE")
    assert spark.sql.call_count == 3
