"""Performance policies reuse saved outcomes and never infer health from absent labels."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_metrics as monitoring_metrics,
)


def policy():
    """Pin an explicit error baseline and one completed daily performance window."""
    return {
        "mode": "retrain",
        "metric": "mae",
        "direction": "lower",
        "baseline": {"kind": "training_holdout", "model_version": "2"},
        "tolerance": 1.0,
        "tolerance_mode": "absolute",
        "window_hours": 24,
        "label_delay_hours": 24,
        "minimum_labeled_rows": 2,
        "minimum_label_coverage": 0.5,
        "consecutive_windows": 1,
    }


def test_performance_measurement_joins_mature_labels_and_excludes_unscored_rows():
    """Coverage and error must describe eligible keyed outcomes, not row order."""
    now = datetime(2026, 10, 5, tzinfo=UTC)
    predictions = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prediction": [2.0, 5.0, 9.0, None],
            "scoring_status": ["predicted", "predicted", "predicted", "excluded"],
        }
    )
    labels = pd.DataFrame(
        {
            "id": [2, 1, 3, 4],
            "y": [3.0, 1.0, 0.0, 20.0],
            "available": [now, now, now + timedelta(days=1), now],
        }
    )
    result = monitoring_metrics.build_performance_report(
        predictions,
        labels,
        record_key_columns=("id",),
        target_column="y",
        result_available_at_column="available",
        as_of=now,
        task="regression",
    )
    assert result["labeled_rows"] == 2
    assert result["label_coverage"] == pytest.approx(2 / 3)
    assert result["values"]["mae"] == 1.5


def test_performance_measurement_missing_labels_has_no_values():
    """A successful scoring run cannot imply measured model performance."""
    result = monitoring_metrics.build_performance_report(
        pd.DataFrame({"id": [1, 2], "prediction": [0, 1]}),
        None,
        record_key_columns=("id",),
        target_column="y",
        result_available_at_column="available",
        as_of=datetime(2026, 10, 5, tzinfo=UTC),
        task="classification",
        classes=(0, 1),
    )
    assert result["labeled_rows"] == 0
    assert result["label_coverage"] == 0
    assert result["values"]["accuracy"] is None


def test_policy_observes_completed_window_with_current_label_cutoff(monkeypatch):
    """Late labels must be evaluated after maturity against the original saved predictions."""
    from types import SimpleNamespace

    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_performance as performance,
    )
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )

    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    config = MonitorConfig(
        environment="test",
        project="demo",
        model_name="cat.models.model",
        model_version="2",
        source_table="cat.data.source",
        prediction_table="cat.data.predictions",
        label_table="cat.data.labels",
        result_available_at_column="available",
        performance_policy=policy(),
    )
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            task="regression",
            classes=(),
            classification_probabilities=False,
            input_columns=("x",),
            pipeline_sha256="a" * 64,
        ),
        pipeline=SimpleNamespace(config={}),
    )
    spec = SimpleNamespace(record_key_columns=("id",), target_column="y")
    predictions = pd.DataFrame({"id": [1, 2], "prediction": [2.0, 5.0]})
    labels = pd.DataFrame({"id": [2, 1], "y": [3.0, 1.0], "available": [now, now]})
    calls = []

    def read(spark, cfg, version, keys, features, **kwargs):
        """Capture the real boundary contract while replacing external snapshot reads."""
        calls.append(kwargs)
        return (
            pd.DataFrame(),
            predictions,
            {
                "prediction_version": 7,
                "source_table_id": "source",
                "prediction_table_id": "predictions",
            },
            now,
        )

    monkeypatch.setattr(performance, "read_current_observation", read)
    monkeypatch.setattr(
        performance,
        "read_labels",
        lambda *args: (labels, {"label_version": 8, "label_table_id": "labels"}),
    )
    monkeypatch.setattr(performance, "load_performance_history", lambda *args: [])
    contract = performance.performance_contract(artifact, spec, config)
    baseline = {
        "model_version": "2",
        "metric": "mae",
        "contract_digest": contract,
        "value": 0.25,
        "labeled_rows": 10,
        "label_coverage": 1.0,
    }
    result = performance.observe_performance(
        None,
        "cat.monitoring",
        config,
        artifact,
        spec,
        {"model_version": "2", "performance_baseline": baseline},
        now,
    )
    assert calls[0]["start"] == datetime(2026, 10, 3, tzinfo=UTC)
    assert calls[0]["end"] == datetime(2026, 10, 4, tzinfo=UTC)
    assert calls[0]["as_of"] == now
    assert result["status"] == "degraded"
    assert result["current_value"] == 1.5
    assert result["action"] == "request_eligible"
    assert result["measurement"]["evidence"]["label_version"] == 8


def test_performance_retrain_mode_does_not_enable_drift():
    """Independent performance opt-in must reach shared guards without enabling drift."""
    import json

    from skyulf.integrations.databricks.jobs.lifecycle.retraining_task import retraining_policy

    result = retraining_policy(
        {"monitoring_performance_policies": json.dumps({"cat.models.model": policy()})}
    )
    assert result["mode"] == "retrain"
    assert result["drift_enabled"] is False


@pytest.mark.parametrize(
    "drift_enabled,performance_ready,expected",
    [
        (False, True, ["performance"]),
        (True, True, ["performance"]),
        (True, False, []),
        (False, False, []),
        (True, None, []),
        (False, None, []),
    ],
)
@pytest.mark.parametrize("drift_present", [False, True])
def test_performance_only_failure_reaches_training_data_gate(
    monkeypatch, drift_enabled, performance_ready, expected, drift_present
):
    """A healthy feature distribution must not mask measured performance degradation."""
    import json

    from skyulf.integrations.databricks.data.training import retraining_data
    from skyulf.integrations.databricks.jobs.lifecycle import retraining_task
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
        json_digest,
    )

    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    config = MonitorConfig(
        environment="test",
        project="demo",
        model_name="cat.models.model",
        model_version="2",
        source_table="cat.data.source",
        prediction_table="cat.data.predictions",
        label_table="cat.data.labels",
        result_available_at_column="available",
        performance_policy=policy(),
    )
    from skyulf.integrations.databricks.observability.monitoring.performance.performance_policy import (
        evaluate_performance,
    )

    current = {
        "model_version": "2",
        "metric": "mae",
        "contract_digest": "a" * 64,
        "value": 5.0 if performance_ready else 0.2,
        "labeled_rows": 0 if performance_ready is None else 5,
        "label_coverage": 0 if performance_ready is None else 1,
        "as_of": now.isoformat(),
        "window_start": "2026-10-03T00:00:00+00:00",
        "window_end": "2026-10-04T00:00:00+00:00",
    }
    baseline = {**current, "value": 0.25}
    verdict = evaluate_performance(policy(), current, baseline, [], now=now)
    from test_retraining_task import observation

    row = {
        "report_id": "b" * 64,
        "monitor_id": config.monitor_id,
        "config_digest": json_digest(config.payload()),
        "model_version": "2",
        "status": "drift" if drift_present else "healthy",
        "observed_at": now,
        "report_json": json.dumps(
            {
                "metrics": json.loads(observation(config, now)["report_json"])["metrics"]
                if drift_present
                else [],
                "performance": verdict,
            }
        ),
    }
    calls = []

    def assess(*args):
        """Record the actual training guard boundary independently of trigger evaluation."""
        calls.append(args)
        return {"changed_rows": 0}

    monkeypatch.setattr(retraining_data, "assess_training_data", assess)
    result = retraining_task._candidate(None, row, config, {}, now, 1, drift_enabled=drift_enabled)
    expected = (["drift"] if drift_enabled and drift_present else []) + expected
    assert result["triggers"] == expected
    assert result["drift_reason"] == (
        "disabled" if not drift_enabled else "ready" if drift_present else "no_drift"
    )
    assert len(calls) == bool(expected)
    if expected:
        assert result["status"] == "no_new_training_data"


def test_noop_without_new_commit_still_measures_mature_performance(monkeypatch):
    """A no-op must revisit delayed labels even without a new scoring manifest."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks.jobs.monitoring import monitoring_tasks as tasks
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )

    config = MonitorConfig(
        environment="test",
        project="demo",
        model_name="cat.models.model",
        model_version="2",
        source_table="cat.data.source",
        prediction_table="cat.data.predictions",
        label_table="cat.data.labels",
        result_available_at_column="available",
        performance_policy=policy(),
    )
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = {
        "monitoring_catalog": "cat",
        "monitoring_schema": "monitor",
        "monitoring_enabled": "true",
    }
    monkeypatch.setattr(tasks, "read_notebook_config", lambda _: {"model_name": config.model_name})
    monkeypatch.setattr(
        tasks,
        "_scoring_request",
        lambda *args: {
            "namespace": "cat.monitor",
            "configs": [config.payload()],
            "noop": True,
            "has_saved_batch": False,
        },
    )
    run = Mock(return_value={"results": [], "failed": 0})
    monkeypatch.setattr(tasks, "run_monitoring", run)
    result = tasks.run_scoring_monitor_notebook(None, dbutils)
    assert result == {"results": [], "failed": 0}
    assert run.call_args.kwargs["performance_only"] is True
    assert dbutils.jobs.taskValues.set.call_args.kwargs["value"]["status"] == "ready"


def test_population_identity_changes_when_label_table_is_replaced():
    """A same-name label replacement must not inherit a production baseline or failure streak."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_performance import (
        population_contract,
    )

    original = {
        "source_table_id": "source",
        "prediction_table_id": "predictions",
        "label_table_id": "labels",
    }
    assert population_contract("a" * 64, original) != population_contract(
        "a" * 64, original | {"label_table_id": "replacement"}
    )


def test_failed_observation_retains_unavailable_policy_window(monkeypatch):
    """Registry failures must break the streak and appear unavailable rather than disabled."""
    import json
    from unittest.mock import Mock

    from skyulf.integrations.databricks.observability.monitoring import monitoring as monitoring
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )

    config = MonitorConfig(
        environment="test",
        project="demo",
        model_name="cat.models.model",
        model_version="2",
        source_table="cat.data.source",
        prediction_table="cat.data.predictions",
        label_table="cat.data.labels",
        result_available_at_column="available",
        performance_policy=policy(),
    )
    monkeypatch.setattr(
        monitoring, "observe_model", Mock(side_effect=ValueError("missing registry"))
    )
    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    row = monitoring._observe_or_failure(
        None, config, as_of=now, window_start=now - timedelta(days=1), window_end=now
    )
    saved = json.loads(row["report_json"])["performance"]
    assert saved["status"] == "unavailable"
    assert saved["window_end"] == "2026-10-04T00:00:00+00:00"
    assert saved["consecutive_failures"] == 0
