"""One failed model cannot disappear or prevent other models from being monitored."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest


def test_models_are_enrolled_before_failures_and_other_models_continue(monkeypatch):
    """A missing registry model remains visible while independent observations still complete."""
    from skyulf.integrations.databricks import monitoring

    enrolled, saved = [], []
    monkeypatch.setattr(
        monitoring, "enroll_monitor", lambda spark, ns, cfg: enrolled.append(cfg.model_name)
    )
    monkeypatch.setattr(monitoring, "persist_report", lambda spark, ns, row: saved.append(row))

    def observe(spark, config, **kwargs):
        """Require all inventory rows before the first external observation starts."""
        assert len(enrolled) == 2
        if config.project == "broken":
            raise ValueError("Missing model")
        return {"report_id": "good", "status": "healthy", "model_name": config.model_name}

    monkeypatch.setattr(monitoring, "observe_model", observe)
    models = [
        {
            "environment": "test",
            "project": project,
            "model_name": f"cat.{project}.model",
            "model_version": "1",
            "source_table": "cat.data.source",
            "prediction_table": "cat.data.predictions",
        }
        for project in ("broken", "working")
    ]
    result = monitoring.run_monitoring(
        Mock(),
        "ops.monitoring",
        models,
        as_of=datetime(2026, 10, 1, tzinfo=UTC),
        tracking_uri=None,
        experiment_name=None,
    )
    assert [row["status"] for row in saved] == ["failed", "healthy"]
    assert result["failed"] == 1
    assert saved[0]["error_message"] == "ValueError: Missing model"


def test_disabled_models_remain_enrolled_without_observation(monkeypatch):
    """Operators can pause a monitor without losing its inventory and historical results."""
    from skyulf.integrations.databricks import monitoring

    enroll, observe = Mock(), Mock()
    monkeypatch.setattr(monitoring, "enroll_monitor", enroll)
    monkeypatch.setattr(monitoring, "observe_model", observe)
    result = monitoring.run_monitoring(
        Mock(),
        "ops.monitoring",
        [
            {
                "environment": "test",
                "project": "paused",
                "model_name": "cat.schema.model",
                "model_version": "1",
                "source_table": "cat.data.source",
                "prediction_table": "cat.data.predictions",
                "enabled": False,
            }
        ],
        as_of=datetime(2026, 10, 1, tzinfo=UTC),
        tracking_uri=None,
        experiment_name=None,
    )
    enroll.assert_called_once()
    observe.assert_not_called()
    assert result["disabled"] == 1


def test_completed_report_reuses_mlflow_run(tmp_path):
    """A retry after a persistence failure must not create a second finished evidence run."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.databricks.monitoring import _log_observation
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    uri = f"sqlite:///{(tmp_path / 'monitor.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("monitor", artifact_location=(tmp_path / "runs").as_uri())
    config = MonitorConfig(
        environment="test",
        project="retry",
        model_name="cat.schema.model",
        model_version="1",
        source_table="cat.schema.source",
        prediction_table="cat.schema.predictions",
    )
    report = {"status": "healthy", "metrics": []}
    row = {"report_id": "abcdef123456", "model_version": "1"}
    first = _log_observation(config, row, report, {}, uri, "monitor")
    second = _log_observation(config, row, report, {}, uri, "monitor")
    assert first == second
    assert len(client.search_runs([experiment])) == 1


def test_notebook_fails_after_results_are_saved(tmp_path, monkeypatch):
    """A model failure remains queryable and also makes the scheduled job visibly fail."""
    from skyulf.integrations.databricks import monitoring

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = {
        "monitoring_catalog": "cat",
        "monitoring_schema": "monitoring",
    }
    monkeypatch.setattr(monitoring, "initialize_monitoring_store", lambda *args: "cat.monitoring")
    run = Mock(return_value={"failed": 1})
    monkeypatch.setattr(monitoring, "run_monitoring", run)
    with pytest.raises(RuntimeError, match="results were saved"):
        monitoring.run_monitoring_notebook(Mock(), dbutils)
    run.assert_called_once()


def test_invalid_notebook_cutoff_does_not_create_tables(monkeypatch):
    """Malformed observation cutoffs must fail before provisioning the central namespace."""
    from skyulf.integrations.databricks import monitoring

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = {"as_of": "not-a-date"}
    initialize = Mock()
    monkeypatch.setattr(monitoring, "initialize_monitoring_store", initialize)
    with pytest.raises(ValueError):
        monitoring.run_monitoring_notebook(Mock(), dbutils)
    initialize.assert_not_called()


def test_inventory_run_does_not_overwrite_concurrent_producer_settings(monkeypatch):
    """Reading the inventory must never re-enroll an older snapshot over another repo's update."""
    from skyulf.integrations.databricks import monitoring

    rows = [
        {
            "environment": "test",
            "project": "other-repo",
            "model_name": "models.risk.model",
            "model_version": "3",
            "source_table": "data.risk.source",
            "prediction_table": "outputs.risk.predictions",
        }
    ]
    monkeypatch.setattr(monitoring, "load_enrolled_models", lambda spark, ns: rows)
    monkeypatch.setattr(
        monitoring, "enroll_monitor", Mock(side_effect=AssertionError("Stale inventory write"))
    )
    monkeypatch.setattr(
        monitoring,
        "_observe_or_failure",
        lambda spark, config, **kwargs: {
            "report_id": "report",
            "model_name": config.model_name,
            "status": "healthy",
        },
    )
    saved = []
    monkeypatch.setattr(monitoring, "persist_report", lambda spark, ns, row: saved.append(row))
    result = monitoring.run_monitoring(Mock(), "ops.monitoring")
    assert result["failed"] == 0
    assert saved[0]["model_name"] == "models.risk.model"
