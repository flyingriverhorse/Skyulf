"""Guard automatic training against unrelated, stale or unusable observations."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.monitoring_config import MonitorConfig, json_digest
from skyulf.integrations.databricks.retraining_task import (
    observation_decision,
    retraining_policy,
)


@pytest.mark.parametrize("prefix", ["", "dev_murat_"])
def test_train_lookup_preserves_development_name_prefix(prefix):
    """Personal targets must never resolve the shared production training job."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from skyulf.integrations.databricks.retraining_task import _training_job_id

    workspace = Mock()
    workspace.jobs.list.return_value = iter(
        [SimpleNamespace(job_id=7, settings=SimpleNamespace(name=prefix + "demo_train"))]
    )
    values = {"train_job_name": "demo_train", "score_job_name": prefix + "demo_score"}
    assert _training_job_id(workspace, values) == 7
    workspace.jobs.list.assert_called_once_with(name=prefix + "demo_train", limit=2)


def monitor():
    """Use a versioned model owned by one project."""
    return MonitorConfig(
        environment="test",
        project="demo",
        model_name="cat.models.model",
        model_version="2",
        source_table="cat.input.scoring",
        prediction_table="cat.output.predictions",
    )


def observation(config, now):
    """Build one fresh successful saved feature drift report."""
    return {
        "monitor_id": config.monitor_id,
        "config_digest": json_digest(config.payload()),
        "model_version": "2",
        "status": "drift",
        "observed_at": now,
        "report_json": json.dumps(
            {
                "metrics": [
                    {
                        "category": "drift",
                        "column_name": "x",
                        "metric_name": "psi",
                        "has_issue": True,
                        "status": "measured",
                    },
                    {
                        "category": "drift",
                        "column_name": "x",
                        "metric_name": "statistical_evidence",
                        "value": 0.01,
                        "threshold": 0.05,
                        "has_issue": False,
                        "status": "measured",
                        "evidence": {
                            "status": "supported",
                            "test": "ks_2samp",
                            "p_value": 0.01,
                            "adjusted_p_value": 0.01,
                            "significance_level": 0.05,
                            "reference_count": 100,
                            "current_count": 100,
                            "reason": None,
                        },
                    },
                ]
            }
        ),
    }


@pytest.mark.parametrize(
    "values",
    [
        {"on_drift": "yes"},
        {"on_drift": "retrain", "on_drift_cooldown_hours": "NaN"},
        {"on_drift": "retrain", "on_drift_min_new_training_rows": "0"},
    ],
)
def test_invalid_policy_fails_before_submission(values):
    """Mistyped policies must not silently start training."""
    with pytest.raises(ValueError):
        retraining_policy(values)


def test_policy_defaults_disabled():
    """Existing bundles do not acquire an automatic mutation on upgrade."""
    assert retraining_policy({})["mode"] == "disabled"


@pytest.mark.parametrize(
    "case,expected",
    [
        ("good", "ready"),
        ("stale", "stale_observation"),
        ("old_version", "superseded"),
        ("healthy", "no_drift"),
        ("degraded", "incomplete_drift"),
        ("quality", "quality_issue"),
        ("schema", "quality_issue"),
        ("unavailable", "incomplete_drift"),
        ("disabled", "disabled"),
    ],
)
def test_observation_requires_current_clean_distribution_drift(case, expected):
    """Data corruption and obsolete reports must not trigger a new model."""
    from dataclasses import replace

    now = datetime(2026, 10, 1, tzinfo=UTC)
    config = monitor()
    row = observation(config, now)
    if case == "stale":
        row["observed_at"] = now - timedelta(hours=25)
    elif case == "old_version":
        row["model_version"] = "1"
    elif case in {"healthy", "degraded"}:
        row["status"] = case
    elif case == "disabled":
        config = replace(config, enabled=False)
    elif case in {"quality", "schema", "unavailable"}:
        report = json.loads(row["report_json"])
        report["metrics"].append(
            {
                "category": "quality" if case == "quality" else "drift",
                "metric_name": "schema_missing" if case == "schema" else "ks_statistic",
                "status": "unavailable" if case == "unavailable" else "measured",
                "has_issue": case != "unavailable",
            }
        )
        row["report_json"] = json.dumps(report)
    assert observation_decision(row, config, now) == expected


def _ready_candidate(config, *, content="a" * 64):
    """Represent verified fresh training data tied to its exact enrolled model."""
    return {
        "status": "ready",
        "model_name": config.model_name,
        "monitor_id": config.monitor_id,
        "config_digest": json_digest(config.payload()),
        "baseline_model_version": config.model_version,
        "source_table": "cat.input.training",
        "content_sha256": content,
    }


def _job(name="demo_train", job_id=123):
    """Retain the server-returned name so lookup cannot trust a broad API filter."""
    return SimpleNamespace(job_id=job_id, settings=SimpleNamespace(name=name))


def test_multiple_ready_models_submit_one_training_job(monkeypatch):
    """Several drifting components must produce one idempotent whole-job request."""
    from skyulf.integrations.databricks import retraining_requests, retraining_task

    first = monitor()
    second = replace(first, model_name="cat.models.other")
    candidates = [_ready_candidate(second, content="b" * 64), _ready_candidate(first)]
    spark = Mock()
    workspace = Mock()
    workspace.jobs.list.return_value = iter([_job()])
    current = Mock(return_value={item.monitor_id: item for item in (first, second)})
    monkeypatch.setattr(retraining_task, "_current_configs", current)
    submit = Mock(return_value={"status": "submitted", "run_id": 456})
    monkeypatch.setattr(retraining_requests, "submit_retraining", submit)
    now = datetime(2026, 10, 1, tzinfo=UTC)
    values = {"train_job_name": "demo_train"}

    result = retraining_task._submit_candidates(
        spark, workspace, "cat.monitoring", values, {"cooldown_hours": 24}, candidates, now
    )

    identity = sorted(
        (item["model_name"], "2", "cat.input.training", item["content_sha256"])
        for item in candidates
    )
    submit.assert_called_once_with(
        spark,
        workspace,
        namespace="cat.monitoring",
        job_id=123,
        request_id=json_digest({"job_id": 123, "training_data": identity}),
        evidence={"models": candidates},
        cooldown_hours=24,
        now=now,
    )
    current.assert_called_once_with(spark, "cat.monitoring", values)
    assert result == {"status": "submitted", "run_id": 456, "models": candidates}


@pytest.mark.parametrize("change", ["removed", "promoted", "disabled", "threshold"])
def test_enrollment_reread_blocks_superseded_candidates(monkeypatch, change):
    """Changes during data assessment must stop obsolete observations before Jobs lookup."""
    from skyulf.integrations.databricks import retraining_requests, retraining_task

    original = monitor()
    changes = {
        "removed": {},
        "promoted": {original.monitor_id: replace(original, model_version="3")},
        "disabled": {original.monitor_id: replace(original, enabled=False)},
        "threshold": {original.monitor_id: replace(original, thresholds={"psi": 0.4})},
    }
    monkeypatch.setattr(retraining_task, "_current_configs", Mock(return_value=changes[change]))
    submit = Mock()
    monkeypatch.setattr(retraining_requests, "submit_retraining", submit)
    workspace = Mock()
    result = retraining_task._submit_candidates(
        Mock(),
        workspace,
        "cat.monitoring",
        {},
        {"cooldown_hours": 24},
        [_ready_candidate(original)],
        datetime(2026, 10, 1, tzinfo=UTC),
    )
    submit.assert_not_called()
    assert workspace.mock_calls == []
    assert result["status"] == "superseded"


def test_no_eligible_candidates_do_not_contact_jobs_or_inventory(monkeypatch):
    """Healthy or unchanged data must return without creating cloud submission work."""
    from skyulf.integrations.databricks import retraining_requests, retraining_task

    current = Mock()
    submit = Mock()
    monkeypatch.setattr(retraining_task, "_current_configs", current)
    monkeypatch.setattr(retraining_requests, "submit_retraining", submit)
    workspace = Mock()
    candidates = [{"status": "no_drift"}, {"status": "no_new_training_data"}]
    result = retraining_task._submit_candidates(
        Mock(),
        workspace,
        "cat.monitoring",
        {},
        {"cooldown_hours": 24},
        candidates,
        datetime(2026, 10, 1, tzinfo=UTC),
    )
    current.assert_not_called()
    submit.assert_not_called()
    assert workspace.mock_calls == []
    assert result == {"status": "not_requested", "models": candidates}


def test_disabled_notebook_publishes_status_without_cloud_work(monkeypatch):
    """The default-disabled task must only publish its local decision and optional display."""
    from skyulf.integrations.databricks import retraining_task

    enabled = Mock()
    monkeypatch.setattr(retraining_task, "_run_enabled", enabled)
    spark = Mock()
    workspace = Mock()
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = {}
    display = Mock()
    result = retraining_task.run_retraining_notebook(
        spark, dbutils, workspace=workspace, display_html=display
    )
    enabled.assert_not_called()
    assert spark.mock_calls == workspace.mock_calls == []
    dbutils.jobs.taskValues.get.assert_not_called()
    dbutils.jobs.taskValues.set.assert_called_once_with(
        key="retraining_result", value={"status": "disabled"}
    )
    assert "disabled" in display.call_args.args[0]
    assert result == {"status": "disabled"}


def test_training_job_lookup_accepts_one_exact_positive_identity():
    """Only one exactly named deployed train job can own the automatic request."""
    from skyulf.integrations.databricks.retraining_task import _training_job_id

    workspace = Mock()
    workspace.jobs.list.return_value = iter([_job()])
    assert _training_job_id(workspace, {"train_job_name": "demo_train"}) == 123
    workspace.jobs.list.assert_called_once_with(name="demo_train", limit=2)


@pytest.mark.parametrize(
    "jobs",
    [
        [],
        [_job(), _job(job_id=456)],
        [_job(job_id=None)],
        [_job(job_id=0)],
        [_job(job_id=-1)],
        [_job(job_id=True)],
        [_job(name="DEMO_TRAIN")],
        [_job(name="other")],
    ],
)
def test_training_job_lookup_rejects_missing_ambiguous_or_inexact_jobs(jobs):
    """API filters and malformed identities must not redirect the training mutation."""
    from skyulf.integrations.databricks.retraining_task import _training_job_id

    workspace = Mock()
    workspace.jobs.list.return_value = iter(jobs)
    with pytest.raises(ValueError):
        _training_job_id(workspace, {"train_job_name": "demo_train"})


def test_training_job_lookup_stops_after_second_match():
    """The SDK limit controls page size, so ambiguity must stop further pagination locally."""
    from skyulf.integrations.databricks.retraining_task import _training_job_id

    def matches():
        """Fail if the lookup requests another page after duplicate ownership is proven."""
        yield _job()
        yield _job(job_id=456)
        raise AssertionError("Lookup consumed results after the second matching job.")

    workspace = Mock()
    workspace.jobs.list.return_value = matches()
    with pytest.raises(ValueError):
        _training_job_id(workspace, {"train_job_name": "demo_train"})


@pytest.mark.parametrize(
    "status,expected", [("ready", True), ("cooldown", False), ("not_requested", False)]
)
def test_check_notebook_publishes_boolean_without_submission(monkeypatch, status, expected):
    """The Databricks condition consumes an explicit Boolean eligibility result."""
    from skyulf.integrations.databricks import retraining_task

    enabled = Mock(return_value={"status": status})
    monkeypatch.setattr(retraining_task, "_run_enabled", enabled)
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = {"on_drift": "retrain"}
    result = retraining_task.run_retraining_check_notebook(Mock(), dbutils, workspace=Mock())
    assert enabled.call_args.kwargs["preview_only"] is True
    dbutils.jobs.taskValues.set.assert_any_call(key="retraining_needed", value=expected)
    assert result["status"] == status


def test_disabled_check_publishes_false_without_cloud_work(monkeypatch):
    """Disabled automation must visibly take the false branch without remote reads."""
    from skyulf.integrations.databricks import retraining_task

    enabled = Mock()
    monkeypatch.setattr(retraining_task, "_run_enabled", enabled)
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = {}
    result = retraining_task.run_retraining_check_notebook(Mock(), dbutils)
    enabled.assert_not_called()
    dbutils.jobs.taskValues.set.assert_any_call(key="retraining_needed", value=False)
    assert result == {"status": "disabled"}
