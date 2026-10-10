"""Optional scheduled jobs compose durable rollout and native publication services."""

import json
import runpy
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("mlflow")


@pytest.fixture
def api():
    """Require the actual notebook service rather than a template-only placeholder."""
    return import_module("skyulf.integrations.databricks.jobs.serving_job")


@pytest.fixture
def job(api, monkeypatch):
    """Preserve job orchestration while stubbing SDK and proven controller boundaries."""
    from skyulf.integrations.databricks.serving.rollout import RolloutResult
    from skyulf.integrations.databricks.serving.rollout_policy import RolloutState

    now = datetime(2026, 10, 10, tzinfo=UTC).isoformat()
    state = RolloutState(
        "rollout", "endpoint", "main.ml.model", "1", "main.ml.model", "2", 0, now, now
    )
    settings = {
        "enabled": True,
        "exclusive_writer": True,
        "run_id": "run",
        "comparison_sha256": "a" * 64,
        "approval": {"model_name": "main.ml.model", "metric": "r2", "quality_threshold": 0.5},
        "tracking_uri": "databricks",
        "registry_uri": "databricks-uc",
    }
    approval = settings["approval"] | {
        "tracking_uri": "databricks",
        "registry_uri": "databricks-uc",
    }
    pin = {
        "auto_promote": True,
        "comparison_sha256": "a" * 64,
        "config_sha256": api.approval_config_digest(approval),
    }
    record = {
        "receipt_id": "receipt",
        "status": "COMMITTED",
        "state": asdict(state),
        "promotion": pin,
        "plan": {},
    }
    store = Mock()
    store.run_id = "run"
    store.load.return_value = record
    monkeypatch.setattr(api, "MLflowRolloutStore", Mock(return_value=store))
    reconciled = RolloutResult(state, "COMMITTED", "receipt")
    reconcile = Mock(return_value=reconciled)
    monkeypatch.setattr(api, "reconcile_rollout", reconcile)
    producer, advance, promote = Mock(), Mock(return_value=reconciled), Mock()
    monkeypatch.setattr(api, "rollout_plan_from_dict", Mock(return_value=object()))
    monkeypatch.setattr(api, "observe_bootstrap_rollout", producer)
    monkeypatch.setattr(api, "advance_rollout", advance)
    monkeypatch.setattr(api, "promote_completed_rollout", promote)
    persist = Mock()
    monkeypatch.setattr(api, "persist_rollout_observation", persist)
    return settings, store, reconciled, producer, advance, promote, persist


def test_disabled_jobs_do_not_open_clients_or_mutate(api):
    """Generated jobs remain inert until explicitly configured and enabled."""
    client = Mock()
    result = api.run_daily_rollout(None, {"enabled": False}, client=client, registry_client=client)
    assert result["status"] == "DISABLED"
    assert client.mock_calls == []


@pytest.mark.parametrize("exclusive", [False, "true", None])
def test_job_requires_explicit_exclusive_writer_contract(api, job, exclusive):
    """Per-job serialization cannot silently stand in for shared endpoint ownership."""
    settings, store, result, producer, advance, promote, persist = job
    settings["exclusive_writer"] = exclusive
    with pytest.raises(ValueError, match="exclusive_writer"):
        api.run_daily_rollout(None, settings, client=Mock(), registry_client=Mock())
    advance.assert_not_called()


def test_bootstrap_evidence_is_persisted_before_traffic_mutation(api, job):
    """Scheduled traffic decisions retain the actual producer details for audit."""
    settings, store, result, producer, advance, promote, persist = job
    events = []
    persist.side_effect = lambda *args: events.append("saved")
    advance.side_effect = lambda *args, **kwargs: events.append("advance") or result
    output = api.run_daily_rollout(None, settings, client=Mock(), registry_client=Mock())
    assert events == ["saved", "advance"]
    assert output["state"]["challenger_percentage"] == 0
    producer.assert_called_once()
    promote.assert_not_called()


def test_failed_observation_storage_prevents_traffic_write(api, job):
    """A job cannot mutate traffic when its evidence cannot be retained and verified."""
    settings, store, result, producer, advance, promote, persist = job
    persist.side_effect = ValueError("observation readback differs")
    with pytest.raises(ValueError, match="readback"):
        api.run_daily_rollout(None, settings, client=Mock(), registry_client=Mock())
    advance.assert_not_called()


def test_reconciled_prepared_stage_does_not_advance_again(api, job):
    """Restart recovery commits at most the previously attempted traffic step."""
    settings, store, result, producer, advance, promote, persist = job
    store.load.return_value["status"] = "PREPARED"
    output = api.run_daily_rollout(None, settings, client=Mock(), registry_client=Mock())
    assert output["status"] == "COMMITTED"
    producer.assert_not_called()
    advance.assert_not_called()


def test_completed_stage_automatically_invokes_guarded_promotion(api, job):
    """The user's selected automatic champion behavior follows durable completion."""
    from dataclasses import replace

    from skyulf.integrations.mlflow.lifecycle.promotion import AliasChangeReceipt

    settings, store, result, producer, advance, promote, persist = job
    complete = replace(
        result, state=replace(result.state, phase="COMPLETE", challenger_percentage=100)
    )
    api.reconcile_rollout.return_value = complete
    promote.return_value = AliasChangeReceipt(
        "event", "promotion", "main.ml.model", "champion", "1", "2", "a" * 64, None
    )
    output = api.run_daily_rollout(None, settings, client=Mock(), registry_client=Mock())
    assert output["promotion"]["new_version"] == "2"
    promote.assert_called_once()
    producer.assert_not_called()


def test_changed_saved_comparison_rejected_before_observation(api, job):
    """A new job setting cannot replace the original promotion evidence pin."""
    settings, store, result, producer, advance, promote, persist = job
    settings["comparison_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="comparison"):
        api.run_daily_rollout(None, settings, client=Mock(), registry_client=Mock())
    producer.assert_not_called()


def test_online_job_exposes_pending_sync_without_claiming_readiness(api, monkeypatch):
    """Job submission success remains distinct from a completed native sync."""
    publish, status = (
        Mock(return_value={"status": "SUBMITTED"}),
        Mock(return_value={"status": "PENDING"}),
    )
    monkeypatch.setattr(api, "publish_online_features", publish)
    monkeypatch.setattr(api, "online_publication_status", status)
    settings = {
        "enabled": True,
        "exclusive_writer": True,
        "source_table": "main.f.source",
        "online_table": "main.f.online",
        "online_store": "store",
        "source_table_id": "id",
    }
    output = api.run_online_publication(None, settings, client=Mock(), feature_client=Mock())
    assert output["status"] == "PENDING"
    publish.assert_called_once()


def test_evidence_artifact_readback_checks_full_content(api, tmp_path):
    """MLflow acknowledgement alone is insufficient to accept producer evidence."""
    from skyulf.integrations.databricks.serving.rollout_evidence import RolloutEvidenceResult
    from skyulf.integrations.databricks.serving.rollout_policy import RolloutEvidence

    now = datetime(2026, 10, 10, tzinfo=UTC).isoformat()
    evidence = RolloutEvidence(
        "rollout",
        "endpoint",
        "main.ml.model",
        "1",
        "main.ml.model",
        "2",
        0,
        now,
        now,
        now,
        now,
        "PASS",
        "BOOTSTRAP",
        "passed",
    )
    observed = RolloutEvidenceResult(evidence, {"smoke": "actual result"})
    path = tmp_path / "observation.json"
    path.write_text("{}")
    client = Mock()
    client.download_artifacts.return_value = str(path)
    with pytest.raises(ValueError, match="observation"):
        api.persist_rollout_observation(SimpleNamespace(client=client, run_id="run"), observed)
    client.log_dict.assert_called_once()


@pytest.mark.parametrize(
    "entrypoint", ["run_daily_rollout_notebook", "run_online_publication_notebook"]
)
def test_notebook_entrypoints_return_mappings_for_generated_summary(api, tmp_path, entrypoint):
    """Generated notebook summary and exit serialize the same mapping exactly once."""
    path = tmp_path / "serving.yml"
    path.write_text("rollout:\n  enabled: false\nonline_publication:\n  enabled: false\n")
    dbutils = Mock()
    dbutils.widgets.get.return_value = str(path)
    result = getattr(api, entrypoint)(None, dbutils)
    assert result == {"status": "DISABLED"}


@pytest.mark.parametrize("job_name", ["daily_rollout", "online_publication"])
def test_real_disabled_notebook_exits_with_one_json_object(tmp_path, job_name):
    """The actual template and runtime must agree before any remote work is enabled."""
    config = tmp_path / "serving.yml"
    config.write_text("rollout:\n  enabled: false\nonline_publication:\n  enabled: false\n")
    dbutils = Mock()
    dbutils.widgets.get.return_value = str(config)
    dbutils.widgets.getAll.return_value = {}
    notebook = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/jobs"
        / f"{job_name}.py"
    )
    runpy.run_path(
        str(notebook), run_name="__main__", init_globals={"spark": None, "dbutils": dbutils}
    )
    dbutils.notebook.exit.assert_called_once()
    assert json.loads(dbutils.notebook.exit.call_args.args[0]) == {"status": "DISABLED"}
