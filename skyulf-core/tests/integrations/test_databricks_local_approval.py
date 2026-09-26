"""Manual approval must use an existing model and its pinned training evidence."""

import hashlib
import json
from dataclasses import asdict
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks import job_runtime, local_retraining, local_workflow
from skyulf.integrations.databricks.training_dates import TrainingDateSpec
from skyulf.integrations.mlflow.promotion import AliasConflictError

mlflow = pytest.importorskip("mlflow")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("evidence_mode", ["explicit", "saved"])
@pytest.mark.parametrize("cleanup", [False, True])
def test_manual_approval_reuses_registered_versions_and_replays_receipt(
    tmp_path, monkeypatch, engine, evidence_mode, cleanup
):
    """Approval, including bootstrap and retry, must never retrain or register a new model."""
    frame = pd.DataFrame(
        {
            "id": range(12),
            "x": [float(i) for i in range(12)],
            "target": [10.0 + 2 * i for i in range(12)],
            "event_time": pd.to_datetime(["2026-01-10"] * 8 + ["2026-02-10"] * 4, utc=True),
            "label_at": pd.to_datetime(["2026-01-11"] * 8 + ["2026-02-11"] * 4, utc=True),
        }
    )
    if cleanup:
        frame["age"] = [float(i) for i in range(12)]
        frame.loc[frame.id.isin([1, 8]), "age"] = -1.0
    frame["event_time"] = (
        frame["event_time"].dt.tz_convert("Asia/Tokyo").dt.strftime("%d/%m/%Y %H:%M")
    )
    frame["label_at"] = frame["label_at"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    read_snapshot = Mock(return_value=frame)
    monkeypatch.setattr(local_retraining, "read_training_snapshot", read_snapshot)
    monkeypatch.setattr(local_workflow, "read_training_snapshot", Mock(return_value=frame))
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("approval", artifact_location=(tmp_path / "runs").as_uri())
    config: dict[str, Any] = {
        "engine": engine,
        "training_table": "workspace.test.source",
        "training_version": 4,
        "model_name": "approval_model",
        "tracking_uri": uri,
        "registry_uri": uri,
        "score_model_selection": "pinned_version",
        "promotion_policy": "manual_approval",
        "score_handoff": "after_alias_change",
        "model_version": "1",
        "record_key_columns": ["id"],
        "input_columns": ["x"],
        "target_column": "target",
        "split_strategy": "temporal",
        "training_window_mode": "fixed_window",
        "filter_unavailable_results": True,
        "result_cutoff": "2026-03-01T00:00:00+00:00",
        "event_column": "event_time",
        "result_available_at_column": "label_at",
        "event_time_parsing": {"format": "%d/%m/%Y %H:%M", "timezone": "Asia/Tokyo"},
        "result_time_parsing": {"format": "%Y-%m-%dT%H:%M:%S%z"},
        "start": "2026-01-01T00:00:00+00:00",
        "holdout_start": "2026-02-01T00:00:00+00:00",
        "cutoff": "2026-03-01T00:00:00+00:00",
        "max_rows": 20,
        "max_input_mb": 2,
        "metric": "heldout_rmse",
        "min_improvement": 0.1,
        "quality_threshold": 100.0,
        "pipeline": {
            "preprocessing": [],
            "modeling": {"type": "linear_regression", "params": {"fit_intercept": False}},
        },
    }
    if cleanup:
        config["pre_split_steps"] = [
            {
                "name": "valid_age",
                "transformer": "ManualBounds",
                "params": {"bounds": {"age": {"lower": 0}}},
            }
        ]

    def operator_action(spark, config, action, **kwargs):
        """Exercise real operator parameter decoding and copyable evidence with MLflow artifacts."""
        experiment = kwargs.pop("experiment_name", None)
        artifact = kwargs.pop("artifact_path", None)
        parameters = {"lifecycle_action": action}
        for key, value in kwargs.items():
            if key == "promotion_receipt":
                parameters["promotion_receipt_json"] = json.dumps(asdict(value))
            else:
                parameters[key] = "none" if value is None else value
        outcome = job_runtime.run_bundle_action(
            spark,
            config,
            parameters,
            task_role="lifecycle",
            experiment_name=experiment,
            artifact_path=artifact,
        )
        assert outcome.score_requested == (action in {"approve", "rollback"})
        if action == "train":
            assert (
                outcome.next_actions["approve"]["candidate_version"] == outcome.result.model_version
            )
            assert outcome.next_actions["approve"]["expected_champion_version"] == (
                outcome.result.comparison.champion_version or "none"
            )
        if action == "approve" and outcome.result.kind == "promotion":
            decoded = json.loads(outcome.next_actions["rollback"]["promotion_receipt_json"])
            assert decoded == asdict(outcome.result)
        return outcome.result

    def train():
        """Create actual artifacts before the separate approval operation."""
        version = len(client.search_model_versions("name = 'approval_model'")) + 1
        return operator_action(
            None,
            config,
            "train",
            experiment_name="approval",
            artifact_path=tmp_path / f"model-{version}",
        )

    def approve(candidate, expected, **changes):
        """Pin the exact saved comparison instead of accepting a latest-model fallback."""
        digest = hashlib.sha256(
            json.dumps(asdict(candidate.comparison), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        options: dict[str, Any] = {
            "candidate_version": candidate.model_version,
            "comparison_sha256": digest if evidence_mode == "explicit" else "",
            "expected_champion_version": expected,
        }
        options.update(changes)
        return operator_action(None, config, "approve", **options)

    first = train()
    assert not client.get_registered_model(config["model_name"]).aliases.get("champion")
    # Never derive approval data from today's training configuration.
    config["max_input_mb"] = 1
    config["training_version"] = 99
    config["training_table"] = "workspace.changed.source"
    config["input_columns"] = ["changed_feature"]
    original_parsing = config["event_time_parsing"]
    config["event_time_parsing"] = {"format": "%d/%m/%Y %H:%M", "timezone": "UTC"}
    with monkeypatch.context() as no_training:
        no_training.setattr(
            local_workflow, "train_local_candidate", Mock(side_effect=AssertionError("fit"))
        )
        receipt = approve(first, None)
        assert receipt.kind == "initial" and receipt.new_version == "1"
        assert approve(first, None) == receipt
    assert read_snapshot.call_count == 2
    assert read_snapshot.call_args.args[1].version == 4
    replay_spec = read_snapshot.call_args.args[1]
    if cleanup:
        assert replay_spec.pre_split_steps == tuple(config["pre_split_steps"])
        assert "age" in replay_spec.source_columns and "age" not in replay_spec.input_columns
    assert replay_spec.event_time_parsing == TrainingDateSpec(
        format="%d/%m/%Y %H:%M", timezone="Asia/Tokyo"
    )
    assert replay_spec.result_time_parsing == TrainingDateSpec(format="%Y-%m-%dT%H:%M:%S%z")
    assert replay_spec.max_bytes == 1024 * 1024
    assert replay_spec.dataset_id == first.dataset_id
    assert len(client.search_model_versions("name = 'approval_model'")) == 1

    config["training_version"] = 4
    config["training_table"] = "workspace.test.source"
    config["input_columns"] = ["x"]
    config["event_time_parsing"] = original_parsing
    config["pipeline"]["modeling"]["params"]["fit_intercept"] = True
    second = train()
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "1"
    with pytest.raises(ValueError, match="digest"):
        approve(second, "1", comparison_sha256="0" * 64)
    original_report = asdict(second.comparison)
    client.log_dict(
        second.run_id, {**original_report, "quality_threshold": 999.0}, "candidate_comparison.json"
    )
    with pytest.raises(ValueError, match="digest"):
        approve(second, "1")
    client.log_dict(second.run_id, original_report, "candidate_comparison.json")
    with pytest.raises(ValueError, match="expected|Expected"):
        approve(second, "9")
    config["quality_threshold"] = 200.0
    with pytest.raises(ValueError, match="policy"):
        approve(second, "1")
    config["quality_threshold"] = 100.0
    client.set_registered_model_tag(config["model_name"], "pending_alias_event", "uncertain-test")
    with pytest.raises(AliasConflictError, match="pending"):
        approve(second, "1")
    client.delete_registered_model_tag(config["model_name"], "pending_alias_event")
    changed_frame = frame.copy()
    changed_frame.loc[8:, "target"] += 10.0
    read_snapshot.return_value = changed_frame
    with pytest.raises(ValueError, match="comparison"):
        approve(second, "1")
    read_snapshot.return_value = frame
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "1"
    with monkeypatch.context() as no_training:
        no_training.setattr(
            local_workflow, "train_local_candidate", Mock(side_effect=AssertionError("fit"))
        )
        promoted = approve(second, "1")
        assert promoted.kind == "promotion" and promoted.prior_version == "1"
        assert approve(second, "1") == promoted
    assert len(client.search_model_versions("name = 'approval_model'")) == 2
    assert config["model_version"] == "1"
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "2"
    client.set_registered_model_alias(config["model_name"], "previous_challenger", "1")
    with pytest.raises(AliasConflictError, match="history"):
        approve(second, "1")
    client.delete_registered_model_alias(config["model_name"], "previous_challenger")
    with pytest.raises(AliasConflictError):
        approve(first, None)

    tied = train()
    with pytest.raises(ValueError, match="approve|promotion"):
        approve(tied, "2")
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "2"

    tied_digest = hashlib.sha256(
        json.dumps(asdict(tied.comparison), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if evidence_mode == "saved":
        tied_digest = ""
    with monkeypatch.context() as no_training:
        no_training.setattr(
            local_workflow, "train_local_candidate", Mock(side_effect=AssertionError("fit"))
        )
        no_training.setattr(
            local_retraining,
            "read_training_snapshot",
            Mock(side_effect=AssertionError("data read")),
        )
        marker = client.get_registered_model(config["model_name"]).tags["champion_current_event"]
        client.set_registered_model_tag(
            config["model_name"],
            "champion_current_event",
            json.dumps({"event_id": promoted.event_id, "version": "1"}),
        )
        with pytest.raises(AliasConflictError, match="receipt disagrees"):
            operator_action(
                None,
                config,
                "reject",
                candidate_version=tied.model_version,
                comparison_sha256=tied_digest,
                expected_champion_version="2",
                rejection_reason="Retain the current production model",
            )
        assert (
            "approval_status"
            not in client.get_model_version(config["model_name"], tied.model_version).tags
        )
        client.set_registered_model_tag(config["model_name"], "champion_current_event", marker)
        rejected = operator_action(
            None,
            config,
            "reject",
            candidate_version=tied.model_version,
            comparison_sha256=tied_digest,
            expected_champion_version="2",
            rejection_reason="Retain the current production model",
        )
        assert (
            operator_action(
                None,
                config,
                "reject",
                candidate_version=tied.model_version,
                comparison_sha256=tied_digest,
                expected_champion_version="2",
                rejection_reason="Retain the current production model",
            )
            == rejected
        )
        with pytest.raises(AliasConflictError, match="rejected"):
            approve(tied, "2")
        config["promotion_policy"] = "automatic"
        rollback = operator_action(
            None, config, "rollback", promotion_receipt=promoted, expected_champion_version="2"
        )
        assert rollback.kind == "rollback" and rollback.new_version == "1"
        assert (
            operator_action(
                None, config, "rollback", promotion_receipt=promoted, expected_champion_version="2"
            )
            == rollback
        )
    assert len(client.search_model_versions("name = 'approval_model'")) == 3
    assert str(client.get_model_version_by_alias(config["model_name"], "challenger").version) == "3"
    assert config["model_version"] == "1"


@pytest.mark.parametrize("selection", ["pinned_version", "champion"])
def test_approval_refuses_automatic_policy_before_registry_access(selection):
    """Operator approval is an explicit manual-policy action, not a policy override."""
    with pytest.raises(ValueError, match="manual_approval"):
        local_workflow.run_action(
            None,
            {"score_model_selection": selection, "promotion_policy": "automatic"},
            "approve",
            candidate_version="2",
            comparison_sha256="a" * 64,
        )


def test_direct_approval_service_requires_explicit_manual_policy():
    """Calling the public service directly must not bypass its operator-action policy."""
    from skyulf.integrations.databricks.local_approval import approve_local_candidate

    with pytest.raises(ValueError, match="manual_approval"):
        approve_local_candidate(
            None,
            {"promotion_policy": "automatic"},
            candidate_version="2",
            comparison_sha256="a" * 64,
            expected_champion_version="1",
        )
