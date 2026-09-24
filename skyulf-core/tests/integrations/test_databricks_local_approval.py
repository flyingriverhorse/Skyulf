"""Manual approval must use an existing model and its pinned training evidence."""

import hashlib
import json
from dataclasses import asdict
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks import local_retraining, local_workflow
from skyulf.integrations.mlflow.promotion import AliasConflictError

mlflow = pytest.importorskip("mlflow")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_manual_approval_reuses_registered_versions_and_replays_receipt(
    tmp_path, monkeypatch, engine
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
    read_snapshot = Mock(return_value=frame)
    monkeypatch.setattr(local_retraining, "read_training_snapshot", read_snapshot)
    monkeypatch.setattr(local_workflow, "read_training_snapshot", Mock(return_value=frame))
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("approval", artifact_location=(tmp_path / "runs").as_uri())
    config = {
        "engine": engine,
        "training_table": "workspace.test.source",
        "training_version": 4,
        "model_name": "approval_model",
        "tracking_uri": uri,
        "registry_uri": uri,
        "score_model_selection": "pinned_version",
        "promotion_policy": "manual_approval",
        "model_version": "1",
        "row_keys": ["id"],
        "input_columns": ["x"],
        "target_column": "target",
        "event_column": "event_time",
        "label_time_column": "label_at",
        "start": "2026-01-01T00:00:00+00:00",
        "holdout_start": "2026-02-01T00:00:00+00:00",
        "cutoff": "2026-03-01T00:00:00+00:00",
        "max_rows": 20,
        "max_bytes": 100_000,
        "metric": "heldout_rmse",
        "min_improvement": 0.1,
        "quality_threshold": 100.0,
        "pipeline": {
            "preprocessing": [],
            "modeling": {"type": "linear_regression", "params": {"fit_intercept": False}},
        },
    }

    def train():
        """Create actual artifacts before the separate approval operation."""
        version = len(client.search_model_versions("name = 'approval_model'")) + 1
        return local_workflow.run_action(
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
            "comparison_sha256": digest,
            "expected_champion_version": expected,
        }
        options.update(changes)
        return local_workflow.run_action(None, config, "approve", **options)

    first = train()
    assert not client.get_registered_model(config["model_name"]).aliases.get("champion")
    # Never derive approval data from today's training configuration.
    config["training_version"] = 99
    with monkeypatch.context() as no_training:
        no_training.setattr(
            local_workflow, "train_local_candidate", Mock(side_effect=AssertionError("fit"))
        )
        receipt = approve(first, None)
        assert receipt.kind == "initial" and receipt.new_version == "1"
        assert approve(first, None) == receipt
    assert read_snapshot.call_count == 2
    assert read_snapshot.call_args.args[1].version == 4
    assert len(client.search_model_versions("name = 'approval_model'")) == 1

    config["training_version"] = 4
    config["pipeline"]["modeling"]["params"]["fit_intercept"] = True
    second = train()
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "1"
    with pytest.raises(ValueError, match="digest"):
        approve(second, "1", comparison_sha256="0" * 64)
    with pytest.raises(ValueError, match="expected|Expected"):
        approve(second, "9")
    config["quality_threshold"] = 200.0
    with pytest.raises(ValueError, match="policy"):
        approve(second, "1")
    config["quality_threshold"] = 100.0
    config["input_columns"] = ["wrong_feature"]
    with pytest.raises(ValueError, match="data contract"):
        approve(second, "1")
    config["input_columns"] = ["x"]
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
    with pytest.raises(AliasConflictError):
        approve(first, None)

    tied = train()
    with pytest.raises(ValueError, match="approve|promotion"):
        approve(tied, "2")
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "2"


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
