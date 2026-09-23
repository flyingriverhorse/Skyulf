"""Explicit alias promotion must preserve pinned comparisons and rollback receipts."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import save_local_pipeline
from skyulf.integrations.mlflow.local_model import log_local_model
from skyulf.integrations.mlflow.promotion import (
    AliasConflictError,
    AliasOutcomeUnknownError,
    LocalAliasAdmission,
    _admission,
    alias_resource_id,
    promote_candidate,
    rollback_promotion,
)
from skyulf.integrations.mlflow.registry import RegistryAccessError, register_model, resolve_model
from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run
from skyulf.integrations.mlflow.validation import (
    ModelComparisonReport,
    compare_registered_local_models,
)
from skyulf.pipeline import SkyulfPipeline

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def case(tmp_path: Path):
    """Use real local MLflow versions so alias and receipt behavior cannot be mocked away."""
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    name = "sm22-promotion"
    tracking = TrackingConfig(enabled=True, tracking_uri=uri, experiment_name=name)
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment(name, artifact_location=(tmp_path / "mlruns").as_uri())
    x = np.arange(12, dtype="float64")
    holdout_x = np.arange(20, 25, dtype="float64")
    heldout = pd.DataFrame({"x": holdout_x, "target": 2.0 * holdout_x})
    for offset in (10.0, 0.0):
        train = pd.DataFrame({"x": x, "target": 2.0 * x + offset})
        pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
        pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="target")
        artifact_path = tmp_path / f"model-{int(offset)}"
        save_local_pipeline(pipeline, artifact_path)
        with track_run(tracking, run_name=f"fit-{offset}") as run:
            assert run.run_id is not None
            model_uri = log_local_model(
                artifact_path, run_id=run.run_id, artifact_path="model", tracking_uri=uri
            )
        register_model(model_uri, name, tracking_uri=uri, registry_uri=uri)
    client.set_registered_model_alias(name, "champion", "1")
    candidate = resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri)
    champion = resolve_model(name, version="1", tracking_uri=uri, registry_uri=uri)
    report = compare_registered_local_models(
        candidate,
        champion,
        heldout,
        target_column="target",
        dataset_id="labels@5/heldout",
        metric="heldout_mse",
        min_improvement=1.0,
        quality_threshold=1.0,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert report.eligible
    return client, uri, name, heldout, report, LocalAliasAdmission(tmp_path / "locks")


def _promote(case, **changes):
    """Call the real promotion path with one validated local comparison."""
    _, uri, _, heldout, report, admission = case
    options = {
        "target_column": "target",
        "expected_champion_version": "1",
        "admission": admission,
        "max_rows": 10,
        "max_bytes": 10_000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    options.update(changes)
    selected = cast(ModelComparisonReport, options.pop("report", report))
    return promote_candidate(selected, heldout, **options)


def test_promote_and_rollback_persist_receipts(case) -> None:
    """An accepted candidate moves the alias and leaves auditable forward and reverse receipts."""
    client, uri, name, _, _, admission = case

    receipt = _promote(case)
    assert receipt.prior_version == "1"
    assert receipt.new_version == "2"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "2"
    tag = client.get_model_version(name, "2").tags[f"promotion_{receipt.event_id}"]
    assert len(tag.encode("utf-8")) <= 256
    assert json.loads(tag)["s"] == "committed"
    assert json.loads(client.get_registered_model(name).tags["champion_current_event"]) == {
        "event_id": receipt.event_id,
        "version": "2",
    }

    reversal = rollback_promotion(
        receipt,
        expected_current_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert reversal.prior_version == "2"
    assert reversal.new_version == "1"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"
    assert json.loads(client.get_registered_model(name).tags["champion_current_event"]) == {
        "event_id": reversal.event_id,
        "version": "1",
    }


def test_stale_alias_and_modified_report_cannot_promote(case) -> None:
    """The lock path must reject stale alias state and forged comparison fields."""
    client, _, name, _, report, _ = case
    client.set_registered_model_alias(name, "champion", "2")
    with pytest.raises(AliasConflictError, match="expected"):
        _promote(case)
    client.set_registered_model_alias(name, "champion", "1")

    tampered = replace(report, eligible=True, candidate_metrics={"heldout_mse": -1.0})
    with pytest.raises(ValueError, match="comparison"):
        _promote(case, report=tampered)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_missing_champion_and_contention_refuse_promotion(case) -> None:
    """The first alias must be initialized separately, and a second writer cannot enter."""
    client, _, name, _, _, admission = case
    client.delete_registered_model_alias(name, "champion")
    with pytest.raises(AliasConflictError, match="missing"):
        _promote(case)
    client.set_registered_model_alias(name, "champion", "1")
    with (
        admission.hold(alias_resource_id(name, "champion")),
        pytest.raises(AliasConflictError, match="admission"),
    ):
        _promote(case)


def test_rollback_rejects_newer_alias_state(case) -> None:
    """A receipt cannot roll back a different current model version."""
    client, uri, name, _, _, admission = case
    receipt = _promote(case)
    client.set_registered_model_alias(name, "champion", "1")
    with pytest.raises(AliasConflictError, match="expected"):
        rollback_promotion(
            receipt,
            expected_current_version="2",
            admission=admission,
            tracking_uri=uri,
            registry_uri=uri,
        )


def test_permission_denial_does_not_move_alias(case, monkeypatch) -> None:
    """A denied registry write is typed and leaves the previous alias selected."""
    client, _, name, _, _, _ = case
    error = mlflow.exceptions.MlflowException("denied")
    error.error_code = "PERMISSION_DENIED"

    def denied(*args, **kwargs):
        """Simulate a restricted registry principal at the mutation boundary."""
        raise error

    monkeypatch.setattr(client, "set_registered_model_alias", denied)
    monkeypatch.setattr("skyulf.integrations.mlflow.promotion._make_client", lambda *args: client)
    with pytest.raises(RegistryAccessError):
        _promote(case)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_lost_alias_write_response_is_reported_as_unknown(case, monkeypatch) -> None:
    """A committed alias move with a lost response must never be reported as safely denied."""
    client, _, name, _, _, _ = case
    original = client.set_registered_model_alias

    def move_then_fail(model_name, alias, version):
        """Simulate a transport failure after the registry accepted the alias move."""
        original(model_name, alias, version)
        raise mlflow.exceptions.MlflowException("response lost")

    monkeypatch.setattr(client, "set_registered_model_alias", move_then_fail)
    monkeypatch.setattr("skyulf.integrations.mlflow.promotion._make_client", lambda *args: client)
    with pytest.raises(AliasOutcomeUnknownError, match="inspect prepared event"):
        _promote(case)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "2"


def test_old_receipt_cannot_rollback_newer_promotion_to_same_version(case) -> None:
    """An old receipt must not undo a later promotion even when its version numbers match."""
    client, uri, name, _, _, admission = case
    old_receipt = _promote(case)
    rollback_promotion(
        old_receipt,
        expected_current_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    new_receipt = _promote(case)
    assert new_receipt.event_id != old_receipt.event_id
    with pytest.raises(AliasConflictError, match="superseded"):
        rollback_promotion(
            old_receipt,
            expected_current_version="2",
            admission=admission,
            tracking_uri=uri,
            registry_uri=uri,
        )
    assert str(client.get_model_version_by_alias(name, "champion").version) == "2"


def test_global_uc_registry_rejects_local_admission(tmp_path, monkeypatch) -> None:
    """Implicit UC registry configuration must still require distributed admission."""
    monkeypatch.setattr(mlflow, "get_registry_uri", lambda: "databricks-uc")
    with pytest.raises(ValueError, match="distributed"):
        _admission(LocalAliasAdmission(tmp_path / "locks"), None)
