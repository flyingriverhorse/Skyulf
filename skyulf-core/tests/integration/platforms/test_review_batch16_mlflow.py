"""Model transport must use registered artifacts and retain only prediction state."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from test_mlflow_promotion import _promote, _stage, case  # noqa: F401 - pytest fixture registration

from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.integrations.mlflow.registration import registry
from skyulf.pipeline import SkyulfPipeline


def test_local_artifact_drops_training_weights_without_mutating_fit(tmp_path):
    """Row-level training weights are not prediction state and must stay out of saved models."""
    frame = pd.DataFrame({"x": np.arange(12.0), "target": np.arange(12.0) ** 2})
    weights = np.arange(1.0, 13.0)
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(frame, target_column="target", sample_weight=weights)
    before = pipeline.feature_engineer.train_sample_weight_.copy()
    expected = pipeline.predict(frame[["x"]])
    save_local_pipeline(pipeline, tmp_path / "model")
    restored = load_local_pipeline(tmp_path / "model")
    assert restored.pipeline.feature_engineer.train_sample_weight_ is None
    np.testing.assert_array_equal(pipeline.feature_engineer.train_sample_weight_, before)
    np.testing.assert_array_equal(restored.pipeline.predict(frame[["x"]]), expected)


@pytest.mark.parametrize(
    "bound_uri,explicit_uri,expected",
    [
        ("databricks-uc", "databricks-uc", "models:/catalog.schema.model/7"),
        ("databricks-uc://review", "databricks-uc://review", "models:/catalog.schema.model/7"),
        ("databricks-uc://implicit", None, "models:/catalog.schema.model/7"),
        ("DATABRICKS-UC://review", "DATABRICKS-UC://review", "models:/catalog.schema.model/7"),
        ("databricks://review", "databricks://review", "models:/catalog.schema.model/7"),
        ("sqlite:///registry.db", "sqlite:///registry.db", "registered-copy:/model/7"),
    ],
)
def test_resolve_uses_bound_registry_transport(
    monkeypatch, tmp_path, bound_uri, explicit_uri, expected
):
    """UC copies need scoped registry credentials; OSS copies need the explicitly bound store."""
    client = Mock()
    client._registry_uri = bound_uri
    client.get_model_version.return_value = SimpleNamespace(version="7", source="runs:/old/model")
    client.get_model_version_download_uri.return_value = "registered-copy:/model/7"
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(registry, "make_registry_client", lambda *args: client)
    monkeypatch.setattr(
        mlflow, "get_registry_uri", lambda: pytest.fail("Unrelated global registry")
    )
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    monkeypatch.setattr(
        mlflow.models.Model,
        "load",
        lambda *args: SimpleNamespace(metadata={"local_pipeline_digest": "digest"}, signature=None),
    )
    resolved = registry.resolve_model(
        "catalog.schema.model", version="7", registry_uri=explicit_uri
    )
    if expected.startswith("models:/"):
        client.get_model_version_download_uri.assert_not_called()
    else:
        client.get_model_version_download_uri.assert_called_once_with("catalog.schema.model", "7")
    assert download.call_args.kwargs["artifact_uri"] == expected
    assert download.call_args.kwargs["registry_uri"] == bound_uri
    assert resolved.digest == "digest"


@pytest.mark.parametrize("operation", ["stage", "promote", "rollback"])
def test_refused_first_alias_write_can_be_retried(case, monkeypatch, operation):
    """A verified permission refusal must not leave a pending marker blocking later promotion."""
    from skyulf.integrations.mlflow.lifecycle import promotion

    client, _, name, _, _, _ = case
    if operation != "stage":
        _stage(case)
    receipt = _promote(case) if operation == "rollback" else None

    def change_alias():
        """Exercise each public caller of the shared non-atomic alias transaction."""
        if operation == "stage":
            return _stage(case)
        if operation == "promote":
            return _promote(case)
        assert receipt is not None
        return promotion.rollback_promotion(
            receipt,
            expected_current_version="2",
            admission=case[5],
            tracking_uri=case[1],
            registry_uri=case[1],
        )

    original = client.set_registered_model_alias
    error = mlflow.exceptions.MlflowException("denied")
    error.error_code = "PERMISSION_DENIED"
    monkeypatch.setattr(client, "set_registered_model_alias", Mock(side_effect=error))
    monkeypatch.setattr(promotion, "make_registry_client", lambda *args: client)
    with pytest.raises(registry.RegistryAccessError):
        change_alias()
    assert not client.get_registered_model(name).tags.get("pending_alias_event")
    expected_before = "2" if operation == "rollback" else "1"
    assert str(client.get_model_version_by_alias(name, "champion").version) == expected_before
    monkeypatch.setattr(client, "set_registered_model_alias", original)
    result = change_alias()
    assert result.new_version == ("1" if operation == "rollback" else "2")


def test_failed_refusal_cleanup_keeps_pending_state(case, monkeypatch):
    """A registry cleanup failure remains explicitly uncertain instead of allowing unsafe retry."""
    from skyulf.integrations.mlflow.lifecycle import promotion

    client, _, name, _, _, _ = case
    _stage(case)
    error = mlflow.exceptions.MlflowException("denied")
    error.error_code = "PERMISSION_DENIED"
    monkeypatch.setattr(client, "set_registered_model_alias", Mock(side_effect=error))
    monkeypatch.setattr(client, "delete_registered_model_tag", Mock(side_effect=error))
    monkeypatch.setattr(promotion, "make_registry_client", lambda *args: client)
    with pytest.raises(promotion.AliasOutcomeUnknownError):
        _promote(case)
    assert client.get_registered_model(name).tags.get("pending_alias_event")
