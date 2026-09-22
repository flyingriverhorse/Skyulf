"""Contract tests for the optional MLflow model registry adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import InferenceBundle, build_bundle

mlflow = pytest.importorskip("mlflow")
from skyulf.integrations.mlflow.model import log_model
from skyulf.integrations.mlflow.registry import (
    RegistryAccessError,
    RegistryDependencyError,
    RegistryModelNotFoundError,
    register_model,
    resolve_model,
)
from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run
from skyulf.pipeline import SkyulfPipeline


def _config(tmp_path: Path, name: str) -> TrackingConfig:
    """Create an isolated tracking and registry store for one test."""
    db_path = (tmp_path / f"{name}.db").resolve()
    uri = f"sqlite:///{db_path.as_posix()}"
    return TrackingConfig(enabled=True, tracking_uri=uri, experiment_name=name)


def _bundle() -> tuple[InferenceBundle, pd.DataFrame]:
    """Fit a tiny regression bundle suitable for local registry tests."""
    frame = pd.DataFrame({"x": np.arange(8, dtype="float64"), "target": np.arange(8) * 2.0})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    return build_bundle(pipeline, input_stage="raw", feature_order=("x",)), frame[["x"]].head(2)


def _publish(tmp_path: Path, name: str = "skyulf-registry") -> tuple[str, TrackingConfig]:
    """Log and explicitly register one model in a test-owned local store."""
    bundle, _ = _bundle()
    config = _config(tmp_path, name)
    with track_run(config, run_name="registry-fit") as run:
        model_uri = log_model(
            bundle,
            run_id=run.run_id,
            artifact_path="model",
            tracking_uri=config.tracking_uri,
        )
    register_model(
        model_uri,
        name,
        tracking_uri=config.tracking_uri,
        registry_uri=config.tracking_uri,
    )
    return model_uri, config


def test_register_and_resolve_concrete_version_with_signature_and_digest(tmp_path: Path) -> None:
    """A registered model resolves to a versioned URI with Skyulf metadata."""
    model_uri, config = _publish(tmp_path)
    client = mlflow.MlflowClient(tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri)
    version = client.get_model_version("skyulf-registry", "1").version

    resolved = resolve_model(
        "skyulf-registry",
        version=version,
        tracking_uri=config.tracking_uri,
        registry_uri=config.tracking_uri,
    )

    assert resolved.name == "skyulf-registry"
    assert resolved.version == "1"
    assert resolved.model_uri == "models:/skyulf-registry/1"
    assert resolved.signature is not None
    assert resolved.digest
    assert model_uri.startswith("runs:/")


def test_alias_resolves_once_to_concrete_version(tmp_path: Path) -> None:
    """Alias resolution returns a stable version URI rather than a moving alias URI."""
    model_uri, config = _publish(tmp_path)
    client = mlflow.MlflowClient(tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri)
    client.set_registered_model_alias("skyulf-registry", "champion", "1")

    resolved = resolve_model(
        "skyulf-registry",
        alias="champion",
        tracking_uri=config.tracking_uri,
        registry_uri=config.tracking_uri,
    )
    register_model(
        model_uri,
        "skyulf-registry",
        tracking_uri=config.tracking_uri,
        registry_uri=config.tracking_uri,
    )
    client.set_registered_model_alias("skyulf-registry", "champion", "2")

    assert resolved.version == "1"
    assert resolved.model_uri == "models:/skyulf-registry/1"


def test_register_does_not_promote_an_alias(tmp_path: Path) -> None:
    """Publishing a version leaves promotion as an explicit caller operation."""
    _, config = _publish(tmp_path)
    version = mlflow.MlflowClient(
        tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri
    ).get_model_version("skyulf-registry", "1")

    assert version.aliases == []


def test_registry_store_can_be_separate_from_tracking_store(tmp_path: Path) -> None:
    """Registry metadata and run artifacts can use separate explicit stores."""
    bundle, _ = _bundle()
    tracking = _config(tmp_path, "tracking")
    registry = _config(tmp_path, "registry")
    with track_run(tracking, run_name="separate-stores") as run:
        model_uri = log_model(
            bundle,
            run_id=run.run_id,
            artifact_path="model",
            tracking_uri=tracking.tracking_uri,
        )

    register_model(
        model_uri,
        "separate-store-model",
        tracking_uri=tracking.tracking_uri,
        registry_uri=registry.tracking_uri,
    )
    resolved = resolve_model(
        "separate-store-model",
        version="1",
        tracking_uri=tracking.tracking_uri,
        registry_uri=registry.tracking_uri,
    )

    assert resolved.model_uri == "models:/separate-store-model/1"
    assert resolved.digest


def test_register_rejects_non_run_artifact_uri() -> None:
    """Registry publication accepts only the run artifact URI produced by packaging."""
    with pytest.raises(ValueError, match="runs:/"):
        register_model("models:/already-registered/1", "skyulf-registry")


@pytest.mark.parametrize(
    ("alias", "version"),
    [(None, None), ("champion", "1")],
)
def test_resolve_requires_exactly_one_reference(alias: str | None, version: str | None) -> None:
    """A registry lookup must not silently choose latest or prefer one selector."""
    with pytest.raises(ValueError, match="exactly one of alias or version"):
        resolve_model("skyulf-registry", alias=alias, version=version)


def test_missing_model_and_permission_are_typed_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing resources and access failures remain distinct from dependency failures."""
    config = _config(tmp_path, "missing")
    with pytest.raises(RegistryModelNotFoundError):
        resolve_model(
            "missing-model",
            version="1",
            tracking_uri=config.tracking_uri,
            registry_uri=config.tracking_uri,
        )

    class ForbiddenClient:
        def get_model_version(self, name: str, version: str) -> object:
            error = mlflow.exceptions.MlflowException("denied")
            error.error_code = "PERMISSION_DENIED"
            raise error

    monkeypatch.setattr(
        "skyulf.integrations.mlflow.registry._make_client",
        lambda *args, **kwargs: ForbiddenClient(),
    )
    with pytest.raises(RegistryAccessError):
        resolve_model("private-model", version="1")


def test_missing_mlflow_is_a_dependency_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing optional package has a different remediation from registry access."""
    monkeypatch.setattr(
        "skyulf.integrations.mlflow.registry._require_mlflow",
        lambda: (_ for _ in ()).throw(RegistryDependencyError("install mlflow")),
    )
    with pytest.raises(RegistryDependencyError):
        resolve_model("skyulf-registry", version="1")


def test_unity_catalog_requires_three_part_model_name_without_network() -> None:
    """UC registry configuration rejects non-qualified names before any client call."""
    with pytest.raises(ValueError, match="catalog.schema.model"):
        resolve_model(
            "skyulf-registry",
            version="1",
            registry_uri="databricks-uc",
        )
