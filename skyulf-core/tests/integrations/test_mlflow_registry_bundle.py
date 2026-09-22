"""Local SQLite coverage for loading pinned registered inference bundles."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import build_bundle, predict_local

mlflow = pytest.importorskip("mlflow")
from skyulf.integrations.mlflow import registry
from skyulf.integrations.mlflow.model import log_model
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Package two different fitted models in separate tracking and registry stores."""
    root = tmp_path_factory.mktemp("registered-bundle")
    tracking_uri = f"sqlite:///{(root / 'tracking.db').as_posix()}"
    registry_uri = f"sqlite:///{(root / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    experiment = client.create_experiment(
        "bundle-load", artifact_location=(root / "artifacts").as_uri()
    )
    for slope in (2.0, 3.0):
        frame = pd.DataFrame({"x": np.arange(8, dtype="float64"), "target": np.arange(8) * slope})
        pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
        pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
        bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
        run = client.create_run(experiment)
        uri = log_model(
            bundle, run_id=run.info.run_id, artifact_path="model", tracking_uri=tracking_uri
        )
        registry.register_model(uri, "pinned", tracking_uri=tracking_uri, registry_uri=registry_uri)
        client.set_terminated(run.info.run_id)
    resolved = registry.resolve_model(
        "pinned", version="1", tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    source = client.get_model_version("pinned", "1").source
    package = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=source, tracking_uri=tracking_uri, registry_uri=registry_uri
        )
    )
    return {
        "client": client,
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
        "resolved": resolved,
        "package": package,
    }


def _load(published: dict[str, Any], resolved: registry.ResolvedModel | None = None) -> Any:
    """Load through the public boundary using the fixture's explicit stores."""
    return registry.load_registered_bundle(
        resolved or published["resolved"],
        tracking_uri=published["tracking_uri"],
        registry_uri=published["registry_uri"],
    )


def test_load_pinned_bundle_after_alias_moves(published: dict[str, Any]) -> None:
    """Moving an alias must not replace the artifact already selected for a job."""
    client = published["client"]
    client.set_registered_model_alias("pinned", "champion", "1")
    resolved = registry.resolve_model(
        "pinned",
        alias="champion",
        tracking_uri=published["tracking_uri"],
        registry_uri=published["registry_uri"],
    )
    client.set_registered_model_alias("pinned", "champion", "2")
    loaded = _load(published, resolved)
    output = predict_local(pd.DataFrame({"x": [1.0, 4.0]}), loaded)
    assert loaded.semantic_digest == resolved.digest
    np.testing.assert_allclose(output["prediction"], [2.0, 8.0])


def test_explicit_stores_preserve_global_uris_and_active_run(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """Loading must not use or mutate an unrelated caller's fluent MLflow state."""
    old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    caller_uri = f"sqlite:///{(tmp_path / 'caller.db').as_posix()}"
    try:
        mlflow.set_tracking_uri(caller_uri)
        mlflow.set_registry_uri(caller_uri)
        with mlflow.start_run() as active:
            loaded = _load(published)
            assert mlflow.active_run().info.run_id == active.info.run_id
            assert mlflow.get_tracking_uri() == caller_uri
            assert mlflow.get_registry_uri() == caller_uri
            assert loaded.semantic_digest == published["resolved"].digest
    finally:
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)


@pytest.mark.parametrize(
    "changes",
    [
        {"digest": None},
        {"digest": ""},
        {"version": "champion"},
        {"model_uri": "models:/pinned@champion"},
    ],
)
def test_reject_incomplete_or_unpinned_reference(
    published: dict[str, Any], changes: dict[str, Any]
) -> None:
    """A caller cannot bypass concrete version and digest identity requirements."""
    with pytest.raises(ValueError, match="digest|concrete|version"):
        _load(published, replace(published["resolved"], **changes))


def test_reject_resolved_digest_mismatch(published: dict[str, Any]) -> None:
    """A changed package cannot silently replace the digest resolved earlier."""
    with pytest.raises(ValueError, match="digest"):
        _load(published, replace(published["resolved"], digest="0" * 64))


def _register_copy(published: dict[str, Any], package: Path) -> registry.ResolvedModel:
    """Register a test-owned package copy without changing the shared model artifacts."""
    name = package.parent.name
    client = published["client"]
    client.create_registered_model(name)
    client.create_model_version(name, package.as_uri())
    return replace(published["resolved"], name=name, model_uri=f"models:/{name}/1")


def test_uses_packaged_artifact_path(published: dict[str, Any], tmp_path: Path) -> None:
    """The loader follows MLmodel's actual artifact map rather than a guessed directory."""
    package = Path(shutil.copytree(published["package"], tmp_path / "model"))
    model = mlflow.models.Model.load(str(package))
    artifact = model.flavors["python_function"]["artifacts"]["bundle"]
    (package / artifact["path"]).rename(package / "relocated")
    artifact["path"] = "relocated"
    model.save(str(package / "MLmodel"))
    loaded = _load(published, _register_copy(published, package))
    assert loaded.semantic_digest == published["resolved"].digest


@pytest.mark.parametrize(
    "damage",
    [
        "metadata-digest",
        "bundle-digest",
        "missing-artifact",
        "missing-directory",
        "traversal",
        "absolute",
        "windows-drive",
    ],
)
def test_reject_invalid_packaged_bundle(
    published: dict[str, Any], tmp_path: Path, damage: str
) -> None:
    """Missing identity and unsafe artifact paths fail before serving a registered model."""
    package = Path(shutil.copytree(published["package"], tmp_path / "model"))
    model = mlflow.models.Model.load(str(package))
    artifact = model.flavors["python_function"]["artifacts"]["bundle"]
    resolved = _register_copy(published, package)
    if damage == "metadata-digest":
        model.metadata.pop("skyulf_bundle_digest")
    elif damage == "bundle-digest":
        model.metadata["skyulf_bundle_digest"] = "0" * 64
        resolved = replace(resolved, digest="0" * 64)
    elif damage == "missing-artifact":
        model.flavors["python_function"]["artifacts"].pop("bundle")
    elif damage == "missing-directory":
        artifact["path"] = "missing"
    else:
        artifact["path"] = {
            "traversal": "../outside",
            "absolute": str(tmp_path.resolve()),
            "windows-drive": "C:\\outside",
        }[damage]
    model.save(str(package / "MLmodel"))
    with pytest.raises(ValueError, match="digest|artifact|relative|contained"):
        _load(published, resolved)
