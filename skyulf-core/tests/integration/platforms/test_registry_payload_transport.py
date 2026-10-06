"""Direct loaders transfer only validated metadata and the declared Skyulf payload."""

import os
import shutil
import stat
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
REAL_DOWNLOAD = mlflow.artifacts.download_artifacts
from skyulf.data.dataset import SplitDataset
from skyulf.inference._manifest import ColumnSpec
from skyulf.inference._model_set_manifest import ComponentReference
from skyulf.inference.bundle import build_bundle, predict_local, save_bundle
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.inference.model_set import save_model_set
from skyulf.inference.model_set_scoring import predict_model_set
from skyulf.integrations.mlflow.models import model_set
from skyulf.integrations.mlflow.registration import registry
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture(params=["bundle", "local_pipeline", "model_set"])
def package(request, tmp_path, monkeypatch):
    """Supply real fitted payloads through a recording fake of remote file transport."""
    kind = request.param
    monkeypatch.setattr(registry.tempfile, "tempdir", str(tmp_path))
    frame = pd.DataFrame({"x": np.arange(6, dtype=float), "y": np.arange(6) * 2.0})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="y")
    source = tmp_path / "remote"
    payload = source / "nested" / "payload"
    payload.parent.mkdir(parents=True)
    if kind == "bundle":
        bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
        save_bundle(bundle, payload)
        digest = bundle.semantic_digest
    else:
        local_path = payload if kind == "local_pipeline" else tmp_path / "component"
        save_local_pipeline(pipeline, local_path)
        local = load_local_pipeline(local_path)
        digest = local.manifest.pipeline_sha256
        if kind == "model_set":
            artifact = save_model_set(
                payload,
                {
                    "main": (
                        ComponentReference(name="component", version="1", digest=digest),
                        local_path,
                    )
                },
                record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
            )
            digest = artifact.manifest.set_sha256
    model = mlflow.models.Model(
        metadata={
            "skyulf_artifact_kind": kind,
            "skyulf_execution_scope": "whole_frame_local",
            "skyulf_fitted_engine": "pandas",
            f"{kind}_digest": digest,
        },
        flavors={"python_function": {"artifacts": {kind: {"path": "nested/payload"}}}},
    )
    model.save(str(source / "MLmodel"))
    (source / "code").mkdir()
    (source / "code" / "unneeded_runtime.py").write_text("# Must not be transferred")
    client = Mock()
    client._registry_uri = "databricks-uc://selected"
    monkeypatch.setattr(registry, "make_registry_client", lambda *args: client)
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **kwargs: client)
    calls = []

    def download(*, artifact_uri, tracking_uri, registry_uri, dst_path=None):
        """Copy just the requested remote subtree, like MLflow's artifact transport."""
        assert tracking_uri == "databricks://selected"
        assert registry_uri == "databricks-uc://selected"
        relative = artifact_uri.removeprefix("models:/catalog.schema.model/7").lstrip("/")
        calls.append(relative)
        selected = source / relative
        if not selected.exists():
            error = mlflow.exceptions.MlflowException("missing")
            error.error_code = "RESOURCE_DOES_NOT_EXIST"
            raise error
        if dst_path is None:
            return str(selected)
        destination = Path(dst_path) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if selected.is_dir():
            shutil.copytree(selected, destination)
        else:
            shutil.copy2(selected, destination)
        return str(destination)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    loaders = {
        "bundle": registry.load_registered_bundle,
        "local_pipeline": registry.load_registered_local_pipeline,
        "model_set": model_set.load_registered_model_set,
    }
    resolved = registry.ResolvedModel(
        "catalog.schema.model", "7", "models:/catalog.schema.model/7", None, digest
    )

    def load():
        """Keep the real direct loader and all semantic validation in the exercise."""
        return loaders[kind](
            resolved, tracking_uri="databricks://selected", registry_uri="databricks-uc://selected"
        )

    return kind, source, model, calls, load


def test_direct_loader_downloads_only_declared_payload(package):
    """Unrelated runtime source must not cross the direct-loader network boundary."""
    kind, source, model, calls, load = package
    artifact = load()
    query = pd.DataFrame({"id": [1, 2], "x": [2.0, 4.0]})
    if kind == "bundle":
        output = predict_local(query[["x"]], artifact)["prediction"]
    elif kind == "local_pipeline":
        output = predict_local_pipeline(query[["x"]], artifact)["prediction"]
    else:
        assert artifact.directory.is_dir()
        assert not (artifact.directory.parents[1] / "code").exists()
        output = predict_model_set(query, artifact)["main__prediction"]
    np.testing.assert_allclose(output, [4.0, 8.0])
    assert calls == ["MLmodel", "nested/payload"]


def test_direct_loader_uses_real_mlflow_models_download_layout(package, monkeypatch):
    """MLflow models transport retains the full artifact prefix inside the destination."""
    from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
    from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
    from mlflow.tracking import artifact_utils

    kind, source, model, calls, load = package
    repository = object.__new__(ModelsArtifactRepository)
    repository.repo = LocalArtifactRepository(source.as_uri())
    original = artifact_utils.get_artifact_repository

    def transport(artifact_uri, **kwargs):
        """Replace only remote repository construction, leaving MLflow layout logic real."""
        if artifact_uri == "models:/catalog.schema.model/7":
            return repository
        return original(artifact_uri, **kwargs)

    monkeypatch.setattr(artifact_utils, "get_artifact_repository", transport)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", REAL_DOWNLOAD)

    artifact = load()

    if kind == "model_set":
        assert artifact.directory.is_dir()
    else:
        assert artifact is not None
    downloaded = list(source.parent.glob("skyulf-registry-payload-*/nested/payload"))
    assert len(downloaded) == 1
    assert not (downloaded[0].parents[1] / "code").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "../outside",
        "/outside",
        "C:\\outside",
        ".",
        "nested/../../outside",
        "nested/%2e%2e/payload",
        "nested/payload?query",
        "nested/payload#fragment",
    ],
)
def test_malformed_path_rejected_before_payload_transport(package, relative):
    """Untrusted artifact maps cannot make the transport read outside the package."""
    kind, source, model, calls, load = package
    model.flavors["python_function"]["artifacts"][kind]["path"] = relative
    model.save(str(source / "MLmodel"))
    with pytest.raises(ValueError, match="relative|contained"):
        load()
    assert calls == ["MLmodel"]
    assert not list(source.parent.glob("skyulf-registry-payload-*"))


@pytest.mark.parametrize("damage", ["digest", "missing_directory", "missing_manifest"])
def test_invalid_payload_cleans_failed_download(package, damage):
    """Failed loads must retain validation errors and remove their owned temp roots."""
    kind, source, model, calls, load = package
    if damage == "digest":
        model.metadata[f"{kind}_digest"] = "0" * 64
        model.save(str(source / "MLmodel"))
    elif damage == "missing_directory":
        model.flavors["python_function"]["artifacts"][kind]["path"] = "absent"
        model.save(str(source / "MLmodel"))
    else:
        (source / "nested" / "payload" / "manifest.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load()
    assert not list(source.parent.glob("skyulf-registry-payload-*"))


def test_full_package_transport_keeps_runtime_files(package):
    """The complete-package helper used by Spark must continue retrieving runtime code."""
    kind, source, model, calls, load = package
    client = Mock()
    client._registry_uri = "databricks-uc://selected"
    path = registry.download_registered_package(
        mlflow, client, "catalog.schema.model", "7", "databricks://selected"
    )
    assert (Path(path) / "code" / "unneeded_runtime.py").is_file()
    assert calls == [""]


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        ("PERMISSION_DENIED", registry.RegistryAccessError),
        ("RESOURCE_DOES_NOT_EXIST", registry.RegistryModelNotFoundError),
        ("INTERNAL_ERROR", registry.RegistryOperationError),
    ],
)
def test_oss_download_uri_errors_remain_typed(package, code, error_type):
    """OSS download URI lookup retains typed registry errors before file transport."""
    kind, source, model, calls, load = package
    client = registry.make_registry_client(None, None, None)
    client._registry_uri = "https://registry.example"
    error = mlflow.exceptions.MlflowException("registry URI lookup failed")
    error.error_code = code
    client.get_model_version_download_uri.side_effect = error
    with pytest.raises(error_type):
        load()
    assert calls == []
    assert not list(source.parent.glob("skyulf-registry-payload-*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows rejects removal of read-only files")
def test_readonly_cleanup_does_not_mask_validation_failure(package, monkeypatch):
    """A failed Windows temp cleanup must not replace the actionable digest error."""
    kind, source, model, calls, load = package
    model.metadata[f"{kind}_digest"] = "0" * 64
    model.save(str(source / "MLmodel"))
    download = mlflow.artifacts.download_artifacts

    def readonly_metadata(**kwargs):
        """Retain a real Windows read-only file in the downloaded temp package."""
        result = download(**kwargs)
        if Path(result).is_file():
            Path(result).chmod(stat.S_IREAD)
        return result

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", readonly_metadata)
    try:
        with pytest.raises(ValueError, match="digest"):
            load()
    finally:
        for leftover in source.parent.glob("skyulf-registry-payload-*/MLmodel"):
            leftover.chmod(stat.S_IWRITE)
    assert calls == ["MLmodel"]
