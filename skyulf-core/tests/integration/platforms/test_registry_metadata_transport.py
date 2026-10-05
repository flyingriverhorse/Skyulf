"""Registry resolution fetches metadata without transferring fitted packages."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

mlflow = pytest.importorskip("mlflow")
from skyulf.integrations.mlflow.registration import registry


@pytest.mark.parametrize(
    ("registry_uri", "package_uri"),
    [
        ("databricks", "models:/catalog.schema.model/7"),
        ("databricks://profile", "models:/catalog.schema.model/7"),
        ("databricks-uc", "models:/catalog.schema.model/7"),
        ("databricks-uc://profile", "models:/catalog.schema.model/7"),
        ("https://registry.example", "s3://bucket/registered/model"),
        ("sqlite:///registry.db", "file:///registered/model/"),
    ],
)
@pytest.mark.parametrize("selector", [{"alias": "champion"}, {"version": "7"}])
def test_resolution_downloads_only_pinned_metadata(
    tmp_path: Path, monkeypatch, registry_uri, package_uri, selector
) -> None:
    """Large source packages must not be downloaded merely to identify a model."""
    metadata_path = tmp_path / "MLmodel"
    signature = mlflow.models.infer_signature([1.0], [2.0])
    mlflow.models.Model(
        signature=signature, metadata={"local_pipeline_digest": "fitted-digest"}
    ).save(str(metadata_path))
    client = Mock()
    client._registry_uri = registry_uri
    client.get_model_version.return_value = SimpleNamespace(version="7")
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
    client.get_model_version_download_uri.return_value = package_uri
    if registry_uri.startswith("databricks"):
        client.get_model_version_download_uri.side_effect = AssertionError(
            "Databricks requires credential-bearing models transport"
        )
    monkeypatch.setattr(registry, "make_registry_client", lambda *args: client)
    download = Mock(return_value=str(metadata_path))
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)

    resolved = registry.resolve_model(
        "catalog.schema.model",
        tracking_uri="https://tracking.example",
        registry_uri=registry_uri,
        **selector,
    )

    assert resolved.model_uri == "models:/catalog.schema.model/7"
    assert resolved.digest == "fitted-digest"
    assert resolved.signature == signature
    download.assert_called_once_with(
        artifact_uri=package_uri.rstrip("/") + "/MLmodel",
        tracking_uri="https://tracking.example",
        registry_uri=registry_uri,
    )
    if "alias" in selector:
        client.get_model_version_by_alias.assert_called_once_with(
            "catalog.schema.model", "champion"
        )
        client.get_model_version.assert_not_called()
    else:
        client.get_model_version.assert_called_once_with("catalog.schema.model", "7")
        client.get_model_version_by_alias.assert_not_called()


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        ("PERMISSION_DENIED", registry.RegistryAccessError),
        ("RESOURCE_DOES_NOT_EXIST", registry.RegistryModelNotFoundError),
        ("INTERNAL_ERROR", registry.RegistryOperationError),
    ],
)
def test_metadata_transport_preserves_typed_errors(monkeypatch, code, error_type) -> None:
    """Metadata-only transport keeps access, missing and operational failures distinct."""
    client = Mock()
    client._registry_uri = "databricks-uc"
    client.get_model_version.return_value = SimpleNamespace(version="7")
    monkeypatch.setattr(registry, "make_registry_client", lambda *args: client)
    error = mlflow.exceptions.MlflowException("metadata unavailable")
    error.error_code = code
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", Mock(side_effect=error))
    with pytest.raises(error_type) as captured:
        registry.resolve_model("catalog.schema.model", version="7")
    assert captured.value.__cause__ is error


def test_signed_metadata_uri_preserves_query_and_fragment(tmp_path: Path, monkeypatch) -> None:
    """Appending MLmodel must change the URI path without corrupting transport tokens."""
    metadata_path = tmp_path / "MLmodel"
    mlflow.models.Model(metadata={"bundle_digest": "signed-digest"}).save(str(metadata_path))
    client = Mock()
    client._registry_uri = "https://registry.example"
    client.get_model_version.return_value = SimpleNamespace(version="7")
    client.get_model_version_download_uri.return_value = (
        "https://artifacts.example/model/?token=a%2Fb&mode=read#revision"
    )
    monkeypatch.setattr(registry, "make_registry_client", lambda *args: client)
    download = Mock(return_value=str(metadata_path))
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)

    resolved = registry.resolve_model("model", version="7")

    assert resolved.digest == "signed-digest"
    download.assert_called_once_with(
        artifact_uri="https://artifacts.example/model/MLmodel?token=a%2Fb&mode=read#revision",
        tracking_uri=None,
        registry_uri="https://registry.example",
    )


@pytest.mark.parametrize("path_style", ["native", "posix", "file_uri"])
def test_local_metadata_transport_uses_real_filesystem(
    tmp_path: Path, monkeypatch, path_style
) -> None:
    """Native Windows paths and file URIs must still reach a real metadata file."""
    mlflow.models.Model(metadata={"bundle_digest": "local-digest"}).save(str(tmp_path / "MLmodel"))
    package_uri = {
        "native": str(tmp_path),
        "posix": tmp_path.as_posix(),
        "file_uri": tmp_path.as_uri(),
    }[path_style]
    client = Mock()
    client._registry_uri = "sqlite:///unused.db"
    client.get_model_version.return_value = SimpleNamespace(version="7")
    client.get_model_version_download_uri.return_value = package_uri
    monkeypatch.setattr(registry, "make_registry_client", lambda *args: client)

    resolved = registry.resolve_model("model", version="7")

    assert resolved.digest == "local-digest"
    assert resolved.model_uri == "models:/model/7"
