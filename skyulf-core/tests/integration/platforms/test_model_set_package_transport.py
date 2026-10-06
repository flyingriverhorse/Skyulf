"""Model-set packaging copies exact component bytes without unused runtime code."""

import pytest

pytest.importorskip("mlflow")
from test_registry_payload_transport import package as package

from skyulf.integrations.databricks.model_sets import model_set_project as project
from skyulf.integrations.mlflow.registration import registry


@pytest.mark.parametrize("package", ["local_pipeline"], indirect=True)
def test_packaging_downloads_only_pinned_component_payload(package, monkeypatch):
    """Registering a set must retain model bytes without downloading every runtime module."""
    kind, source, model, calls, load = package
    resolved = registry.ResolvedModel(
        "catalog.schema.model",
        "7",
        "models:/catalog.schema.model/7",
        None,
        model.metadata["local_pipeline_digest"],
    )
    client = registry.make_registry_client(None, None, None)
    client.get_model_version.return_value.source = resolved.model_uri
    monkeypatch.setattr(project, "make_registry_client", lambda *args: client)

    directory = project._component_directory(
        resolved, "databricks://selected", "databricks-uc://selected"
    )

    expected = source / "nested" / "payload"
    assert directory.is_dir()
    for path in expected.rglob("*"):
        if path.is_file():
            assert (directory / path.relative_to(expected)).read_bytes() == path.read_bytes()
    assert calls == ["MLmodel", "nested/payload"]
