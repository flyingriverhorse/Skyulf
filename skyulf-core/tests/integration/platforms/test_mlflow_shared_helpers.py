"""Shared adapter contracts preserve dtype aliases and whole-package worker snapshots."""

from pathlib import Path

import pytest

import skyulf
from skyulf.integrations.mlflow.shared import _model_metadata
from skyulf.integrations.mlflow.spark import _spark_environment, spark_model


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("OBJECT", "string"),
        ("str", "string"),
        ("Utf8", "string"),
        ("Boolean", "bool"),
        ("Int64", "int64"),
        ("Float32", "float32"),
        ("Category", "category"),
    ],
)
def test_shared_dtype_preserves_adapter_aliases(dtype, expected):
    """Signature and worker validation must normalize the same dtype vocabulary."""
    assert _model_metadata.normalized_dtype(dtype) == expected
    assert spark_model._normalized_dtype(dtype) == expected


def test_local_dtype_wrapper_keeps_existing_entry_point():
    """Existing callers of the local adapter must retain the shared alias behavior."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.models.pipeline_model import normalized_dtype

    assert normalized_dtype("BOOLEAN") == _model_metadata.normalized_dtype("BOOLEAN") == "bool"


def test_runtime_digest_covers_the_whole_package():
    """Relocating the adapter cannot narrow its certificate to integration sources."""
    root = Path(skyulf.__file__).resolve().parent
    assert spark_model.runtime_source_digest() == _spark_environment.source_digest(root)


def test_snapshot_keeps_package_root_after_adapter_relocation(tmp_path, monkeypatch):
    """Workers need inference and pipeline sources as well as relocated adapter modules."""
    root = Path(skyulf.__file__).resolve().parent
    monkeypatch.setattr(_spark_environment, "__file__", str(tmp_path / "moved" / "helper.py"))
    snapshot = _spark_environment._snapshot_source(tmp_path)
    expected = {
        path.relative_to(root) for path in root.rglob("*.py") if "__pycache__" not in path.parts
    }
    copied = {path.relative_to(snapshot) for path in snapshot.rglob("*.py")}
    assert copied == expected
    assert (snapshot / "inference" / "fitted_pipeline.py").is_file()
    assert (snapshot / "pipeline" / "__init__.py").is_file()


@pytest.mark.parametrize("supports_uv_project", [False, True])
@pytest.mark.parametrize("certified", [False, True])
def test_pyfunc_environment_only_packages_certified_workers(
    tmp_path, monkeypatch, supports_uv_project, certified
):
    """Local models keep pins unchanged; worker snapshots and uv options remain opt-in."""
    mlflow = pytest.importorskip("mlflow")

    def save_without_uv(path):
        """Expose an older save signature without executing model publication."""
        raise AssertionError("Environment preparation must not save a model.")

    def save_with_uv(path, uv_project_path=None):
        """Expose the newer optional project argument without saving anything."""
        raise AssertionError("Environment preparation must not save a model.")

    monkeypatch.setattr(
        mlflow.pyfunc, "save_model", save_with_uv if supports_uv_project else save_without_uv
    )
    snapshots = []
    pins = ["skyulf-core==0.9.2", "mlflow==3.10.0"]

    def snapshot(directory, requirements):
        """Observe the existing wheel boundary without creating a second wheel fixture."""
        snapshots.append((directory, list(requirements)))
        return ["snapshot", "wheel.whl"], ["code/wheel.whl", requirements[1]], "source-hash"

    monkeypatch.setattr(_spark_environment, "snapshot_worker_environment", snapshot)
    options, digest = _spark_environment.pyfunc_environment(
        tmp_path, pins, spark_certified=certified
    )
    expected = {"pip_requirements": ["skyulf-core==0.9.2", "mlflow==3.10.0"]}
    if supports_uv_project:
        expected["uv_project_path"] = str(tmp_path)
    if certified:
        expected.update(
            code_paths=["snapshot", "wheel.whl"],
            pip_requirements=["code/wheel.whl", "mlflow==3.10.0"],
        )
    assert options == expected
    assert digest == ("source-hash" if certified else None)
    assert snapshots == ([(tmp_path, pins)] if certified else [])
    assert pins == ["skyulf-core==0.9.2", "mlflow==3.10.0"]
