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
    from skyulf.integrations.mlflow.models.local_model import normalized_dtype

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
    assert (snapshot / "inference" / "local_pipeline.py").is_file()
    assert (snapshot / "pipeline" / "__init__.py").is_file()
