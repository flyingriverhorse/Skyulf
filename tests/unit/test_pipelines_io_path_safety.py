"""Unit tests for `_pipeline_json_path`'s dataset_id sanitization.

Regression coverage for a CodeQL "Uncontrolled data used in path
expression" finding: `dataset_id` is a raw URL path segment used to build
an on-disk file path for the "json" pipeline-storage backend, so it is
guarded twice — a character allowlist before any `Path` is constructed,
then a resolve+contain check on the joined path. Each layer is tested
independently.
"""

import re

import pytest

from backend.ml_pipeline._internal._routers import pipelines_io
from backend.ml_pipeline._internal._routers.pipelines_io import _pipeline_json_path


def test_valid_dataset_id_resolves_inside_storage_dir(tmp_path) -> None:
    """A normal alphanumeric/dash/underscore id resolves to the expected path."""
    root = tmp_path.resolve()
    result = _pipeline_json_path(tmp_path, "abc-123_XYZ")
    assert result == root / "abc-123_XYZ.json"
    assert result.parent == root


@pytest.mark.parametrize(
    "dataset_id",
    [
        "..",
        "../secret",
        "../../etc/passwd",
        "/etc/passwd",
        "a/b",
        "a\\b",
        "a/../../b",
        "",
        "id with spaces",
        "id;rm -rf /",
        "id\x00null",
        "id\n",
        ".",
        "./x",
    ],
)
def test_malicious_or_malformed_dataset_id_rejected(tmp_path, dataset_id: str) -> None:
    """Anything containing a path separator, '..', or non-allowlisted
    character must raise ValueError before a Path is ever constructed."""
    with pytest.raises(ValueError):
        _pipeline_json_path(tmp_path, dataset_id)


def test_containment_check_rejects_escape_if_allowlist_is_loosened(tmp_path, monkeypatch) -> None:
    """Layer 2 stands alone: if `_SAFE_DATASET_ID_RE` were ever widened, the
    resolve+contain check must still refuse a path that escapes storage_dir."""
    monkeypatch.setattr(pipelines_io, "_SAFE_DATASET_ID_RE", re.compile(r".*", re.DOTALL))
    # Positive control — a contained id still resolves, so the rejection below
    # comes from the containment check and not from the helper raising always.
    assert pipelines_io._pipeline_json_path(tmp_path, "ok").name == "ok.json"
    with pytest.raises(ValueError):
        pipelines_io._pipeline_json_path(tmp_path, "../escape")
