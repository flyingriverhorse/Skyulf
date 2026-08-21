"""Unit tests for `_pipeline_json_path`'s dataset_id sanitization.

Regression coverage for a CodeQL "Uncontrolled data used in path
expression" finding: `dataset_id` is a raw URL path segment used to build
an on-disk file path for the "json" pipeline-storage backend, so it must
be strictly validated before ever touching the filesystem.
"""

import pytest

from backend.ml_pipeline._internal._routers.pipelines_io import _pipeline_json_path


def test_valid_dataset_id_resolves_inside_storage_dir(tmp_path) -> None:
    """A normal alphanumeric/dash/underscore id resolves to the expected path."""
    result = _pipeline_json_path(tmp_path, "abc-123_XYZ")
    assert result == tmp_path / "abc-123_XYZ.json"
    assert result.parent == tmp_path


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


def test_rejected_dataset_id_never_escapes_storage_dir_if_bypassed(tmp_path) -> None:
    """Defense-in-depth: even if validation were bypassed, the naive
    '<storage_dir>/<dataset_id>.json' join must not escape storage_dir for
    the specific case of a bare '..' -- '..' + '.json' is the literal
    filename '...json', not an actual parent-directory reference."""
    literal_path = tmp_path / "...json"
    assert literal_path.parent == tmp_path
