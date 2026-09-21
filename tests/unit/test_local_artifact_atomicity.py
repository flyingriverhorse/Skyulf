"""Failed publication must never replace a readable model with a partial file."""

from pathlib import Path

import numpy as np
import pytest

from backend.ml_pipeline.artifacts import local


@pytest.mark.parametrize("existing", [False, True])
def test_failed_dump_preserves_previous_artifact(tmp_path, monkeypatch, existing):
    """An interrupted serializer must leave the previous artifact or no final key."""
    store = local.LocalArtifactStore(str(tmp_path))
    if existing:
        store.save("model", {"version": "old"})

    def partial_dump(data, destination):
        """Simulate a serializer that fails after writing part of its output."""
        if hasattr(destination, "write"):
            destination.write(b"partial pickle")
        else:
            Path(destination).write_bytes(b"partial pickle")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(local.joblib, "dump", partial_dump)
    with pytest.raises(OSError, match="simulated disk failure"):
        store.save("model", {"version": "new"})
    if existing:
        assert store.load("model") == {"version": "old"}
    else:
        assert not store.exists("model")
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ["model.joblib"] if existing else []
    )


def test_successful_replace_preserves_numpy_payload(tmp_path):
    """A successful overwrite must publish a complete artifact and remove staging files."""
    store = local.LocalArtifactStore(str(tmp_path))
    store.save("model", {"version": "old"})
    values = np.arange(5000).reshape(100, 50)
    store.save("model", {"values": values})
    np.testing.assert_array_equal(store.load("model")["values"], values)
    assert store.list_artifacts() == ["model"]
    assert [path.name for path in tmp_path.iterdir()] == ["model.joblib"]


def test_failed_replace_preserves_previous_artifact(tmp_path, monkeypatch):
    """A Windows-style publication failure must preserve the old file and clean staging."""
    store = local.LocalArtifactStore(str(tmp_path))
    store.save("model", {"version": "old"})

    def fail_replace(source, destination):
        """Verify staging is complete and closed before simulating a locked destination."""
        assert Path(source).parent == Path(destination).parent
        with Path(source).open("rb+") as stream:
            assert local.joblib.load(stream) == {"version": "new"}
        assert store.load("model") == {"version": "old"}
        raise PermissionError("destination locked")

    monkeypatch.setattr(local.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="destination locked"):
        store.save("model", {"version": "new"})
    assert store.load("model") == {"version": "old"}
    assert [path.name for path in tmp_path.iterdir()] == ["model.joblib"]


def test_staging_file_is_not_listed_during_save(tmp_path, monkeypatch):
    """Readers must see published artifacts only while a replacement is serialized."""
    store = local.LocalArtifactStore(str(tmp_path))
    store.save("model", {"version": "old"})
    dump = local.joblib.dump
    observed = []

    def inspect_dump(data, destination):
        """Observe the public listing while the staging file exists."""
        observed.extend(store.list_artifacts())
        return dump(data, destination)

    monkeypatch.setattr(local.joblib, "dump", inspect_dump)
    store.save("model", {"version": "new"})
    assert observed == ["model"]


def test_leftover_staging_files_are_not_listed(tmp_path):
    """Crash leftovers must stay hidden without hiding unrelated legacy artifacts."""
    store = local.LocalArtifactStore(str(tmp_path))
    (tmp_path / ".skyulf-artifact-abandoned.tmp").write_bytes(b"partial pickle")
    (tmp_path / "legacy.tmp").write_bytes(b"legacy data")
    store.save(".skyulf-artifact-model", {"version": "old"})
    assert sorted(store.list_artifacts()) == [".skyulf-artifact-model", "legacy.tmp"]
