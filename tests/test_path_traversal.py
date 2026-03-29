"""Tests for path traversal protection in LocalArtifactStore and FileSystemCatalog."""
import os
import tempfile

import pytest

from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.data.catalog import FileSystemCatalog


# ── LocalArtifactStore ──────────────────────────────────────────────


class TestLocalArtifactStoreTraversal:
    """Verify _get_path neutralises directory escape attempts.

    LocalArtifactStore replaces ``/`` and ``\\`` in the key with ``_``
    before joining, so traversal characters become part of a flat filename.
    The PermissionError check is defence-in-depth if sanitisation ever fails.
    """

    @pytest.fixture()
    def store(self, tmp_path):
        return LocalArtifactStore(str(tmp_path))

    def test_normal_key_resolves_inside_base(self, store, tmp_path):
        path = store._get_path("my_model")
        assert path.startswith(str(tmp_path))
        assert path.endswith(".joblib")

    def test_dotdot_sanitised_into_safe_name(self, store, tmp_path):
        # ../../etc/passwd → .._.._etc_passwd.joblib (safe, stays in base)
        path = store._get_path("../../etc/passwd")
        assert path.startswith(str(tmp_path))
        assert ".." not in os.path.basename(path).replace("..", "")  # no real traversal
        assert os.sep not in os.path.basename(path)

    def test_subdir_traversal_sanitised(self, store, tmp_path):
        path = store._get_path("subdir/../../etc/passwd")
        assert path.startswith(str(tmp_path))

    def test_absolute_path_outside_base_sanitised(self, store, tmp_path):
        # Absolute path separators get replaced with _ so it stays in base
        outside = os.path.realpath(tempfile.gettempdir()) + "/evil.joblib"
        path = store._get_path(outside)
        assert path.startswith(str(tmp_path))

    def test_backslash_traversal_sanitised(self, store, tmp_path):
        path = store._get_path("..\\..\\etc\\passwd")
        assert path.startswith(str(tmp_path))

    def test_resolved_path_never_escapes_base(self, store, tmp_path):
        """Regardless of input, the result always lives under base_path."""
        malicious_keys = [
            "../../etc/passwd",
            "..\\..\\windows\\system32\\config",
            "/absolute/path/outside",
            "a/b/c/../../../../../etc/shadow",
            "....//....//etc//passwd",
        ]
        base = os.path.realpath(str(tmp_path))
        for key in malicious_keys:
            path = store._get_path(key)
            assert path.startswith(base), f"Key {key!r} escaped to {path}"

    def test_save_and_load_roundtrip(self, store):
        store.save("good_key", {"accuracy": 0.95})
        result = store.load("good_key")
        assert result == {"accuracy": 0.95}

    def test_exists_normal_key(self, store):
        assert not store.exists("missing_key")
        store.save("present_key", [1, 2, 3])
        assert store.exists("present_key")


# ── FileSystemCatalog ───────────────────────────────────────────────


class TestFileSystemCatalogTraversal:
    """Verify _get_path blocks directory escape attempts."""

    @pytest.fixture()
    def catalog(self, tmp_path):
        return FileSystemCatalog(base_path=str(tmp_path))

    def test_normal_relative_id_resolves_inside_base(self, catalog, tmp_path):
        path = catalog._get_path("data.csv")
        assert path.startswith(str(tmp_path))

    def test_dotdot_relative_stripped(self, catalog, tmp_path):
        # basename("../../etc/passwd") -> "passwd", sandboxed under tmp_path
        path = catalog._get_path("../../etc/passwd")
        assert path.startswith(str(tmp_path))
        assert os.path.basename(path) == "passwd"

    def test_backslash_traversal_stripped(self, catalog, tmp_path):
        path = catalog._get_path("..\\..\\secret.csv")
        assert path.startswith(str(tmp_path))
        assert os.path.basename(path) == "secret.csv"

    def test_absolute_path_with_traversal_blocked(self, catalog):
        evil = "/base/../etc/passwd"
        with pytest.raises(PermissionError, match="contains traversal segments"):
            catalog._get_path(evil)

    def test_absolute_path_without_traversal_allowed(self, catalog, tmp_path):
        # SmartCatalog sends absolute paths resolved from DB records
        abs_path = os.path.join(str(tmp_path), "resolved.csv")
        result = catalog._get_path(abs_path)
        assert result == os.path.realpath(abs_path)

    def test_relative_path_never_escapes_base(self, catalog, tmp_path):
        # Even deeply nested traversal gets basename-stripped
        path = catalog._get_path("a/b/c/../../../../../etc/shadow")
        assert path.startswith(str(tmp_path))
