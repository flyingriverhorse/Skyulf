import os
import shutil
import sys
import tempfile

import pandas as pd
import pytest

# Add root to path
sys.path.append(os.getcwd())

from backend.data.catalog import FileSystemCatalog


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


class TestFileSystemCatalog:
    @pytest.fixture(autouse=True)
    def _pandas_engine(self, monkeypatch):
        # These round-trip tests pin the legacy pandas path; the polars path
        # is covered by tests/integration/test_catalog_polars_ingestion.py.
        from backend.config import get_settings

        monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas", raising=False)

    @pytest.fixture
    def catalog(self):
        tmp_dir = tempfile.mkdtemp()
        cat = FileSystemCatalog(base_path=tmp_dir)
        yield cat
        shutil.rmtree(tmp_dir)

    @pytest.mark.skipif(
        not _has_pyarrow(),
        reason="pyarrow not installed",
    )
    def test_save_and_load_pandas(self, catalog):
        """Test that we can act like normal with Pandas despite Any types."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        # Save as parquet default
        catalog.save("test_data", df)

        # Load
        loaded_df = catalog.load("test_data")

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_and_load_csv(self, catalog):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        catalog.save("test_data.csv", df)

        loaded_df = catalog.load("test_data.csv")
        pd.testing.assert_frame_equal(df, loaded_df)


if __name__ == "__main__":
    # Manually run if executed as script
    try:
        tmp_dir = tempfile.mkdtemp()
        cat = FileSystemCatalog(base_path=tmp_dir)

        print("Testing Save/Load Parquet...")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        cat.save("test_data", df)
        loaded = cat.load("test_data")
        print("Success Parquet!")

        shutil.rmtree(tmp_dir)
    except Exception as e:  # noqa: BLE001 - report failure with traceback
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
