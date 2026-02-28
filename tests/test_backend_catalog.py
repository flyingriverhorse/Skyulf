
import os
import shutil
import tempfile
import pandas as pd
import pytest
import sys

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
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc() 
