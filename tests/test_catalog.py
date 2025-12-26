import os
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from backend.data.catalog import FileSystemCatalog, SmartCatalog, S3Catalog

# Mock data
MOCK_DF = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

class TestFileSystemCatalog:
    def test_load_save_csv(self, tmp_path):
        catalog = FileSystemCatalog(base_path=str(tmp_path))
        dataset_id = "test_data.csv"
        
        # Save
        catalog.save(dataset_id, MOCK_DF)
        assert (tmp_path / "test_data.csv").exists()
        
        # Load
        loaded_df = catalog.load(dataset_id)
        pd.testing.assert_frame_equal(MOCK_DF, loaded_df)

    def test_load_save_parquet(self, tmp_path):
        catalog = FileSystemCatalog(base_path=str(tmp_path))
        dataset_id = "test_data.parquet"
        
        # Save
        catalog.save(dataset_id, MOCK_DF)
        assert (tmp_path / "test_data.parquet").exists()
        
        # Load
        loaded_df = catalog.load(dataset_id)
        pd.testing.assert_frame_equal(MOCK_DF, loaded_df)

    def test_security_check(self, tmp_path):
        catalog = FileSystemCatalog(base_path=str(tmp_path))
        # Attempt directory traversal
        dataset_id = "../secret.txt"
        
        # Should resolve to base_path/secret.txt (basename check), NOT outside
        path = catalog._get_path(dataset_id)
        assert path == str(tmp_path / "secret.txt")


class TestSmartCatalog:
    def test_resolve_numeric_id(self):
        # Mock Session and DataSource
        session = MagicMock()
        mock_ds = MagicMock()
        mock_ds.to_dict.return_value = {
            "config": {"file_path": "uploads/data/resolved.csv"}
        }
        session.query.return_value.filter.return_value.first.return_value = mock_ds
        
        # Mock Catalogs
        fs_catalog = MagicMock()
        fs_catalog.load.return_value = MOCK_DF
        
        # Create SmartCatalog
        catalog = SmartCatalog(session=session, fs_catalog=fs_catalog)
        
        # Patch extract_file_path_from_source where it is defined
        with patch("backend.utils.file_utils.extract_file_path_from_source") as mock_extract:
            mock_extract.return_value = "uploads/data/resolved.csv"
            
            # Load with numeric ID
            catalog.load("28")
            
            # Verify fs_catalog called with RESOLVED path
            fs_catalog.load.assert_called_with("uploads/data/resolved.csv")

    def test_resolve_numeric_id_with_credentials(self):
        # Mock Session and DataSource with credentials
        session = MagicMock()
        mock_ds = MagicMock()
        mock_ds.to_dict.return_value = {
            "config": {
                "file_path": "s3://bucket/data.csv",
                "storage_options": {"key": "AKIA", "secret": "SECRET"}
            }
        }
        # Also need to mock config access on the object itself for _resolve_id
        mock_ds.config = {
            "file_path": "s3://bucket/data.csv",
            "storage_options": {"key": "AKIA", "secret": "SECRET"}
        }
        
        session.query.return_value.filter.return_value.first.return_value = mock_ds
        
        # Mock Catalogs
        s3_catalog = MagicMock()
        s3_catalog.load.return_value = MOCK_DF
        
        # Create SmartCatalog
        catalog = SmartCatalog(session=session, s3_catalog=s3_catalog)
        
        with patch("backend.utils.file_utils.extract_file_path_from_source") as mock_extract:
            mock_extract.return_value = "s3://bucket/data.csv"
            
            # Load with numeric ID
            catalog.load("99")
            
            # Verify s3_catalog called with RESOLVED path AND credentials
            s3_catalog.load.assert_called_with(
                "s3://bucket/data.csv", 
                storage_options={"key": "AKIA", "secret": "SECRET"}
            )

    def test_pass_through_string_id(self):
        session = MagicMock()
        fs_catalog = MagicMock()
        catalog = SmartCatalog(session=session, fs_catalog=fs_catalog)
        
        catalog.load("some_file.csv")
        fs_catalog.load.assert_called_with("some_file.csv")

    def test_dispatch_to_s3(self):
        session = MagicMock()
        fs_catalog = MagicMock()
        s3_catalog = MagicMock()
        s3_catalog.load.return_value = MOCK_DF
        
        catalog = SmartCatalog(session=session, fs_catalog=fs_catalog, s3_catalog=s3_catalog)
        
        # Load S3 path
        catalog.load("s3://my-bucket/data.csv")
        
        # Verify s3_catalog called
        s3_catalog.load.assert_called_with("s3://my-bucket/data.csv")
        # Verify fs_catalog NOT called
        fs_catalog.load.assert_not_called()


class TestS3Catalog:
    def test_s3_paths(self):
        # We mock s3fs import check
        with patch.dict("sys.modules", {"s3fs": MagicMock()}):
            catalog = S3Catalog(bucket_name="my-bucket")
            
            assert catalog._get_s3_path("data.csv") == "s3://my-bucket/data.csv"
            assert catalog._get_s3_path("s3://other-bucket/data.csv") == "s3://other-bucket/data.csv"

    @patch("pandas.read_csv")
    def test_load_s3(self, mock_read_csv):
        with patch.dict("sys.modules", {"s3fs": MagicMock()}):
            catalog = S3Catalog(bucket_name="my-bucket")
            catalog.load("data.csv")
            
            mock_read_csv.assert_called_with(
                "s3://my-bucket/data.csv", 
                nrows=None, 
                storage_options={}
            )
