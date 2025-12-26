import os
import pytest
from unittest.mock import MagicMock, patch
from backend.data.catalog import create_catalog_from_options, S3Catalog, FileSystemCatalog

# Mock Node structure
class MockNode:
    def __init__(self, params):
        self.params = params

def test_create_catalog_detects_s3_from_nodes_dict():
    """
    Test that create_catalog_from_options detects S3 path in node params (dict format)
    even if storage_options is None.
    """
    nodes = [
        {"params": {"dataset_id": "local.csv"}},
        {"params": {"path": "s3://my-bucket/data.csv"}}
    ]
    
    # Mock s3fs to avoid import error
    with patch.dict("sys.modules", {"s3fs": MagicMock()}):
        catalog = create_catalog_from_options(storage_options=None, nodes=nodes)
        assert isinstance(catalog, S3Catalog)
        assert catalog.bucket_name == "my-bucket"

def test_create_catalog_detects_s3_from_nodes_object():
    """
    Test that create_catalog_from_options detects S3 path in node params (object format)
    even if storage_options is None.
    """
    nodes = [
        MockNode(params={"dataset_id": "local.csv"}),
        MockNode(params={"path": "s3://other-bucket/data.parquet"})
    ]
    
    # Mock s3fs to avoid import error
    with patch.dict("sys.modules", {"s3fs": MagicMock()}):
        catalog = create_catalog_from_options(storage_options=None, nodes=nodes)
        assert isinstance(catalog, S3Catalog)
        assert catalog.bucket_name == "other-bucket"

def test_create_catalog_fallback_to_filesystem():
    """
    Test that it falls back to FileSystemCatalog if no S3 path is found.
    """
    nodes = [
        {"params": {"dataset_id": "local.csv"}},
        {"params": {"path": "uploads/data.csv"}}
    ]
    
    catalog = create_catalog_from_options(storage_options=None, nodes=nodes)
    assert isinstance(catalog, FileSystemCatalog)

def test_s3_catalog_init_with_env_vars():
    """
    Test that S3Catalog initializes s3fs correctly when relying on env vars.
    """
    mock_s3fs = MagicMock()
    
    # Simulate env vars
    env_vars = {
        "AWS_ACCESS_KEY_ID": "env_key",
        "AWS_SECRET_ACCESS_KEY": "env_secret",
        "AWS_REGION": "us-east-1"
    }
    
    with patch.dict(os.environ, env_vars):
        with patch.dict("sys.modules", {"s3fs": mock_s3fs}):
            # Init with empty options
            catalog = S3Catalog(bucket_name="test-bucket", storage_options={})
            
            # Verify s3fs.S3FileSystem was called
            # It should be called with empty kwargs if we rely on env vars, 
            # OR our _prepare_s3fs_options might pick up env vars if we explicitly coded that (we didn't, s3fs does it).
            # So we expect empty kwargs or just what was passed.
            
            mock_s3fs.S3FileSystem.assert_called()
            call_kwargs = mock_s3fs.S3FileSystem.call_args[1]
            
            # We expect NO explicit key/secret in kwargs if they weren't in storage_options
            assert "key" not in call_kwargs
            assert "secret" not in call_kwargs
            
            # s3fs will pick them up from os.environ itself.

def test_s3_catalog_explicit_creds_mapping():
    """
    Test that S3Catalog maps explicit aws_access_key_id to key for s3fs.
    """
    mock_s3fs = MagicMock()
    
    options = {
        "aws_access_key_id": "explicit_key",
        "aws_secret_access_key": "explicit_secret",
        "region": "us-west-2"
    }
    
    with patch.dict("sys.modules", {"s3fs": mock_s3fs}):
        catalog = S3Catalog(bucket_name="test-bucket", storage_options=options)
        
        mock_s3fs.S3FileSystem.assert_called()
        call_kwargs = mock_s3fs.S3FileSystem.call_args[1]
        
        assert call_kwargs["key"] == "explicit_key"
        assert call_kwargs["secret"] == "explicit_secret"
        assert call_kwargs["client_kwargs"]["region_name"] == "us-west-2"

def test_s3_catalog_region_arg_priority():
    """
    Test that region_name argument in __init__ is respected.
    """
    mock_s3fs = MagicMock()
    
    with patch.dict("sys.modules", {"s3fs": mock_s3fs}):
        # Pass region_name explicitly
        catalog = S3Catalog(bucket_name="test-bucket", region_name="eu-central-1")
        
        mock_s3fs.S3FileSystem.assert_called()
        call_kwargs = mock_s3fs.S3FileSystem.call_args[1]
        
        assert call_kwargs["client_kwargs"]["region_name"] == "eu-central-1"
