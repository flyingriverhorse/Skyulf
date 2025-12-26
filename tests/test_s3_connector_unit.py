import pytest
from backend.data_ingestion.connectors.s3 import S3Connector

def test_s3_connector_storage_options_conversion():
    """Test that storage_options are correctly converted to strings."""
    
    # Case 1: Mixed types (int, None, str)
    options = {
        "aws_access_key_id": "TEST_KEY",
        "aws_secret_access_key": "TEST_SECRET",
        "client_kwargs": {"region_name": "us-east-1"}, # Nested dict - might be an issue if Polars expects flat strings, but let's see what the fix does
        "use_ssl": True,
        "timeout": 30,
        "none_value": None
    }
    
    connector = S3Connector("s3://bucket/data.csv", storage_options=options)
    processed_options = connector._get_storage_options()
    
    assert processed_options["aws_access_key_id"] == "TEST_KEY"
    assert processed_options["aws_secret_access_key"] == "TEST_SECRET"
    assert processed_options["use_ssl"] == "True"
    assert processed_options["timeout"] == "30"
    assert "none_value" not in processed_options
    
    # Check that nested dicts are also stringified (Polars might not like this, but the fix does str(v))
    assert processed_options["client_kwargs"] == "{'region_name': 'us-east-1'}"

def test_s3_connector_empty_options():
    """Test with empty options."""
    connector = S3Connector("s3://bucket/data.csv", storage_options={})
    assert connector._get_storage_options() == {}

def test_s3_connector_none_options():
    """Test with None options."""
    connector = S3Connector("s3://bucket/data.csv", storage_options=None)
    assert connector._get_storage_options() == {}
