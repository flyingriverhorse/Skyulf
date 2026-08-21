from types import SimpleNamespace

from backend.data_ingestion.connectors import s3 as s3_connector_module
from backend.data_ingestion.connectors.s3 import S3Connector


def test_s3_connector_storage_options_conversion():
    """Test that storage_options are correctly converted to strings."""

    # Case 1: Mixed types (int, None, str)
    options = {
        "aws_access_key_id": "TEST_KEY",
        "aws_secret_access_key": "TEST_SECRET",
        "client_kwargs": {
            "region_name": "us-east-1"
        },  # Nested dict - might be an issue if Polars expects flat strings, but let's see what the fix does
        "use_ssl": True,
        "timeout": 30,
        "none_value": None,
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


def test_s3_connector_strips_caller_supplied_endpoint_url(monkeypatch):
    """Caller-supplied endpoint_url must never reach the S3 client (SSRF fix).

    Without server-side AWS_ENDPOINT_URL configured, any endpoint_url/aws_endpoint_url
    passed in per-request storage_options must be dropped, not honored.
    """
    monkeypatch.setattr(
        s3_connector_module,
        "get_settings",
        lambda: SimpleNamespace(AWS_ENDPOINT_URL=None),
    )
    connector = S3Connector(
        "s3://bucket/data.csv",
        storage_options={
            "endpoint_url": "http://169.254.169.254/latest/meta-data/",
            "aws_access_key_id": "x",
        },
    )
    processed_options = connector._get_storage_options()

    assert "endpoint_url" not in processed_options
    assert processed_options["aws_access_key_id"] == "x"


def test_s3_connector_uses_only_server_configured_endpoint_url(monkeypatch):
    """When AWS_ENDPOINT_URL is configured server-side, it is used regardless of
    what the caller supplied — the caller's value is discarded, not merged."""
    monkeypatch.setattr(
        s3_connector_module,
        "get_settings",
        lambda: SimpleNamespace(AWS_ENDPOINT_URL="https://trusted-minio.internal:9000"),
    )
    connector = S3Connector(
        "s3://bucket/data.csv",
        storage_options={"endpoint_url": "http://169.254.169.254/latest/meta-data/"},
    )
    processed_options = connector._get_storage_options()

    assert processed_options["endpoint_url"] == "https://trusted-minio.internal:9000"
