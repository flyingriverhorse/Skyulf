"""Source readers implementing the ``BaseConnector`` contract.

``base`` defines the interface; ``file`` reads CSV, Excel, Parquet and JSON from
the configured upload directory; ``s3`` reads Parquet and CSV objects from
S3-compatible storage. Nothing is re-exported here — import the connector you
need by path.
"""
