import logging
from typing import cast

import polars as pl

from backend.data_ingestion.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class S3Connector(BaseConnector):
    def __init__(self, path: str, storage_options: dict | None = None):
        self.path = path
        self.storage_options = storage_options or {}
        logger.info(
            "Initialized S3Connector for %s with option keys: %s",
            path,
            list(self.storage_options.keys()),
        )

    @staticmethod
    def _sanitize_error(error: Exception) -> str:
        message = str(error)
        for secret in ("aws_secret_access_key", "aws_access_key_id", "secret=", "key="):
            if secret in message:
                return "redacted sensitive S3 error"
        return message

    def _get_storage_options(self) -> dict[str, str]:
        """
        Ensure all storage options are strings for Polars and map common keys.
        """
        options = self.storage_options.copy() if self.storage_options else {}

        # Map s3fs/boto3 keys to Polars/object_store keys
        # Polars expects: aws_access_key_id, aws_secret_access_key, region, endpoint_url

        if "key" in options and "aws_access_key_id" not in options:
            options["aws_access_key_id"] = options.pop("key")

        if "secret" in options and "aws_secret_access_key" not in options:
            options["aws_secret_access_key"] = options.pop("secret")

        if "region_name" in options and "region" not in options:
            options["region"] = options.pop("region_name")

        # Convert to strings for Polars
        return {k: str(v) for k, v in options.items() if v is not None}

    async def connect(self) -> bool:
        # Simple check by trying to read schema
        try:
            await self.get_schema()
            return True
        except Exception as e:
            logger.error(
                "S3 connection check failed for %s: %s", self.path, self._sanitize_error(e)
            )
            raise ConnectionError(f"Failed to connect to S3 path {self.path}") from e

    async def get_schema(self) -> dict[str, str]:
        options = self._get_storage_options()

        # Optimization: Check extension first
        if self.path.lower().endswith(".csv"):
            try:
                lf = pl.scan_csv(self.path, storage_options=options)
                return {name: str(dtype) for name, dtype in lf.collect_schema().items()}
            except Exception:
                pass  # nosec B110 - Expected fallback: CSV extension but try standard flow next

        # Try Parquet first, then CSV
        try:
            # Use read_parquet_schema if available or scan
            # scan_parquet is lazy and efficient
            lf = pl.scan_parquet(self.path, storage_options=options)
            return {name: str(dtype) for name, dtype in lf.collect_schema().items()}
        except Exception:
            try:
                lf = pl.scan_csv(self.path, storage_options=options)
                return {name: str(dtype) for name, dtype in lf.collect_schema().items()}
            except Exception as e:
                msg = str(e)
                if "169.254.169.254" in msg:
                    raise ValueError(
                        "S3 Connection Error: Could not find AWS credentials. "
                        "If running locally, ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                        "are set or passed in storage_options."
                    ) from e
                raise ValueError(
                    f"Could not infer schema for {self.path}. Ensure it is a valid Parquet or CSV file "
                    "and that the configured S3 credentials have access."
                ) from e

    async def fetch_data(
        self, query: str | None = None, limit: int | None = None
    ) -> pl.DataFrame:
        options = self._get_storage_options()

        # Determine format based on extension to avoid lazy evaluation errors
        is_csv = self.path.lower().endswith(".csv")
        is_parquet = self.path.lower().endswith(".parquet")

        lf = None

        try:
            if is_csv:
                lf = pl.scan_csv(self.path, storage_options=options)
            elif is_parquet:
                lf = pl.scan_parquet(self.path, storage_options=options)
            else:
                try:
                    temp_lf = pl.scan_parquet(self.path, storage_options=options)
                    temp_lf.collect_schema()
                    lf = temp_lf
                except Exception:
                    lf = pl.scan_csv(self.path, storage_options=options)

            if limit:
                lf = lf.limit(limit)

            return cast(pl.DataFrame, lf.collect())
        except Exception as e:
            logger.error("Failed to fetch data from %s: %s", self.path, self._sanitize_error(e))
            raise RuntimeError(f"Failed to fetch data from S3 path {self.path}") from e

    async def validate(self) -> bool:
        return await self.connect()
