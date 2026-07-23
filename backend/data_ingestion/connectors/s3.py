import logging
import re
from typing import cast

import polars as pl

from backend.config import get_settings
from backend.data_ingestion.connectors.base import BaseConnector
from backend.exceptions.core import ForbiddenException, ResourceNotFoundException

logger = logging.getLogger(__name__)

# Matches a bare HTTP status code (403/404) in an underlying object_store /
# botocore error message, e.g. "... status: 403 Forbidden ...". Used to
# classify S3 errors into typed exceptions instead of leaving callers to
# substring-match on `str(exc)`.
_HTTP_403_RE = re.compile(r"\b403\b")
_HTTP_404_RE = re.compile(r"\b404\b")


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

    @staticmethod
    def _map_storage_option_keys(options: dict) -> dict:
        """Map s3fs/boto3 option keys to the Polars/object_store equivalents, in place."""
        # Polars expects: aws_access_key_id, aws_secret_access_key, region, endpoint_url
        if "key" in options and "aws_access_key_id" not in options:
            options["aws_access_key_id"] = options.pop("key")

        if "secret" in options and "aws_secret_access_key" not in options:
            options["aws_secret_access_key"] = options.pop("secret")

        if "region_name" in options and "region" not in options:
            options["region"] = options.pop("region_name")

        return options

    @staticmethod
    def _apply_trusted_endpoint(options: dict) -> dict:
        """Drop any caller-supplied S3 endpoint and use only the operator-configured one.

        `endpoint_url`/`aws_endpoint_url` in per-request `storage_options` would let a
        caller redirect outbound S3 requests to an arbitrary host (SSRF, including
        cloud metadata endpoints). The endpoint is only ever trusted from server-side
        config (`AWS_ENDPOINT_URL`), never from request-supplied storage options.
        """
        options.pop("endpoint_url", None)
        options.pop("aws_endpoint_url", None)
        configured_endpoint = get_settings().AWS_ENDPOINT_URL
        if configured_endpoint:
            options["endpoint_url"] = configured_endpoint
        return options

    def _get_storage_options(self) -> dict[str, str]:
        """
        Ensure all storage options are strings for Polars and map common keys.
        """
        options = self.storage_options.copy() if self.storage_options else {}
        options = self._map_storage_option_keys(options)
        options = self._apply_trusted_endpoint(options)

        # Convert to strings for Polars
        return {k: str(v) for k, v in options.items() if v is not None}

    async def connect(self) -> bool:
        # Simple check by trying to read schema
        try:
            await self.get_schema()
            return True
        except (ForbiddenException, ResourceNotFoundException):
            # Preserve typed access-error classification for callers instead
            # of collapsing it into a generic ConnectionError.
            raise
        except Exception as e:
            logger.error(
                "S3 connection check failed for %s: %s", self.path, self._sanitize_error(e)
            )
            raise ConnectionError(f"Failed to connect to S3 path {self.path}") from e

    @staticmethod
    def _scan_schema(scan_fn, path: str, options: dict[str, str]) -> dict[str, str]:
        """Lazily scan `path` with the given Polars scan function and collect its schema."""
        lf = scan_fn(path, storage_options=options)
        return {name: str(dtype) for name, dtype in lf.collect_schema().items()}

    def _try_csv_schema(self, options: dict[str, str]) -> dict[str, str] | None:
        """Attempt to read the schema via scan_csv, returning None instead of raising on failure."""
        try:
            return self._scan_schema(pl.scan_csv, self.path, options)
        except Exception:
            return None  # nosec B110 - Expected fallback: CSV extension but try standard flow next

    def _raise_classified_schema_error(self, e: Exception) -> None:
        """Classify a schema-scan failure and raise the corresponding typed exception."""
        msg = str(e)
        if "169.254.169.254" in msg:
            raise ValueError(
                "S3 Connection Error: Could not find AWS credentials. "
                "If running locally, ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "are set or passed in storage_options."
            ) from e
        # Classify into typed exceptions so callers (e.g.
        # DataIngestionService.get_sample) can use isinstance checks
        # instead of substring-matching on the raw error message.
        if _HTTP_403_RE.search(msg):
            raise ForbiddenException(message=f"Access denied reading S3 path {self.path}") from e
        if _HTTP_404_RE.search(msg):
            raise ResourceNotFoundException(message=f"S3 resource not found: {self.path}") from e
        raise ValueError(
            f"Could not infer schema for {self.path}. Ensure it is a valid Parquet or CSV file "
            "and that the configured S3 credentials have access."
        ) from e

    async def get_schema(self) -> dict[str, str]:
        options = self._get_storage_options()

        # Optimization: Check extension first
        if self.path.lower().endswith(".csv"):
            schema = self._try_csv_schema(options)
            if schema is not None:
                return schema

        # Try Parquet first, then CSV
        try:
            # Use read_parquet_schema if available or scan
            # scan_parquet is lazy and efficient
            return self._scan_schema(pl.scan_parquet, self.path, options)
        except Exception:
            try:
                return self._scan_schema(pl.scan_csv, self.path, options)
            except Exception as e:
                self._raise_classified_schema_error(e)
                raise  # pragma: no cover - _raise_classified_schema_error always raises

    async def fetch_data(self, query: str | None = None, limit: int | None = None) -> pl.DataFrame:
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
