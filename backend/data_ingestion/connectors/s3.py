"""S3 / object-store connector for the ingestion pipeline.

Reads Parquet and CSV objects through polars' lazy ``scan_*`` readers. Two
security constraints shape this module and are enforced below rather than left
to callers: a request-supplied ``storage_options`` entry may never set the S3
endpoint — that would let a caller redirect outbound requests to an arbitrary
host, including cloud metadata services — and the object path may itself be a
presigned URL, i.e. a bearer credential, so every log line is passed through
``redact_credentials``.
"""

import logging
import re
from typing import cast

import polars as pl

from backend.config import get_settings
from backend.data_ingestion.connectors.base import BaseConnector
from backend.exceptions.core import ForbiddenException, ResourceNotFoundException
from backend.utils.logging_utils import redact_credentials

logger = logging.getLogger(__name__)

# Matches a bare HTTP status code (403/404) in an underlying object_store /
# botocore error message, e.g. "... status: 403 Forbidden ...". Used to
# classify S3 errors into typed exceptions instead of leaving callers to
# substring-match on `str(exc)`.
_HTTP_403_RE = re.compile(r"\b403\b")
_HTTP_404_RE = re.compile(r"\b404\b")


class S3Connector(BaseConnector):
    """Connector for S3-compatible object storage, backed by polars lazy scans.

    Only Parquet and CSV are supported. Failures from the underlying object
    store are classified into :class:`ForbiddenException` and
    :class:`ResourceNotFoundException` so callers can branch on exception type
    instead of substring-matching an error message.
    """

    def __init__(self, path: str, storage_options: dict | None = None):
        """Store the object path and the per-request client options.

        Args:
            path: ``s3://`` URL of the object, or a presigned HTTPS URL. Either
                form is a bearer credential, so it is redacted before logging.
            storage_options: Credentials and client options forwarded to polars,
                with s3fs/boto3 key spellings mapped to the object_store ones.
                Any ``endpoint_url`` / ``aws_endpoint_url`` entry is discarded:
                the endpoint is taken only from server-side ``AWS_ENDPOINT_URL``.
        """
        self.path = path
        self.storage_options = storage_options or {}
        # path is caller-supplied and may itself be a presigned URL, i.e. a bearer credential.
        logger.info(
            "Initialized S3Connector for %s with option keys: %s",
            redact_credentials(path),
            list(self.storage_options.keys()),
        )

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
        """Ensure all storage options are strings for Polars and map common keys."""
        options = self.storage_options.copy() if self.storage_options else {}
        options = self._map_storage_option_keys(options)
        options = self._apply_trusted_endpoint(options)

        # Convert to strings for Polars
        return {k: str(v) for k, v in options.items() if v is not None}

    async def connect(self) -> bool:
        """Probe connectivity by reading the object's schema.

        Raises:
            ForbiddenException: If the credentials were refused (HTTP 403).
                Re-raised as-is rather than collapsed, because callers branch
                on the type.
            ResourceNotFoundException: If the object does not exist (HTTP 404).
            ConnectionError: For every other failure, with the cause chained.
        """
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
                "S3 connection check failed for %s: %s",
                redact_credentials(self.path),
                redact_credentials(e),
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
        except Exception:  # noqa: BLE001 - CSV schema probe, standard flow follows
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
        """Infer the object's schema with a lazy scan, Parquet first then CSV.

        A path ending in ``.csv`` is scanned as CSV up front to skip the probe
        round-trip. When neither format parses, ``_raise_classified_schema_error``
        converts the failure into a typed exception instead of surfacing the raw
        object-store message.
        """
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
        except Exception:  # noqa: BLE001 - parquet probe, CSV fallback follows
            try:
                return self._scan_schema(pl.scan_csv, self.path, options)
            except Exception as e:
                self._raise_classified_schema_error(e)
                raise  # pragma: no cover - _raise_classified_schema_error always raises

    async def fetch_data(self, limit: int | None = None) -> pl.DataFrame:
        """Collect up to ``limit`` rows from the object into a polars ``DataFrame``.

        The format comes from the extension to avoid a probe round-trip; an
        unrecognized extension is scanned as Parquet and falls back to CSV.
        ``limit`` is pushed into the lazy plan, so only that many rows are
        transferred rather than the whole object.
        """
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
                except Exception:  # noqa: BLE001 - format probe, CSV fallback follows
                    lf = pl.scan_csv(self.path, storage_options=options)

            if limit:
                lf = lf.limit(limit)

            return cast(pl.DataFrame, lf.collect())
        except Exception as e:
            logger.error(
                "Failed to fetch data from %s: %s",
                redact_credentials(self.path),
                redact_credentials(e),
            )
            raise RuntimeError(f"Failed to fetch data from S3 path {self.path}") from e

    async def validate(self) -> bool:
        """Validate by running :meth:`connect`, so the schema probe is the whole check."""
        return await self.connect()
