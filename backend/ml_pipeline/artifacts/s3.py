import contextlib
import logging
from pathlib import Path
from typing import Any

import joblib

from .store import ArtifactStore

logger = logging.getLogger(__name__)


class S3ArtifactStore(ArtifactStore):
    def __init__(self, bucket_name: str, prefix: str = "", storage_options: dict | None = None):
        self.bucket_name = bucket_name.strip()
        self.prefix = self._normalize_prefix(prefix)
        self.storage_options = storage_options or {}
        if not self.bucket_name:
            raise ValueError("S3 bucket name cannot be empty")

        self._remap_aws_credential_keys()
        self._apply_endpoint_url()
        self._apply_region_name()
        self._init_filesystem()

    def _remap_aws_credential_keys(self) -> None:
        """Map standard AWS credential keys to the s3fs-expected keys, in place."""
        if "aws_access_key_id" in self.storage_options and "key" not in self.storage_options:
            self.storage_options["key"] = self.storage_options.pop("aws_access_key_id")
        if "aws_secret_access_key" in self.storage_options and "secret" not in self.storage_options:
            self.storage_options["secret"] = self.storage_options.pop("aws_secret_access_key")

    def _apply_endpoint_url(self) -> None:
        """Move an endpoint_url/aws_endpoint_url option into ``client_kwargs``, in place."""
        endpoint = self.storage_options.pop("endpoint_url", None) or self.storage_options.pop(
            "aws_endpoint_url", None
        )
        if endpoint:
            if "client_kwargs" not in self.storage_options:
                self.storage_options["client_kwargs"] = {}
            if "endpoint_url" not in self.storage_options["client_kwargs"]:
                self.storage_options["client_kwargs"]["endpoint_url"] = endpoint

    def _apply_region_name(self) -> None:
        """Move a region_name/region option into ``client_kwargs``, in place."""
        region = self.storage_options.pop("region_name", None) or self.storage_options.pop(
            "region", None
        )
        if region:
            if "client_kwargs" not in self.storage_options:
                self.storage_options["client_kwargs"] = {}
            if "region_name" not in self.storage_options["client_kwargs"]:
                self.storage_options["client_kwargs"]["region_name"] = region

    def _init_filesystem(self) -> None:
        """Create the s3fs filesystem client, raising a clear error on failure."""
        try:
            import s3fs  # ty: ignore[unresolved-import]

            self.fs = s3fs.S3FileSystem(**self.storage_options)
        except ImportError:
            raise ImportError("s3fs is required for S3ArtifactStore") from None
        except Exception as e:
            logger.error("Failed to initialize S3 filesystem client: %s", self._sanitize_error(e))
            raise RuntimeError("Failed to initialize S3 artifact storage") from e

    @staticmethod
    def _sanitize_error(error: Exception) -> str:
        message = str(error)
        for secret in ("aws_secret_access_key", "aws_access_key_id", "secret=", "key="):
            if secret in message:
                return "redacted sensitive S3 error"
        return message

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        normalized = str(prefix or "").replace("\\", "/").strip("/")
        if not normalized:
            return ""
        parts = [part for part in normalized.split("/") if part]
        if any(part in {".", ".."} for part in parts):
            raise PermissionError("S3 prefix contains invalid path segments")
        return "/".join(parts)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        candidate = str(key).strip()
        if not candidate:
            raise ValueError("Artifact key cannot be empty")

        normalized = candidate.replace("\\", "/").strip("/")
        parts = [part for part in normalized.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise PermissionError("Artifact key contains invalid path segments")
        if len(parts) != 1:
            raise PermissionError("Artifact keys must not contain nested paths")

        filename = parts[0]
        if filename.endswith(".joblib"):
            filename = filename[:-7]
        if not filename:
            raise ValueError("Artifact key cannot be empty")
        return f"{filename}.joblib"

    def _get_s3_path(self, key: str) -> str:
        safe_key = self._sanitize_key(key)

        if self.prefix:
            return f"s3://{self.bucket_name}/{self.prefix}/{safe_key}"
        return f"s3://{self.bucket_name}/{safe_key}"

    def save(self, key: str, data: Any) -> None:
        path = self._get_s3_path(key)
        logger.info(f"Saving artifact to S3: {path}")

        # Use s3fs to open a file-like object for joblib
        try:
            with self.fs.open(path, "wb") as f:
                joblib.dump(data, f)
        except Exception as e:
            logger.error("Failed to save artifact to S3 %s: %s", path, self._sanitize_error(e))
            raise RuntimeError(f"Failed to save artifact to S3: {path}") from e

    def load(self, key: str) -> Any:
        """Load a joblib artifact from S3.

        Warning:
            ``joblib.load`` uses pickle internally and can execute arbitrary code.
            Only load artifacts that were saved by this application.
        """
        path = self._get_s3_path(key)
        logger.info(f"Loading artifact from S3: {path}")

        try:
            if not self.fs.exists(path):
                raise FileNotFoundError(f"Artifact not found: {path}")
            with self.fs.open(path, "rb") as f:
                return joblib.load(f)
        except Exception as e:
            logger.error("Failed to load artifact from S3 %s: %s", path, self._sanitize_error(e))
            if isinstance(e, FileNotFoundError):
                raise
            raise RuntimeError(f"Failed to load artifact from S3: {path}") from e

    def exists(self, key: str) -> bool:
        path = self._get_s3_path(key)
        try:
            return bool(self.fs.exists(path))
        except Exception as e:
            logger.error(
                "Failed to check artifact existence in S3 %s: %s", path, self._sanitize_error(e)
            )
            raise RuntimeError(f"Failed to check artifact existence in S3: {path}") from e

    def list_artifacts(self) -> list[str]:
        """List all artifacts in the store."""
        base_path = (
            f"s3://{self.bucket_name}/{self.prefix}" if self.prefix else f"s3://{self.bucket_name}"
        )

        try:
            # s3fs.ls might raise FileNotFoundError if the prefix doesn't exist
            if not self.fs.exists(base_path):
                return []

            files = self.fs.ls(base_path)
            keys = []
            for f in files:
                normalized = str(f).replace("s3://", "")
                bucket_prefix = f"{self.bucket_name}/"
                normalized = normalized.removeprefix(bucket_prefix)
                if self.prefix:
                    prefix = f"{self.prefix}/"
                    if not normalized.startswith(prefix):
                        continue
                    normalized = normalized.removeprefix(prefix)
                if "/" in normalized.strip("/"):
                    continue
                filename = Path(normalized).name
                if filename.endswith(".joblib"):
                    keys.append(filename.removesuffix(".joblib"))
                else:
                    keys.append(filename)
            return keys
        except Exception as e:
            logger.error("Failed to list artifacts in %s: %s", base_path, self._sanitize_error(e))
            return []

    def get_artifact_uri(self, key: str) -> str:
        """Get the full URI/Path for a given artifact key."""
        return self._get_s3_path(key)

    def close(self) -> None:
        close = getattr(self.fs, "close", None)
        if callable(close):
            close()

    def __del__(self) -> None:
        # best-effort cleanup; logging is unsafe during interpreter shutdown
        with contextlib.suppress(Exception):
            self.close()
