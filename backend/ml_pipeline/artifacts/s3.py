import os
import tempfile
from typing import Any, Optional
import joblib
import logging

from .store import ArtifactStore

logger = logging.getLogger(__name__)

class S3ArtifactStore(ArtifactStore):
    def __init__(self, bucket_name: str, prefix: str = "", storage_options: Optional[dict] = None):
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self.storage_options = storage_options or {}
        
        # Map standard AWS keys to s3fs keys if needed
        if "aws_access_key_id" in self.storage_options and "key" not in self.storage_options:
            self.storage_options["key"] = self.storage_options.pop("aws_access_key_id")
        if "aws_secret_access_key" in self.storage_options and "secret" not in self.storage_options:
            self.storage_options["secret"] = self.storage_options.pop("aws_secret_access_key")
        
        # Handle endpoint_url
        endpoint = self.storage_options.pop("endpoint_url", None) or self.storage_options.pop("aws_endpoint_url", None)
        if endpoint:
            if "client_kwargs" not in self.storage_options:
                self.storage_options["client_kwargs"] = {}
            if "endpoint_url" not in self.storage_options["client_kwargs"]:
                self.storage_options["client_kwargs"]["endpoint_url"] = endpoint

        # Handle region_name (passed by some callers)
        region = self.storage_options.pop("region_name", None) or self.storage_options.pop("region", None)
        if region:
            if "client_kwargs" not in self.storage_options:
                self.storage_options["client_kwargs"] = {}
            if "region_name" not in self.storage_options["client_kwargs"]:
                self.storage_options["client_kwargs"]["region_name"] = region

        try:
            import s3fs
            self.fs = s3fs.S3FileSystem(**self.storage_options)
        except ImportError:
            raise ImportError("s3fs is required for S3ArtifactStore")
        except Exception as e:
            logger.error(f"Failed to initialize S3FileSystem: {e}")
            raise e

    def _get_s3_path(self, key: str) -> str:
        # Ensure key is safe
        safe_key = key.replace("\\", "/")
        if not safe_key.endswith(".joblib"):
            safe_key += ".joblib"
            
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
            logger.error(f"Failed to save artifact to S3 {path}: {e}")
            raise e

    def load(self, key: str) -> Any:
        path = self._get_s3_path(key)
        logger.info(f"Loading artifact from S3: {path}")
        
        if not self.fs.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path}")
            
        try:
            with self.fs.open(path, "rb") as f:
                return joblib.load(f)
        except Exception as e:
            logger.error(f"Failed to load artifact from S3 {path}: {e}")
            raise e

    def exists(self, key: str) -> bool:
        path = self._get_s3_path(key)
        return bool(self.fs.exists(path))

    def list_artifacts(self) -> list[str]:
        """List all artifacts in the store."""
        base_path = f"s3://{self.bucket_name}/{self.prefix}" if self.prefix else f"s3://{self.bucket_name}"
        
        try:
            # s3fs.ls might raise FileNotFoundError if the prefix doesn't exist
            if not self.fs.exists(base_path):
                return []
                
            files = self.fs.ls(base_path)
            keys = []
            for f in files:
                # s3fs ls returns full paths usually without protocol? or with?
                # usually 'bucket/prefix/file.joblib'
                filename = os.path.basename(f)
                if filename.endswith(".joblib"):
                    keys.append(filename[:-7])
                else:
                    keys.append(filename)
            return keys
        except Exception as e:
            logger.error(f"Failed to list artifacts in {base_path}: {e}")
            return []

    def get_artifact_uri(self, key: str) -> str:
        """Get the full URI/Path for a given artifact key."""
        return self._get_s3_path(key)
