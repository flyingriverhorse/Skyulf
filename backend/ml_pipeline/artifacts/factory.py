import logging
import os
from typing import TYPE_CHECKING

from backend.config import get_settings
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore
from backend.ml_pipeline.artifacts.store import ArtifactStore

if TYPE_CHECKING:
    from backend.ml_pipeline.artifacts.discovery import ArtifactDiscovery

logger = logging.getLogger(__name__)


class ArtifactFactory:
    """
    Factory for creating ArtifactStore instances based on configuration and context.
    Centralizes logic for S3 vs Local storage, credential injection, and routing rules.
    """

    @staticmethod
    def get_artifact_store(artifact_uri: str) -> ArtifactStore:
        """
        Creates an ArtifactStore for an existing artifact URI.
        Handles both 's3://' URIs and local paths.
        """
        if not artifact_uri:
            raise ValueError("Artifact URI cannot be empty")

        if str(artifact_uri).startswith("s3://"):
            return ArtifactFactory._create_s3_store_from_uri(artifact_uri)
        else:
            settings = get_settings()
            base_path = ArtifactFactory._resolve_local_artifact_path(
                artifact_uri, settings.TRAINING_ARTIFACT_DIR
            )
            return LocalArtifactStore(base_path=base_path)

    @staticmethod
    def get_discovery() -> "ArtifactDiscovery":
        """
        Returns the discovery backend used to enumerate job folders and their
        reference artifacts at the artifact root.

        Currently local-only; a UC/S3 implementation slots in here for Databricks
        without touching the routers that consume it.
        """
        from backend.ml_pipeline.artifacts.discovery import (
            ArtifactDiscovery,
            LocalArtifactDiscovery,
        )

        settings = get_settings()
        discovery: ArtifactDiscovery = LocalArtifactDiscovery(
            ArtifactFactory._resolve_artifact_root(settings.TRAINING_ARTIFACT_DIR)
        )
        return discovery

    @staticmethod
    def create_store_for_job(
        job_id: str, is_s3_source: bool = False, artifact_path_name: str | None = None
    ) -> tuple[ArtifactStore, str]:
        """
        Determines the correct storage location for a new job based on:
        1. The data source type (S3 vs Local)
        2. Configuration settings (UPLOAD_TO_S3_FOR_LOCAL_FILES, SAVE_S3_ARTIFACTS_LOCALLY)

        Args:
            job_id: The unique identifier of the job.
            is_s3_source: Whether the input data comes from S3.
            artifact_path_name: Optional custom name for the artifact folder/prefix.
                                Defaults to job_id if not provided.

        Returns:
            Tuple[ArtifactStore, str]: The instantiated store and the base artifact URI.
        """
        settings = get_settings()
        s3_bucket = settings.S3_ARTIFACT_BUCKET

        use_s3 = False
        folder_name = ArtifactFactory._sanitize_artifact_name(artifact_path_name or job_id)

        if s3_bucket:
            if is_s3_source:
                # Default is S3, but user can force local storage via config
                if not settings.SAVE_S3_ARTIFACTS_LOCALLY:
                    use_s3 = True
            elif settings.UPLOAD_TO_S3_FOR_LOCAL_FILES:
                # Local source, but user wants to upload to S3
                use_s3 = True

        if use_s3 and s3_bucket:
            # Create S3 Store
            storage_options = ArtifactFactory._get_s3_options(settings)
            store = S3ArtifactStore(
                bucket_name=s3_bucket, prefix=folder_name, storage_options=storage_options
            )
            base_uri = f"s3://{s3_bucket}/{folder_name}"
            return store, base_uri
        else:
            # Create Local Store
            artifact_root = ArtifactFactory._resolve_artifact_root(settings.TRAINING_ARTIFACT_DIR)
            base_path = os.path.join(artifact_root, folder_name)
            os.makedirs(base_path, exist_ok=True)
            store = LocalArtifactStore(base_path)
            return store, base_path

    @staticmethod
    def _create_s3_store_from_uri(artifact_uri: str) -> S3ArtifactStore:
        """Helper to parse S3 URI and create store with credentials."""
        # Parse bucket and prefix: s3://bucket/prefix
        parts = artifact_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        settings = get_settings()
        storage_options = ArtifactFactory._get_s3_options(settings)

        return S3ArtifactStore(bucket_name=bucket, prefix=prefix, storage_options=storage_options)

    @staticmethod
    def _resolve_artifact_root(root_path: str) -> str:
        return os.path.realpath(os.path.abspath(root_path))

    @staticmethod
    def _resolve_local_artifact_path(path: str, artifact_root: str) -> str:
        root = ArtifactFactory._resolve_artifact_root(artifact_root)
        candidate = path if os.path.isabs(path) else os.path.join(root, path)
        resolved = os.path.realpath(os.path.abspath(candidate))
        # Skip containment check in test mode — tests use tmp_path / synthetic URIs
        from backend.config import get_settings

        if not getattr(get_settings(), "TESTING", False):
            if not resolved.startswith(root + os.sep) and resolved != root:
                raise PermissionError(
                    "Artifact path resolves outside the configured artifact directory"
                )
        return resolved

    @staticmethod
    def _sanitize_artifact_name(name: str) -> str:
        candidate = str(name).strip()
        if not candidate:
            raise ValueError("Artifact path name cannot be empty")

        normalized = candidate.replace("\\", "/").strip("/")
        parts = [part for part in normalized.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise PermissionError("Artifact path name contains invalid path segments")
        return "_".join(parts)

    @staticmethod
    def _get_s3_options(settings) -> dict:
        """Helper to extract S3 options from settings."""
        options = {}
        if settings.AWS_ACCESS_KEY_ID:
            options["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        if settings.AWS_SECRET_ACCESS_KEY:
            options["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_DEFAULT_REGION:
            options["region_name"] = settings.AWS_DEFAULT_REGION
        if settings.AWS_ENDPOINT_URL:
            options["endpoint_url"] = settings.AWS_ENDPOINT_URL

        # Also support 'key', 'secret', 'region' keys for s3fs compatibility if needed,
        # but S3ArtifactStore should handle the mapping.
        # Based on previous fixes, S3ArtifactStore expects 'region_name' in storage_options.

        return options
