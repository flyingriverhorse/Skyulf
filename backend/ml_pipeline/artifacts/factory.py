import os
import logging
from typing import Tuple, Union, Optional

from backend.config import get_settings
from backend.ml_pipeline.artifacts.store import ArtifactStore
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore

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
            return LocalArtifactStore(base_path=artifact_uri)

    @staticmethod
    def create_store_for_job(job_id: str, is_s3_source: bool = False) -> Tuple[ArtifactStore, str]:
        """
        Determines the correct storage location for a new job based on:
        1. The data source type (S3 vs Local)
        2. Configuration settings (UPLOAD_TO_S3_FOR_LOCAL_FILES, SAVE_S3_ARTIFACTS_LOCALLY)
        
        Returns:
            Tuple[ArtifactStore, str]: The instantiated store and the base artifact URI.
        """
        settings = get_settings()
        s3_bucket = settings.S3_ARTIFACT_BUCKET
        
        use_s3 = False
        
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
                bucket_name=s3_bucket, 
                prefix=job_id, 
                storage_options=storage_options
            )
            base_uri = f"s3://{s3_bucket}/{job_id}"
            return store, base_uri
        else:
            # Create Local Store
            base_path = os.path.join(settings.TRAINING_ARTIFACT_DIR, job_id)
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
        
        return S3ArtifactStore(
            bucket_name=bucket, 
            prefix=prefix, 
            storage_options=storage_options
        )

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
