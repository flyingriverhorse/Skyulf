"""AWS and S3 configuration."""

from typing import Optional


class AWSMixin:
    """AWS credentials and S3 artifact settings."""

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None
    AWS_DEFAULT_REGION: Optional[str] = "us-east-1"
    AWS_ENDPOINT_URL: Optional[str] = None
    AWS_BUCKET_NAME: Optional[str] = None
    S3_ARTIFACT_BUCKET: Optional[str] = None
    UPLOAD_TO_S3_FOR_LOCAL_FILES: bool = False
    SAVE_S3_ARTIFACTS_LOCALLY: bool = False
