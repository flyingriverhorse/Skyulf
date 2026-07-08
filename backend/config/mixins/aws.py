"""AWS and S3 configuration."""



class AWSMixin:
    """AWS credentials and S3 artifact settings."""

    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_SESSION_TOKEN: str | None = None
    AWS_DEFAULT_REGION: str | None = "us-east-1"
    AWS_ENDPOINT_URL: str | None = None
    AWS_BUCKET_NAME: str | None = None
    S3_ARTIFACT_BUCKET: str | None = None
    UPLOAD_TO_S3_FOR_LOCAL_FILES: bool = False
    SAVE_S3_ARTIFACTS_LOCALLY: bool = False
