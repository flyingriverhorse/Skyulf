import os
import logging
from typing import Optional, Any
import pandas as pd
from sqlalchemy.orm import Session
from skyulf.data.catalog import DataCatalog
from backend.config import get_settings

logger = logging.getLogger(__name__)

class FileSystemCatalog(DataCatalog):
    """
    Concrete implementation that reads/writes files from the local filesystem.
    Replaces the old 'DataLoader'.
    """
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = get_settings().UPLOAD_DIR
        self.base_path = base_path

    def _get_path(self, dataset_id: str) -> str:
        # Security check to prevent directory traversal
        # If dataset_id is an absolute path (legacy behavior), use it directly if it starts with base_path
        # Otherwise, join with base_path
        
        # Simple check: if it looks like a full path, check if it exists
        if os.path.isabs(dataset_id) and os.path.exists(dataset_id):
             return dataset_id

        safe_id = os.path.basename(dataset_id)
        return os.path.join(self.base_path, safe_id)

    def load(self, dataset_id: str, **kwargs) -> pd.DataFrame:
        path = self._get_path(dataset_id)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset {dataset_id} not found at {path}")

        # Handle sampling if requested
        limit = kwargs.get("limit", None)
        
        try:
            if path.endswith(".csv"):
                return pd.read_csv(path, nrows=limit)
            elif path.endswith(".parquet"):
                df = pd.read_parquet(path)
                return df.head(limit) if limit else df
            elif path.endswith(".json"):
                return pd.read_json(path).head(limit) if limit else pd.read_json(path)
            elif path.endswith(".xlsx") or path.endswith(".xls"):
                 return pd.read_excel(path).head(limit) if limit else pd.read_excel(path)
            else:
                # Fallback: try reading as parquet if no extension
                try:
                    df = pd.read_parquet(path)
                    return df.head(limit) if limit else df
                except Exception:
                    raise ValueError(f"Unsupported format or file not found: {dataset_id}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            raise e

    def save(self, dataset_id: str, data: pd.DataFrame, **kwargs) -> None:
        path = self._get_path(dataset_id)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if path.endswith(".csv"):
            data.to_csv(path, index=False)
        elif path.endswith(".parquet"):
            data.to_parquet(path, index=False)
        else:
            # Default to parquet for internal storage if extension not clear
            if not path.endswith(".parquet"):
                path += ".parquet"
            data.to_parquet(path, index=False)

    def exists(self, dataset_id: str) -> bool:
        return os.path.exists(self._get_path(dataset_id))


class S3Catalog(DataCatalog):
    """
    Implementation that reads/writes files from AWS S3.
    Requires 's3fs' and 'boto3' to be installed.
    """
    def __init__(self, bucket_name: str, region_name: str = None):
        self.bucket_name = bucket_name
        self.storage_options = {}
        if region_name:
            self.storage_options["client_kwargs"] = {"region_name": region_name}
        
        # Check for s3fs
        try:
            import s3fs
        except ImportError:
            logger.error("s3fs is required for S3Catalog. Please install it with `pip install s3fs`.")
            raise ImportError("s3fs is required for S3Catalog")

    def _get_s3_path(self, dataset_id: str) -> str:
        # If it already starts with s3://, use it
        if dataset_id.startswith("s3://"):
            return dataset_id
        # Otherwise assume it's a key in the bucket
        return f"s3://{self.bucket_name}/{dataset_id}"

    def load(self, dataset_id: str, **kwargs) -> pd.DataFrame:
        path = self._get_s3_path(dataset_id)
        limit = kwargs.get("limit", None)
        
        # Merge storage options: self.storage_options (global) + kwargs (per-call)
        # kwargs take precedence
        storage_options = self.storage_options.copy()
        if "storage_options" in kwargs:
            storage_options.update(kwargs["storage_options"])
        
        logger.info(f"Loading from S3: {path}")
        
        try:
            # Pandas supports s3:// paths via s3fs
            if path.endswith(".csv"):
                return pd.read_csv(path, nrows=limit, storage_options=storage_options)
            elif path.endswith(".parquet"):
                df = pd.read_parquet(path, storage_options=storage_options)
                return df.head(limit) if limit else df
            elif path.endswith(".json"):
                return pd.read_json(path, storage_options=storage_options).head(limit) if limit else pd.read_json(path, storage_options=storage_options)
            else:
                # Default to parquet
                df = pd.read_parquet(path, storage_options=storage_options)
                return df.head(limit) if limit else df
        except Exception as e:
            logger.error(f"Error loading from S3 {path}: {e}")
            raise e

    def save(self, dataset_id: str, data: pd.DataFrame, **kwargs) -> None:
        path = self._get_s3_path(dataset_id)
        logger.info(f"Saving to S3: {path}")
        
        if path.endswith(".csv"):
            data.to_csv(path, index=False, storage_options=self.storage_options)
        else:
            if not path.endswith(".parquet"):
                path += ".parquet"
            data.to_parquet(path, index=False, storage_options=self.storage_options)

    def exists(self, dataset_id: str) -> bool:
        # This is a bit expensive, but accurate
        import s3fs
        fs = s3fs.S3FileSystem(**self.storage_options)
        path = self._get_s3_path(dataset_id)
        return fs.exists(path)


class SmartCatalog(DataCatalog):
    """
    A wrapper catalog that resolves Database IDs to file paths/keys
    and dispatches to the appropriate underlying catalog (S3 or FileSystem).
    """
    def __init__(self, session: Session, fs_catalog: FileSystemCatalog = None, s3_catalog: S3Catalog = None):
        self.session = session
        self.fs_catalog = fs_catalog or FileSystemCatalog()
        self.s3_catalog = s3_catalog
        
        # Try to init S3 catalog from env if not provided
        if not self.s3_catalog:
            bucket = os.getenv("S3_BUCKET_NAME")
            if bucket:
                try:
                    self.s3_catalog = S3Catalog(bucket_name=bucket)
                except ImportError:
                    pass # s3fs not installed or configured

    def _resolve_id(self, dataset_id: str) -> tuple[str, dict]:
        """
        Resolves a dataset ID to a (path, options) tuple.
        """
        # If it's a numeric ID, try to resolve it via DB
        if str(dataset_id).isdigit():
            try:
                # Local import to avoid circular dependencies
                from backend.database.models import DataSource
                from backend.utils.file_utils import extract_file_path_from_source
                
                ds = self.session.query(DataSource).filter(DataSource.id == int(dataset_id)).first()
                if ds:
                    path = extract_file_path_from_source(ds.to_dict())
                    if path:
                        logger.info(f"SmartCatalog: Resolved ID {dataset_id} -> {path}")
                        
                        # Extract storage options if available (e.g. S3 credentials)
                        options = {}
                        if ds.config and "storage_options" in ds.config:
                            options["storage_options"] = ds.config["storage_options"]
                            
                        return str(path), options
                    else:
                        logger.warning(f"SmartCatalog: DataSource {dataset_id} has no path")
                else:
                    logger.warning(f"SmartCatalog: DataSource {dataset_id} not found")
            except Exception as e:
                logger.error(f"SmartCatalog: Error resolving ID {dataset_id}: {e}")
        
        # Return original if not resolved
        return dataset_id, {}

    def _get_catalog_for_path(self, path: str) -> DataCatalog:
        if path.startswith("s3://"):
            if not self.s3_catalog:
                # If S3 catalog is not configured globally, we might still be able to use it
                # if credentials are provided per-call.
                # We'll try to instantiate a dummy one if needed, or raise error if no creds provided later.
                try:
                    self.s3_catalog = S3Catalog(bucket_name="placeholder") # Bucket name ignored if full path used
                except ImportError:
                     raise ValueError("S3 path encountered but s3fs is not installed.")
            return self.s3_catalog
        return self.fs_catalog

    def load(self, dataset_id: str, **kwargs) -> pd.DataFrame:
        resolved_id, options = self._resolve_id(dataset_id)
        
        # Merge resolved options (from DB) with call-time kwargs
        # Call-time kwargs take precedence? Or DB? 
        # Usually DB has the credentials, so we want to ensure they are passed.
        if options:
            if "storage_options" in kwargs:
                # Merge dictionaries
                merged_opts = options["storage_options"].copy()
                merged_opts.update(kwargs["storage_options"])
                kwargs["storage_options"] = merged_opts
            else:
                kwargs.update(options)

        catalog = self._get_catalog_for_path(resolved_id)
        return catalog.load(resolved_id, **kwargs)

    def save(self, dataset_id: str, data: pd.DataFrame, **kwargs) -> None:
        # We generally don't resolve IDs for saving (usually saving to new artifacts)
        # But if we wanted to overwrite a dataset by ID, we could.
        resolved_id, options = self._resolve_id(dataset_id)
        
        if options:
             if "storage_options" in kwargs:
                merged_opts = options["storage_options"].copy()
                merged_opts.update(kwargs["storage_options"])
                kwargs["storage_options"] = merged_opts
             else:
                kwargs.update(options)

        catalog = self._get_catalog_for_path(resolved_id)
        return catalog.save(resolved_id, data, **kwargs)

    def exists(self, dataset_id: str) -> bool:
        resolved_id, _ = self._resolve_id(dataset_id)
        catalog = self._get_catalog_for_path(resolved_id)
        return catalog.exists(resolved_id)
