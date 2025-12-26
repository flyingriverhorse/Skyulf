import os
import logging
import pandas as pd
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
