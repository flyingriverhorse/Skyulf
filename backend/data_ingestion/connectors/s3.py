import polars as pl
from typing import Dict, Optional
from backend.data_ingestion.connectors.base import BaseConnector

class S3Connector(BaseConnector):
    def __init__(self, path: str, storage_options: Dict = None):
        self.path = path
        self.storage_options = storage_options or {}

    async def connect(self) -> bool:
        # Simple check by trying to read schema
        try:
            await self.get_schema()
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to S3 path {self.path}: {e}")

    async def get_schema(self) -> Dict[str, str]:
        # Try Parquet first, then CSV
        try:
            # Use read_parquet_schema if available or scan
            # scan_parquet is lazy and efficient
            lf = pl.scan_parquet(self.path, storage_options=self.storage_options)
            return {name: str(dtype) for name, dtype in lf.schema.items()}
        except Exception:
            try:
                lf = pl.scan_csv(self.path, storage_options=self.storage_options)
                return {name: str(dtype) for name, dtype in lf.schema.items()}
            except Exception as e:
                raise ValueError(f"Could not infer schema for {self.path}. Ensure it is a valid Parquet or CSV file. Error: {e}")

    async def fetch_data(
        self, query: Optional[str] = None, limit: Optional[int] = None
    ) -> pl.DataFrame:
        try:
            lf = pl.scan_parquet(self.path, storage_options=self.storage_options)
        except Exception:
            lf = pl.scan_csv(self.path, storage_options=self.storage_options)
            
        if limit:
            lf = lf.limit(limit)
            
        return lf.collect()

    async def validate(self) -> bool:
        return await self.connect()
