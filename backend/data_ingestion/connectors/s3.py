import polars as pl
from typing import Dict, Optional
from backend.data_ingestion.connectors.base import BaseConnector

class S3Connector(BaseConnector):
    def __init__(self, path: str, storage_options: Optional[Dict] = None):
        self.path = path
        self.storage_options = storage_options or {}
        # Debug log to confirm code reload
        import logging
        logging.getLogger(__name__).info(f"Initialized S3Connector for {path} with options keys: {list(self.storage_options.keys())}")

    def _get_storage_options(self) -> Dict[str, str]:
        """
        Ensure all storage options are strings for Polars and map common keys.
        """
        options = self.storage_options.copy() if self.storage_options else {}
        
        # Map s3fs/boto3 keys to Polars/object_store keys
        # Polars expects: aws_access_key_id, aws_secret_access_key, region, endpoint_url
        
        if "key" in options and "aws_access_key_id" not in options:
            options["aws_access_key_id"] = options.pop("key")
            
        if "secret" in options and "aws_secret_access_key" not in options:
            options["aws_secret_access_key"] = options.pop("secret")
            
        if "region_name" in options and "region" not in options:
            options["region"] = options.pop("region_name")
            
        # Convert to strings for Polars
        return {k: str(v) for k, v in options.items() if v is not None}

    async def connect(self) -> bool:
        # Simple check by trying to read schema
        try:
            await self.get_schema()
            return True
        except Exception as e:
            # Log the error but don't raise immediately if it's just a connection check
            # Wait, validate() calls connect(), so we should raise or return False.
            # But get_sample calls connect() then fetch_data().
            # If connect fails, fetch_data will likely fail too.
            # Let's raise a clearer error.
            import logging
            logging.getLogger(__name__).error(f"S3 Connect failed for {self.path}: {e}")
            raise ConnectionError(f"Failed to connect to S3 path {self.path}: {e}")

    async def get_schema(self) -> Dict[str, str]:
        options = self._get_storage_options()
        
        # Optimization: Check extension first
        if self.path.lower().endswith(".csv"):
            try:
                lf = pl.scan_csv(self.path, storage_options=options)
                return {name: str(dtype) for name, dtype in lf.collect_schema().items()}
            except Exception:
                pass # Fallback to standard flow

        # Try Parquet first, then CSV
        try:
            # Use read_parquet_schema if available or scan
            # scan_parquet is lazy and efficient
            lf = pl.scan_parquet(self.path, storage_options=options)
            return {name: str(dtype) for name, dtype in lf.collect_schema().items()}
        except Exception:
            try:
                lf = pl.scan_csv(self.path, storage_options=options)
                return {name: str(dtype) for name, dtype in lf.collect_schema().items()}
            except Exception as e:
                msg = str(e)
                if "169.254.169.254" in msg:
                    raise ValueError(f"S3 Connection Error: Could not find AWS credentials. If running locally, ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set or passed in storage_options. Original error: {e}")
                raise ValueError(f"Could not infer schema for {self.path}. Ensure it is a valid Parquet or CSV file. Error: {e}")

    async def fetch_data(
        self, query: Optional[str] = None, limit: Optional[int] = None
    ) -> pl.DataFrame:
        options = self._get_storage_options()
        
        # Determine format based on extension to avoid lazy evaluation errors
        is_csv = self.path.lower().endswith(".csv")
        is_parquet = self.path.lower().endswith(".parquet")
        
        lf = None
        
        if is_csv:
             lf = pl.scan_csv(self.path, storage_options=options)
        elif is_parquet:
             lf = pl.scan_parquet(self.path, storage_options=options)
        else:
            # Fallback: Try Parquet, verify schema, else CSV
            try:
                temp_lf = pl.scan_parquet(self.path, storage_options=options)
                # Force schema check to ensure it's actually Parquet
                temp_lf.collect_schema()
                lf = temp_lf
            except Exception:
                lf = pl.scan_csv(self.path, storage_options=options)
            
        if limit:
            lf = lf.limit(limit)
            
        return lf.collect()

    async def validate(self) -> bool:
        return await self.connect()
