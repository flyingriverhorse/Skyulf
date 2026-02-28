import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from skyulf.engines import SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)

class DataService:
    """
    Unified entry point for data I/O.
    Favors Polars for speed but falls back to Pandas if necessary.
    """
    
    def __init__(self):
        pass

    async def load_file(self, path: Union[str, Path], force_type: Optional[str] = None, storage_options: Optional[dict] = None) -> Any:
        """
        Loads a file into a DataFrame (Polars or Pandas).
        
        Args:
            path: Path to the file.
            force_type: 'pandas' or 'polars'. If None, defaults to Polars if available.
            storage_options: Dictionary containing storage options (e.g. credentials for S3).
            
        Returns:
            A DataFrame object (pl.DataFrame or pd.DataFrame) that is compatible 
            with SkyulfDataFrame protocol.
        """
        path_str = str(path)
        
        # Skip existence check for S3 paths
        if not path_str.startswith("s3://") and not Path(path_str).exists():
            raise FileNotFoundError(f"File not found: {path_str}")

        # Determine which engine to use for loading
        use_polars = HAS_POLARS
        if force_type == "pandas":
            use_polars = False
        elif force_type == "polars" and not HAS_POLARS:
            logger.warning("Polars requested but not installed. Falling back to Pandas.")
            use_polars = False

        try:
            if use_polars:
                try:
                    return self._load_polars(path_str, storage_options)
                except Exception as pl_err:
                    logger.warning(f"Polars load failed for {path_str}: {pl_err}. Falling back to Pandas.")
                    # Fallback to pandas, then convert to Polars
                    pdf = self._load_pandas(path_str, storage_options)
                    return pl.from_pandas(pdf)
            else:
                return self._load_pandas(path_str, storage_options)
        except Exception as e:
            logger.error(f"Failed to load file {path_str}: {e}")
            raise

    def _load_polars(self, path: str, storage_options: Optional[dict] = None) -> Any:
        """Load using Polars."""
        if path.endswith(".parquet"):
            return pl.read_parquet(path, storage_options=storage_options)
        elif path.endswith(".csv"):
            # Polars read_csv supports storage_options in recent versions
            return pl.read_csv(path, ignore_errors=True, storage_options=storage_options) 
        elif path.endswith(".json"):
            return pl.read_json(path) # JSON usually local, but if S3 needed, might need scan_ndjson
        else:
            # Fallback to pandas for other formats, then convert
            return pl.from_pandas(self._load_pandas(path, storage_options))

    def _load_pandas(self, path: str, storage_options: Optional[dict] = None) -> pd.DataFrame:
        """Load using Pandas."""
        if path.endswith(".parquet"):
            return pd.read_parquet(path, storage_options=storage_options)
        elif path.endswith(".csv"):
            return pd.read_csv(path, storage_options=storage_options)
        elif path.endswith(".json"):
            return pd.read_json(path, storage_options=storage_options)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            return pd.read_excel(path, storage_options=storage_options)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    async def get_sample(self, path: Union[str, Path], limit: int = 5) -> Any:
        """
        Get a sample of the data as a list of dictionaries.
        Efficiently reads only the first N rows.
        """
        path_str = str(path)
        
        if HAS_POLARS:
            try:
                if path_str.endswith(".parquet"):
                    return pl.scan_parquet(path_str).limit(limit).collect().to_dicts()
                elif path_str.endswith(".csv"):
                    return pl.scan_csv(path_str, ignore_errors=True).limit(limit).collect().to_dicts()
                elif path_str.endswith(".json"):
                    # JSON scan is experimental/limited, use read
                    return pl.read_json(path_str).head(limit).to_dicts()
            except Exception as e:
                logger.warning(f"Polars lazy scan failed for {path_str}: {e}. Falling back to eager load.")

        # Fallback: Load full file (or eager load) and take head
        data = await self.load_file(path, force_type="pandas" if not HAS_POLARS else None)
        
        # Handle SkyulfWrapper
        if hasattr(data, "to_pandas"): # Wrapper or Polars
             # If it's Polars
             if HAS_POLARS and isinstance(data, pl.DataFrame):
                 return data.head(limit).to_dicts()
             # If it's Wrapper or Pandas
             df = data.to_pandas()
             return df.head(limit).to_dict(orient="records")
        
        # It's likely a Pandas DataFrame
        return data.head(limit).to_dict(orient="records")

    async def save_artifact(self, data: Any, path: Union[str, Path]) -> None:
        """
        Save a DataFrame to disk (Parquet preferred).
        
        Args:
            data: The DataFrame (Polars, Pandas, or Skyulf wrapper).
            path: Destination path.
        """
        path_str = str(path)
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        
        # Unwrap if it's a Skyulf wrapper
        if hasattr(data, "_df"): # Hacky check for wrapper, better to use protocol
             # If it's our wrapper, we can get the underlying df
             # But let's try to use the engine registry to handle it cleanly
             pass

        engine = get_engine(data)
        
        # If we have Polars and the data is compatible, use Polars for writing (faster)
        if HAS_POLARS and engine.__name__ == "PolarsEngine":
            # It's already Polars or wrapped Polars
            if hasattr(data, "write_parquet"):
                data.write_parquet(path_str)
            elif hasattr(data, "_df") and hasattr(data._df, "write_parquet"):
                 data._df.write_parquet(path_str)
            else:
                # Should not happen if engine is PolarsEngine
                logger.warning("PolarsEngine detected but write_parquet missing. Converting...")
                pl.from_pandas(data).write_parquet(path_str)
                
        elif HAS_POLARS:
            # It's Pandas (or other), convert to Polars for fast write
            try:
                import pandas as pd
                # Zero-copy convert via Arrow if possible
                if hasattr(data, "to_arrow"):
                    pl.from_arrow(data.to_arrow()).write_parquet(path_str)
                elif isinstance(data, pd.DataFrame):
                    pl.from_pandas(data).write_parquet(path_str)
                else:
                    self._save_pandas(data, path_str)
            except Exception as e:
                logger.warning(f"Polars write failed ({e}), falling back to Pandas.")
                self._save_pandas(data, path_str)
        else:
            # No Polars, use Pandas
            self._save_pandas(data, path_str)

    def _save_pandas(self, data: Any, path: str):
        if hasattr(data, "to_pandas"):
            df = data.to_pandas()
        else:
            df = data
            
        if path.endswith(".parquet"):
            df.to_parquet(path)
        elif path.endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            # Default to parquet if extension unknown/other
            if not path.endswith(".parquet"):
                path += ".parquet"
            df.to_parquet(path)
