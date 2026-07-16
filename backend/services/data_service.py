import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from skyulf.engines import get_engine

logger = logging.getLogger(__name__)


def _collect(lf: "pl.LazyFrame") -> "pl.DataFrame":
    """Narrow `LazyFrame.collect()` back to `DataFrame` (sync path only)."""
    return cast("pl.DataFrame", lf.collect())


class DataService:
    """
    Unified entry point for data I/O.
    Favors Polars for speed but falls back to Pandas if necessary.
    """

    def __init__(self):
        pass

    async def load_file(
        self,
        path: str | Path,
        force_type: str | None = None,
        storage_options: dict | None = None,
    ) -> Any:
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

        use_polars = self._should_use_polars(force_type)

        try:
            if use_polars:
                return self._load_polars_with_fallback(path_str, storage_options)
            else:
                return self._load_pandas(path_str, storage_options)
        except Exception as e:
            logger.error(f"Failed to load file {path_str}: {e}")
            raise

    def _should_use_polars(self, force_type: str | None) -> bool:
        """Decide whether Polars should be used given availability and the requested engine."""
        if force_type == "pandas":
            return False
        if force_type == "polars" and not HAS_POLARS:
            logger.warning("Polars requested but not installed. Falling back to Pandas.")
            return False
        return HAS_POLARS

    def _load_polars_with_fallback(self, path_str: str, storage_options: dict | None) -> Any:
        """Load via Polars, falling back to Pandas (converted to Polars) on failure."""
        try:
            return self._load_polars(path_str, storage_options)
        except Exception as pl_err:
            logger.warning(f"Polars load failed for {path_str}: {pl_err}. Falling back to Pandas.")
            pdf = self._load_pandas(path_str, storage_options)
            return pl.from_pandas(pdf)

    def _load_polars(self, path: str, storage_options: dict | None = None) -> Any:
        """Load using Polars."""
        if path.endswith(".parquet"):
            return pl.read_parquet(path, storage_options=storage_options)
        elif path.endswith(".csv"):
            # Polars read_csv supports storage_options in recent versions
            return pl.read_csv(path, ignore_errors=True, storage_options=storage_options)
        elif path.endswith(".json"):
            return pl.read_json(
                path
            )  # JSON usually local, but if S3 needed, might need scan_ndjson
        else:
            # Fallback to pandas for other formats, then convert
            return pl.from_pandas(self._load_pandas(path, storage_options))

    def _load_pandas(self, path: str, storage_options: dict | None = None) -> pd.DataFrame:
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

    async def get_sample(self, path: str | Path, limit: int = 5) -> Any:
        """
        Get a sample of the data as a list of dictionaries.
        Efficiently reads only the first N rows.
        """
        path_str = str(path)

        if HAS_POLARS:
            sample = self._try_polars_lazy_sample(path_str, limit)
            if sample is not None:
                return sample

        # Fallback: Load full file (or eager load) and take head
        data = await self.load_file(path, force_type="pandas" if not HAS_POLARS else None)
        return self._sample_from_loaded_data(data, limit)

    def _try_polars_lazy_sample(self, path_str: str, limit: int) -> Any | None:
        """Attempt to lazily scan/read the first `limit` rows via Polars; None on failure."""
        try:
            if path_str.endswith(".parquet"):
                return _collect(pl.scan_parquet(path_str).limit(limit)).to_dicts()
            elif path_str.endswith(".csv"):
                return _collect(pl.scan_csv(path_str, ignore_errors=True).limit(limit)).to_dicts()
            elif path_str.endswith(".json"):
                # JSON scan is experimental/limited, use read
                return pl.read_json(path_str).head(limit).to_dicts()
        except Exception as e:
            logger.warning(
                f"Polars lazy scan failed for {path_str}: {e}. Falling back to eager load."
            )
        return None

    def _sample_from_loaded_data(self, data: Any, limit: int) -> Any:
        """Take the head of an already-loaded DataFrame/wrapper as a list of dicts."""
        # Handle SkyulfWrapper
        if hasattr(data, "to_pandas"):  # Wrapper or Polars
            # If it's Polars
            if HAS_POLARS and isinstance(data, pl.DataFrame):
                return data.head(limit).to_dicts()
            # If it's Wrapper or Pandas
            df = data.to_pandas()
            return df.head(limit).to_dict(orient="records")

        # It's likely a Pandas DataFrame
        return data.head(limit).to_dict(orient="records")

    async def save_artifact(self, data: Any, path: str | Path) -> None:
        """
        Save a DataFrame to disk (Parquet preferred).

        Args:
            data: The DataFrame (Polars, Pandas, or Skyulf wrapper).
            path: Destination path.
        """
        path_str = str(path)
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

        engine = get_engine(data)

        # If we have Polars and the data is compatible, use Polars for writing (faster)
        if HAS_POLARS and engine.__name__ == "PolarsEngine":
            self._save_polars_native(data, path_str)
        elif HAS_POLARS:
            self._save_via_polars_conversion(data, path_str)
        else:
            # No Polars, use Pandas
            self._save_pandas(data, path_str)

    def _save_polars_native(self, data: Any, path_str: str) -> None:
        """Write already-Polars(-wrapped) data directly via `write_parquet`."""
        if hasattr(data, "write_parquet"):
            data.write_parquet(path_str)
        elif hasattr(data, "_df") and hasattr(data._df, "write_parquet"):
            data._df.write_parquet(path_str)
        else:
            # Should not happen if engine is PolarsEngine
            logger.warning("PolarsEngine detected but write_parquet missing. Converting...")
            pl.from_pandas(data).write_parquet(path_str)

    def _save_via_polars_conversion(self, data: Any, path_str: str) -> None:
        """Convert non-Polars data (Pandas/Arrow-capable) to Polars for a fast write."""
        try:
            import pandas as pd

            # Zero-copy convert via Arrow if possible
            if hasattr(data, "to_arrow"):
                cast("pl.DataFrame", pl.from_arrow(data.to_arrow())).write_parquet(path_str)
            elif isinstance(data, pd.DataFrame):
                pl.from_pandas(data).write_parquet(path_str)
            else:
                self._save_pandas(data, path_str)
        except Exception as e:
            logger.warning(f"Polars write failed ({e}), falling back to Pandas.")
            self._save_pandas(data, path_str)

    def _save_pandas(self, data: Any, path: str):
        df = data.to_pandas() if hasattr(data, "to_pandas") else data

        if path.endswith(".parquet"):
            df.to_parquet(path)
        elif path.endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            # Unknown extension — default to Parquet
            if not path.endswith(".parquet"):
                path += ".parquet"
            df.to_parquet(path)
