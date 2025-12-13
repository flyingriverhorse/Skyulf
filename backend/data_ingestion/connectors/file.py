import os
import polars as pl
from typing import Dict, Optional
from .base import BaseConnector

class LocalFileConnector(BaseConnector):
    """
    Connector for local files (CSV, Excel, Parquet, JSON).
    """
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.parquet', '.json'}

    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs
        self._df: Optional[pl.DataFrame] = None

    async def connect(self) -> bool:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        return True

    async def _load_data(self):
        if self._df is not None:
            return

        ext = os.path.splitext(self.file_path)[1].lower()
        
        try:
            if ext == '.csv':
                self._df = pl.read_csv(self.file_path, **self.kwargs)
            elif ext in ['.xlsx', '.xls']:
                self._df = pl.read_excel(self.file_path, **self.kwargs)
            elif ext == '.parquet':
                self._df = pl.read_parquet(self.file_path, **self.kwargs)
            elif ext == '.json':
                self._df = pl.read_json(self.file_path, **self.kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to read file {self.file_path}: {str(e)}")

    async def get_schema(self) -> Dict[str, str]:
        await self._load_data()
        if self._df is None:
            raise RuntimeError("Data not loaded")
            
        return {col: str(dtype) for col, dtype in self._df.schema.items()}

    async def fetch_data(self, query: Optional[str] = None, limit: Optional[int] = None) -> pl.DataFrame:
        await self._load_data()
        if self._df is None:
            raise RuntimeError("Data not loaded")
            
        df = self._df
        
        if limit:
            df = df.head(limit)
            
        return df

    async def validate(self) -> bool:
        return await self.connect()
