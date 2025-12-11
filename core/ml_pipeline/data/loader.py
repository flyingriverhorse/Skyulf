import pandas as pd
from typing import Optional
import os

class DataLoader:
    def load_full(self, source_path: str) -> pd.DataFrame:
        if not os.path.exists(source_path):
             raise FileNotFoundError(f"File not found: {source_path}")

        if source_path.endswith('.csv'):
            return pd.read_csv(source_path)
        elif source_path.endswith('.parquet'):
            return pd.read_parquet(source_path)
        else:
            raise ValueError(f"Unsupported file format: {source_path}")

    def load_sample(self, source_path: str, n: int = 1000) -> pd.DataFrame:
        if not os.path.exists(source_path):
             raise FileNotFoundError(f"File not found: {source_path}")

        if source_path.endswith('.csv'):
            return pd.read_csv(source_path, nrows=n)
        elif source_path.endswith('.parquet'):
            # Parquet doesn't support nrows easily without reading, 
            # but we can read and head. 
            df = pd.read_parquet(source_path)
            return df.head(n)
        else:
            raise ValueError(f"Unsupported file format: {source_path}")
