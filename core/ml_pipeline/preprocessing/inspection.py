from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns

class DatasetProfileCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Generate a lightweight dataset profile
        # We capture shape, types, and basic numeric stats for pipeline metadata
        
        profile = {}
        
        # Shape
        profile['rows'] = len(df)
        profile['columns'] = len(df.columns)
        
        # Column types
        profile['dtypes'] = df.dtypes.astype(str).to_dict()
        
        # Missing values
        profile['missing'] = df.isna().sum().to_dict()
        
        # Numeric stats
        numeric_cols = detect_numeric_columns(df)
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            profile['numeric_stats'] = desc
            
        return {
            'type': 'dataset_profile',
            'profile': profile
        }

class DatasetProfileApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        # Inspection nodes do not modify data
        return df

class DataSnapshotCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Take a snapshot of the first N rows
        n = config.get('n_rows', 5)
        snapshot = df.head(n).to_dict(orient='records')
        
        return {
            'type': 'data_snapshot',
            'snapshot': snapshot
        }

class DataSnapshotApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return df
