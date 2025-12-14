from typing import Dict, Any, Union, Tuple
import pandas as pd
from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns


class DatasetProfileCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[Any, ...]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Generate a lightweight dataset profile
        # We capture shape, types, and basic numeric stats for pipeline metadata
        
        if isinstance(df, tuple):
             # Handle tuple if needed, or just ignore/raise. 
             # Since this is inspection, maybe we inspect the first element?
             # Or just return empty profile?
             # For now, let's assume it's a DataFrame for the logic below, 
             # but we must accept tuple to satisfy the interface.
             # If we get a tuple, we can try to unpack it or just fail gracefully.
             # Given the existing code assumes df is DataFrame, let's just cast or check.
             if len(df) > 0 and isinstance(df[0], pd.DataFrame):
                 df = df[0]
             else:
                 return {'type': 'dataset_profile', 'profile': {}}

        profile: Dict[str, Any] = {}

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
    def apply(self, df: Union[pd.DataFrame, Tuple[Any, ...]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[Any, ...]]:
        # Inspection nodes do not modify data
        return df


class DataSnapshotCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[Any, ...]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Take a snapshot of the first N rows
        if isinstance(df, tuple):
             if len(df) > 0 and isinstance(df[0], pd.DataFrame):
                 df = df[0]
             else:
                 return {'type': 'data_snapshot', 'snapshot': []}

        n = config.get('n_rows', 5)
        snapshot = df.head(n).to_dict(orient='records')

        return {
            'type': 'data_snapshot',
            'snapshot': snapshot
        }


class DataSnapshotApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[Any, ...]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[Any, ...]]:
        return df
