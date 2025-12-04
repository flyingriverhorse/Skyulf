from typing import Dict, Any, List, Optional, Union
import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns, resolve_columns

logger = logging.getLogger(__name__)

# --- Simple Imputer (Mean, Median, Mode) ---
class SimpleImputerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'strategy': 'mean' | 'median' | 'most_frequent' | 'constant', 'columns': [...], 'fill_value': ...}
        strategy = config.get('strategy', 'mean')
        # Map 'mode' to 'most_frequent' for sklearn compatibility
        if strategy == 'mode':
            strategy = 'most_frequent'
            
        fill_value = config.get('fill_value', None)
        
        # Determine detection function based on strategy
        detect_func = detect_numeric_columns if strategy in ['mean', 'median'] else (lambda d: d.columns.tolist())
        
        cols = resolve_columns(df, config, detect_func)
        
        if not cols:
            return {}

        # Sklearn SimpleImputer
        # Note: SimpleImputer expects 2D array
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        
        # Handle potential errors with non-numeric data for mean/median
        if strategy in ['mean', 'median']:
            # Filter for numeric columns only to be safe (double check)
            numeric_cols = detect_numeric_columns(df)
            cols = [c for c in cols if c in numeric_cols]
            if not cols:
                return {}
        
        imputer.fit(df[cols])
        
        # Extract statistics to make them JSON serializable
        statistics = imputer.statistics_.tolist()
        
        # Map columns to their fill values
        fill_values = dict(zip(cols, statistics))
        
        return {
            'type': 'simple_imputer',
            'strategy': strategy,
            'fill_values': fill_values,
            'columns': cols
        }

class SimpleImputerApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        fill_values = params.get('fill_values', {})
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        df_transformed = df.copy()
        
        for col in valid_cols:
            if col in fill_values:
                val = fill_values[col]
                # Handle NaN fill value if it was stored (though unlikely for simple imputer results)
                if val is not None:
                    df_transformed[col] = df_transformed[col].fillna(val)
                    
        return df_transformed

# --- KNN Imputer ---
class KNNImputerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'n_neighbors': 5, 'weights': 'uniform', 'columns': [...]}
        n_neighbors = config.get('n_neighbors', 5)
        weights = config.get('weights', 'uniform')
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputer.fit(df[cols])
        
        return {
            'type': 'knn_imputer',
            'columns': cols,
            'imputer_object': imputer 
        }

class KNNImputerApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        imputer = params.get('imputer_object')
        
        if not imputer or not cols:
            return df
            
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        X = df[valid_cols]
        
        try:
            X_imputed = imputer.transform(X)
            df_transformed = df.copy()
            df_transformed[valid_cols] = X_imputed
            return df_transformed
        except Exception as e:
            logger.error(f"KNN Imputation failed: {e}")
            return df

# --- Iterative Imputer (MICE) ---
class IterativeImputerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'max_iter': 10, 'random_state': 0, 'estimator': 'bayesian_ridge', 'columns': [...]}
        max_iter = config.get('max_iter', 10)
        random_state = config.get('random_state', 0)
        estimator_name = config.get('estimator', 'bayesian_ridge')
        
        estimator = None
        if estimator_name == 'bayesian_ridge':
            estimator = BayesianRidge()
        elif estimator_name == 'decision_tree':
            estimator = DecisionTreeRegressor(max_features='sqrt', random_state=random_state)
        elif estimator_name == 'extra_trees':
            estimator = ExtraTreesRegressor(n_estimators=10, random_state=random_state)
        elif estimator_name == 'knn':
            estimator = KNeighborsRegressor(n_neighbors=5)
            
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state
        )
        imputer.fit(df[cols])
        
        return {
            'type': 'iterative_imputer',
            'columns': cols,
            'imputer_object': imputer
        }

class IterativeImputerApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        imputer = params.get('imputer_object')
        
        if not imputer or not cols:
            return df
            
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        X = df[valid_cols]
        
        try:
            X_imputed = imputer.transform(X)
            df_transformed = df.copy()
            df_transformed[valid_cols] = X_imputed
            return df_transformed
        except Exception as e:
            logger.error(f"Iterative Imputation failed: {e}")
            return df
