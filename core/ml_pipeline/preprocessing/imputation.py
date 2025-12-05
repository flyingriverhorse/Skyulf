from typing import Dict, Any, List, Optional, Union, Tuple
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
from ..utils import detect_numeric_columns, resolve_columns, unpack_pipeline_input, pack_pipeline_output

logger = logging.getLogger(__name__)

# --- Simple Imputer (Mean, Median, Mode) ---
class SimpleImputerCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'strategy': 'mean' | 'median' | 'most_frequent' | 'constant', 'columns': [...], 'fill_value': ...}
        strategy = config.get('strategy', 'mean')
        # Map 'mode' to 'most_frequent' for sklearn compatibility
        if strategy == 'mode':
            strategy = 'most_frequent'
            
        fill_value = config.get('fill_value', None)
        
        # Determine detection function based on strategy
        detect_func = detect_numeric_columns if strategy in ['mean', 'median'] else (lambda d: d.columns.tolist())
        
        cols = resolve_columns(X, config, detect_func)
        
        if not cols:
            return {}

        # Sklearn SimpleImputer
        # Note: SimpleImputer expects 2D array
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        
        # Handle potential errors with non-numeric data for mean/median
        if strategy in ['mean', 'median']:
            # Filter for numeric columns only to be safe (double check)
            numeric_cols = detect_numeric_columns(X)
            cols = [c for c in cols if c in numeric_cols]
            if not cols:
                return {}
        
        imputer.fit(X[cols])
        
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        fill_values = params.get('fill_values', {})
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        for col in valid_cols:
            if col in fill_values:
                val = fill_values[col]
                if val is not None:
                    X_out[col] = X_out[col].fillna(val)
        return pack_pipeline_output(X_out, y, is_tuple)

# --- KNN Imputer ---
class KNNImputerCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'n_neighbors': 5, 'weights': 'uniform', 'columns': [...]}
        n_neighbors = config.get('n_neighbors', 5)
        weights = config.get('weights', 'uniform')
        
        cols = resolve_columns(X, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputer.fit(X[cols])
        
        return {
            'type': 'knn_imputer',
            'columns': cols,
            'imputer_object': imputer 
        }

class KNNImputerApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        imputer = params.get('imputer_object')
        
        if not imputer or not cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
        
        X_sub = X[valid_cols]
        try:
            X_imputed = imputer.transform(X_sub)
            X_out = X.copy()
            X_out[valid_cols] = X_imputed
            return pack_pipeline_output(X_out, y, is_tuple)
        except Exception as e:
            logger.error(f"KNN Imputation failed: {e}")
            return pack_pipeline_output(X, y, is_tuple)

# --- Iterative Imputer (MICE) ---
class IterativeImputerCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

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
            
        cols = resolve_columns(X, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state
        )
        imputer.fit(X[cols])
        
        return {
            'type': 'iterative_imputer',
            'columns': cols,
            'imputer_object': imputer
        }

class IterativeImputerApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        imputer = params.get('imputer_object')
        
        if not imputer or not cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
        
        X_sub = X[valid_cols]
        try:
            X_imputed = imputer.transform(X_sub)
            X_out = X.copy()
            X_out[valid_cols] = X_imputed
            return pack_pipeline_output(X_out, y, is_tuple)
        except Exception as e:
            logger.error(f"Iterative Imputation failed: {e}")
            return pack_pipeline_output(X, y, is_tuple)
