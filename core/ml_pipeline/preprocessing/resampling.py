from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours

from .base import BaseCalculator, BaseApplier
from ..utils import unpack_pipeline_input, pack_pipeline_output

logger = logging.getLogger(__name__)

# --- Oversampling (SMOTE variants) ---
class OversamplingCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'method': 'smote', 'target_column': 'target', 'sampling_strategy': 'auto', ...}
        return {
            'type': 'oversampling',
            'method': config.get('method', 'smote'),
            'target_column': config.get('target_column'),
            'sampling_strategy': config.get('sampling_strategy', 'auto'),
            'random_state': config.get('random_state', 42),
            'k_neighbors': config.get('k_neighbors', 5),
            'n_jobs': config.get('n_jobs', -1)
        }

class OversamplingApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        target_col = params.get('target_column')
        
        # Resampling requires y. If not provided in tuple, try to extract from dataframe using target_column
        if y is None:
            if target_col and target_col in X.columns:
                y = X[target_col]
                X = X.drop(columns=[target_col])
            else:
                # Cannot resample without target
                return pack_pipeline_output(X, y, is_tuple)
        
        method = params.get('method', 'smote')
        strategy = params.get('sampling_strategy', 'auto')
        random_state = params.get('random_state', 42)
        k_neighbors = params.get('k_neighbors', 5)
        n_jobs = params.get('n_jobs', -1)
        
        sampler = None
        if method == 'smote':
            sampler = SMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors)
        elif method == 'adasyn':
            sampler = ADASYN(sampling_strategy=strategy, random_state=random_state, n_neighbors=k_neighbors, n_jobs=n_jobs)
        elif method == 'borderline_smote':
            sampler = BorderlineSMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors, n_jobs=n_jobs)
        elif method == 'svm_smote':
            sampler = SVMSMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors, n_jobs=n_jobs)
        elif method == 'kmeans_smote':
            sampler = KMeansSMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors, n_jobs=n_jobs)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(sampling_strategy=strategy, random_state=random_state, n_jobs=n_jobs)
            
        if not sampler:
            return pack_pipeline_output(X, y, is_tuple)
            
        try:
            X_res, y_res = sampler.fit_resample(X, y)
            
            # Ensure we return DataFrame/Series with correct names/columns
            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=X.columns)
            if not isinstance(y_res, pd.Series):
                y_res = pd.Series(y_res, name=y.name if y is not None else target_col)
                
            return pack_pipeline_output(X_res, y_res, is_tuple)

        except Exception as e:
            logger.error(f"Oversampling failed: {e}")
            return pack_pipeline_output(X, y, is_tuple)

# --- Undersampling ---
class UndersamplingCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'undersampling',
            'method': config.get('method', 'random_under_sampling'),
            'target_column': config.get('target_column'),
            'sampling_strategy': config.get('sampling_strategy', 'auto'),
            'random_state': config.get('random_state', 42),
            'replacement': config.get('replacement', False),
            'version': config.get('version', 1), # For NearMiss
            'n_neighbors': config.get('n_neighbors', 3), # For EditedNearestNeighbours
            'kind_sel': config.get('kind_sel', 'all'), # For EditedNearestNeighbours
            'n_jobs': config.get('n_jobs', -1)
        }

class UndersamplingApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        target_col = params.get('target_column')
        
        # Resampling requires y. If not provided in tuple, try to extract from dataframe using target_column
        if y is None:
            if target_col and target_col in X.columns:
                y = X[target_col]
                X = X.drop(columns=[target_col])
            else:
                # Cannot resample without target
                return pack_pipeline_output(X, y, is_tuple)
        
        method = params.get('method', 'random_under_sampling')
        strategy = params.get('sampling_strategy', 'auto')
        random_state = params.get('random_state', 42)
        replacement = params.get('replacement', False)
        n_jobs = params.get('n_jobs', -1)
        
        sampler = None
        if method == 'random_under_sampling':
            sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state, replacement=replacement)
        elif method == 'nearmiss':
            version = params.get('version', 1)
            sampler = NearMiss(sampling_strategy=strategy, version=version, n_jobs=n_jobs)
        elif method == 'tomek_links':
            sampler = TomekLinks(sampling_strategy=strategy, n_jobs=n_jobs)
        elif method == 'edited_nearest_neighbours':
            n_neighbors = params.get('n_neighbors', 3)
            kind_sel = params.get('kind_sel', 'all')
            sampler = EditedNearestNeighbours(sampling_strategy=strategy, n_neighbors=n_neighbors, kind_sel=kind_sel, n_jobs=n_jobs)
            
        if not sampler:
            return pack_pipeline_output(X, y, is_tuple)
            
        try:
            X_res, y_res = sampler.fit_resample(X, y)
            
            # Ensure we return DataFrame/Series with correct names/columns
            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=X.columns)
            if not isinstance(y_res, pd.Series):
                y_res = pd.Series(y_res, name=y.name if y is not None else target_col)
                
            return pack_pipeline_output(X_res, y_res, is_tuple)

        except Exception as e:
            logger.error(f"Undersampling failed: {e}")
            return pack_pipeline_output(X, y, is_tuple)
