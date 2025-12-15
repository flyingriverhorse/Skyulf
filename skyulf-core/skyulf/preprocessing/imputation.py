import logging
from typing import Any, Dict, Tuple, Union

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from ..utils import (
    detect_numeric_columns,
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
)
from .base import BaseApplier, BaseCalculator

logger = logging.getLogger(__name__)

# --- Simple Imputer (Mean, Median, Mode) ---


class SimpleImputerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'strategy': 'mean' | 'median' | 'most_frequent' | 'constant', 'columns': [...], 'fill_value': ...}
        strategy = config.get("strategy", "mean")
        # Map 'mode' to 'most_frequent' for sklearn compatibility
        if strategy == "mode":
            strategy = "most_frequent"

        fill_value = config.get("fill_value", None)

        # Determine detection function based on strategy
        detect_func = (
            detect_numeric_columns
            if strategy in ["mean", "median"]
            else (lambda d: d.columns.tolist())
        )

        cols = resolve_columns(X, config, detect_func)

        if not cols:
            return {}

        # Sklearn SimpleImputer
        # Note: SimpleImputer expects 2D array
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

        # Handle potential errors with non-numeric data for mean/median
        if strategy in ["mean", "median"]:
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

        # Calculate missing counts for feedback
        missing_counts = X[cols].isnull().sum().to_dict()
        total_missing = int(sum(missing_counts.values()))

        return {
            "type": "simple_imputer",
            "strategy": strategy,
            "fill_values": fill_values,
            "columns": cols,
            "missing_counts": missing_counts,
            "total_missing": total_missing,
        }


class SimpleImputerApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)

        cols = params.get("columns", [])
        fill_values = params.get("fill_values", {})

        if not cols:
            return pack_pipeline_output(X, y, is_tuple)

        X_out = X.copy()

        # Iterate over ALL expected columns, not just valid ones
        for col in cols:
            val = fill_values.get(col)
            if val is None:
                continue

            if col not in X_out.columns:
                # Restore missing column with fill value
                X_out[col] = val
            else:
                # Fill existing NaNs
                X_out[col] = X_out[col].fillna(val)

        return pack_pipeline_output(X_out, y, is_tuple)


# --- KNN Imputer ---


class KNNImputerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'n_neighbors': 5, 'weights': 'uniform'|'distance', 'columns': [...]}
        n_neighbors = config.get("n_neighbors", 5)
        weights = config.get("weights", "uniform")

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        # KNN Imputer is heavy, we need to store the whole training set (or a sample)
        # For now, we store the fitted imputer object directly.
        # WARNING: This is not JSON serializable. We need pickle for this.

        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputer.fit(X[cols])

        return {
            "type": "knn_imputer",
            "imputer_object": imputer,  # Not JSON serializable
            "columns": cols,
            "n_neighbors": n_neighbors,
            "weights": weights,
        }


class KNNImputerApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)

        cols = params.get("columns", [])
        imputer = params.get("imputer_object")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not imputer:
            return pack_pipeline_output(X, y, is_tuple)

        X_out = X.copy()

        # KNN Imputer transforms the matrix
        # We need to ensure column order matches fit
        # If some columns are missing in transform, we can't easily use KNN
        # For now, we assume all columns are present or we skip

        try:
            # Ensure all columns exist, fill missing with NaN to match shape
            X_subset = X_out[cols].copy()

            # Transform
            X_transformed = imputer.transform(X_subset)

            # Update DataFrame
            X_out[cols] = X_transformed

        except Exception as e:
            logger.error(f"KNN Imputation failed: {e}")
            # Fallback? Or raise?
            pass

        return pack_pipeline_output(X_out, y, is_tuple)


# --- Iterative Imputer (MICE) ---


class IterativeImputerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'max_iter': 10, 'estimator': 'BayesianRidge'|'DecisionTree'|'ExtraTrees'|'KNeighbors',
        #          'columns': [...]}
        max_iter = config.get("max_iter", 10)
        estimator_name = config.get("estimator", "BayesianRidge")

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        estimator = None
        if estimator_name == "DecisionTree":
            estimator = DecisionTreeRegressor(max_features="sqrt", random_state=0)
        elif estimator_name == "ExtraTrees":
            estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
        elif estimator_name == "KNeighbors":
            estimator = KNeighborsRegressor(n_neighbors=5)
        else:
            estimator = BayesianRidge()

        imputer = IterativeImputer(
            estimator=estimator, max_iter=max_iter, random_state=0
        )
        imputer.fit(X[cols])

        return {
            "type": "iterative_imputer",
            "imputer_object": imputer,  # Not JSON serializable
            "columns": cols,
            "estimator": estimator_name,
        }


class IterativeImputerApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)

        cols = params.get("columns", [])
        imputer = params.get("imputer_object")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not imputer:
            return pack_pipeline_output(X, y, is_tuple)

        X_out = X.copy()

        try:
            X_subset = X_out[cols].copy()
            X_transformed = imputer.transform(X_subset)
            X_out[cols] = X_transformed
        except Exception as e:
            logger.error(f"Iterative Imputation failed: {e}")
            pass

        return pack_pipeline_output(X_out, y, is_tuple)
