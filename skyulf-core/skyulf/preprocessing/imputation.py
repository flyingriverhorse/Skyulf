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
from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..engines import SkyulfDataFrame, get_engine
from ..engines.sklearn_bridge import SklearnBridge

logger = logging.getLogger(__name__)

# --- Simple Imputer (Mean, Median, Mode) ---


class SimpleImputerApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        fill_values = params.get("fill_values", {})

        if not cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X

            exprs = []
            # Handle existing columns
            for col in X_pl.columns:
                if col in cols and col in fill_values:
                    val = fill_values[col]
                    # fill_null must be applied to the column expression
                    exprs.append(pl.col(col).fill_null(val).alias(col))
                else:
                    exprs.append(pl.col(col))

            # Handle missing columns (restore them)
            # This logic assumes "restore" means adding them if they are completely missing from input X but were present during fit?
            # Or is it just adding constant value cols?
            # The original code iterated cols and checked if col not in X.columns.
            
            for col in cols:
                if col not in X_pl.columns and col in fill_values:
                    val = fill_values[col]
                    exprs.append(pl.lit(val).alias(col))

            X_out = X_pl.select(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
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


@NodeRegistry.register("SimpleImputer", SimpleImputerApplier)
@node_meta(
    id="SimpleImputer",
    name="Simple Imputer",
    category="Preprocessing",
    description="Imputes missing values using mean, median, or constant.",
    params={"strategy": "mean", "fill_value": None, "columns": []}
)
class SimpleImputerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

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
            else (lambda d: d.columns.tolist())  # Explicit type ignored for lambda can be tricky, but logic holds
        )

        cols = resolve_columns(X, config, detect_func)

        if not cols:
            return {}

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X

            fill_values = {}

            if strategy == "constant":
                for col in cols:
                    fill_values[col] = fill_value if fill_value is not None else 0

            elif strategy == "mean":
                stats = X_pl.select([pl.col(c).mean() for c in cols]).to_dict(
                    as_series=False
                )
                for col in cols:
                    fill_values[col] = stats[col][0]

            elif strategy == "median":
                stats = X_pl.select([pl.col(c).median() for c in cols]).to_dict(
                    as_series=False
                )
                for col in cols:
                    fill_values[col] = stats[col][0]

            elif strategy == "most_frequent":
                # Mode in Polars returns a list. We take the first one.
                stats = X_pl.select([pl.col(c).mode().first() for c in cols]).to_dict(
                    as_series=False
                )
                for col in cols:
                    fill_values[col] = stats[col][0]

            # Calculate missing counts
            missing_counts = X_pl.select([pl.col(c).null_count() for c in cols]).to_dict(
                as_series=False
            )
            missing_counts_dict = {c: missing_counts[c][0] for c in cols}
            total_missing = sum(missing_counts_dict.values())

            return {
                "type": "simple_imputer",
                "strategy": strategy,
                "fill_values": fill_values,
                "columns": cols,
                "missing_counts": missing_counts_dict,
                "total_missing": total_missing,
            }

        # Pandas Path
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


# --- KNN Imputer ---


class KNNImputerApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        imputer = params.get("imputer_object")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not imputer:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X

            try:
                X_subset = X_pl.select(cols)
                X_np, _ = SklearnBridge.to_sklearn(X_subset)
                X_transformed = imputer.transform(X_np)

                # Update columns
                new_cols = [
                    pl.Series(col, X_transformed[:, i]) for i, col in enumerate(cols)
                ]
                X_out = X_pl.with_columns(new_cols)
                return pack_pipeline_output(X_out, y, is_tuple)

            except Exception as e:
                logger.error(f"KNN Imputation failed: {e}")
                return pack_pipeline_output(X, y, is_tuple)

        # Pandas Path
        X_out = X.copy()

        # KNN Imputer transforms the matrix
        # We need to ensure column order matches fit
        # If some columns are missing in transform, we can't easily use KNN
        # For now, we assume all columns are present or we skip

        try:
            # Ensure all columns exist, fill missing with NaN to match shape
            X_subset = X_out[cols].copy()

            # Transform
            # Fix for "X has feature names..." warning
            if hasattr(X_subset, "values"):
                X_input = X_subset.values
            else:
                X_input = X_subset
                
            X_transformed = imputer.transform(X_input)

            # Update DataFrame
            X_out[cols] = X_transformed

        except Exception as e:
            logger.error(f"KNN Imputation failed: {e}")
            # Fallback? Or raise?
            pass

        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("KNNImputer", KNNImputerApplier)
@node_meta(
    id="KNNImputer",
    name="KNN Imputer",
    category="Preprocessing",
    description="Impute missing values using k-Nearest Neighbors.",
    params={"n_neighbors": 5, "weights": "uniform", "columns": []}
)
class KNNImputerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

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
        
        # Use Bridge for fitting
        if engine.name == "polars":
            # Polars Path
            X_pl: Any = X
            X_subset = X_pl.select(cols)
        else:
            # Pandas Path
            X_subset = X[cols]

        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        
        imputer.fit(X_np)

        return {
            "type": "knn_imputer",
            "imputer_object": imputer,  # Not JSON serializable
            "columns": cols,
            "n_neighbors": n_neighbors,
            "weights": weights,
        }


# --- Iterative Imputer (MICE) ---


class IterativeImputerApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        imputer = params.get("imputer_object")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not imputer:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X

            try:
                X_subset = X_pl.select(cols)
                X_np, _ = SklearnBridge.to_sklearn(X_subset)
                X_transformed = imputer.transform(X_np)

                new_cols = [
                    pl.Series(col, X_transformed[:, i]) for i, col in enumerate(cols)
                ]
                X_out = X_pl.with_columns(new_cols)
                return pack_pipeline_output(X_out, y, is_tuple)
            except Exception as e:
                logger.error(f"Iterative Imputation failed: {e}")
                return pack_pipeline_output(X, y, is_tuple)

        # Pandas Path
        X_out = X.copy()

        try:
            X_subset = X_out[cols].copy()
            
            # Fix for "X has feature names..." warning
            if hasattr(X_subset, "values"):
                X_input = X_subset.values
            else:
                X_input = X_subset
                
            X_transformed = imputer.transform(X_input)
            X_out[cols] = X_transformed
        except Exception as e:
            logger.error(f"Iterative Imputation failed: {e}")
            pass

        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("IterativeImputer", IterativeImputerApplier)
@node_meta(
    id="IterativeImputer",
    name="Iterative Imputer (MICE)",
    category="Preprocessing",
    description="Multivariate imputation using chained equations.",
    params={"max_iter": 10, "random_state": 0, "estimator": "bayesian_ridge", "columns": []}
)
class IterativeImputerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

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
        
        # Use Bridge for fitting
        if engine.name == "polars":
            # Polars Path
            X_pl: Any = X
            X_subset = X_pl.select(cols)
        else:
            # Pandas Path
            X_subset = X[cols]

        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        
        imputer.fit(X_np)

        return {
            "type": "iterative_imputer",
            "imputer_object": imputer,  # Not JSON serializable
            "columns": cols,
            "estimator": estimator_name,
        }
