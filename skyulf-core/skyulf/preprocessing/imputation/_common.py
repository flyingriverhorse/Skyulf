"""Shared helpers for imputation nodes."""

import logging
from typing import Any

import pandas as pd
import polars as pl
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from ..._validation import raise_invalid_choice
from ...engines.sklearn_bridge import SklearnBridge
from ...utils import detect_numeric_columns, resolve_columns

logger = logging.getLogger(__name__)


def _resolve_simple_columns(X: Any, config: dict[str, Any], strategy: str) -> list[str]:
    """Pick the column-detection function based on strategy and resolve."""
    detect_func = (
        detect_numeric_columns if strategy in ("mean", "median") else (lambda d: list(d.columns))
    )
    return resolve_columns(X, config, detect_func)


def _polars_stat_for_strategy(strategy: str, fill_value: Any) -> Any:
    """Return the Polars expression-builder used to compute the per-column fill value."""
    if strategy == "constant":
        return None  # handled by caller
    if strategy == "mean":
        # polars keeps NaN as a value in mean(); sklearn treats NaN as
        # missing, so drop them first for parity.
        return lambda c: pl.col(c).drop_nans().mean()
    if strategy == "median":
        return lambda c: pl.col(c).drop_nans().median()
    if strategy == "most_frequent":
        # sklearn/scipy break ties by picking the smallest value; polars' .mode()
        # has no guaranteed tie-break order, so sort ascending before taking first.
        # drop_nans() is a no-op on non-float dtypes (this strategy may see strings).
        return lambda c: pl.col(c).drop_nulls().drop_nans().mode().sort().first()
    raise_invalid_choice(strategy, ("constant", "mean", "median", "most_frequent"), "strategy")


def _compute_polars_fill_values(
    X_pl: Any, cols: list[str], strategy: str, fill_value: Any
) -> dict[str, Any]:
    """Compute {col: fill_value} for Polars across all SimpleImputer strategies."""
    if strategy == "constant":
        default = fill_value if fill_value is not None else 0
        return dict.fromkeys(cols, default)

    expr_builder = _polars_stat_for_strategy(strategy, fill_value)
    stats = X_pl.select([expr_builder(c) for c in cols]).to_dict(as_series=False)
    return {c: stats[c][0] for c in cols}


def _polars_missing_counts(X_pl: Any, cols: list[str]) -> tuple[dict[str, int], int]:
    # pandas' isna() counts NaN as missing; polars' null_count() does not, so
    # float columns need an explicit NaN count to keep the artifact in parity.
    exprs = [
        pl.col(c).is_null().sum() + pl.col(c).is_nan().sum()
        if X_pl.schema[c].is_float()
        else pl.col(c).null_count()
        for c in cols
    ]
    raw = X_pl.select(exprs).to_dict(as_series=False)
    counts = {c: int(raw[c][0]) for c in cols}
    return counts, sum(counts.values())


def _sklearn_transform_subset(X: Any, cols: list[str], imputer: Any, is_polars: bool) -> Any:
    """Run a fitted sklearn imputer over X[cols] and write back into a copy of X.

    Used by KNN + Iterative imputers; both share the exact same transform shape.
    Returns the transformed frame (Polars or Pandas, matching the input).
    """
    if is_polars:
        X_subset = X.select(cols)
        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        X_transformed = imputer.transform(X_np)
        if hasattr(X_transformed, "to_numpy"):
            X_transformed = X_transformed.to_numpy()
        new_cols = [pl.Series(col, X_transformed[:, i]) for i, col in enumerate(cols)]
        return X.with_columns(new_cols)

    X_out = X.copy()
    X_subset = X_out[cols].copy()
    # Nullable extension dtypes (Int64...) hand pd.NA to sklearn and refuse
    # float results on write-back; upcast them to float64 like the Polars
    # branch does natively (F-10).
    ext_cols = [c for c in cols if isinstance(X_subset[c].dtype, pd.api.extensions.ExtensionDtype)]
    if ext_cols:
        X_subset[ext_cols] = X_subset[ext_cols].astype("float64")
        X_out[ext_cols] = X_out[ext_cols].astype("float64")
    X_input = X_subset.to_numpy() if hasattr(X_subset, "to_numpy") else X_subset
    X_transformed = imputer.transform(X_input)
    X_out[cols] = X_transformed
    return X_out


def _build_iterative_estimator(name: str) -> Any:
    """Map the public estimator alias to a concrete sklearn regressor."""
    if name == "DecisionTree":
        return DecisionTreeRegressor(max_features="sqrt", random_state=0)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=10, random_state=0)
    if name == "KNeighbors":
        return KNeighborsRegressor(n_neighbors=5)
    return BayesianRidge()
