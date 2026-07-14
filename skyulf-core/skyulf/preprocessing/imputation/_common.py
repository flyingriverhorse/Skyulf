"""Shared helpers for imputation nodes."""

import logging
from typing import Any

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

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
    import polars as pl

    if strategy == "constant":
        return None  # handled by caller
    if strategy == "mean":
        return lambda c: pl.col(c).mean()
    if strategy == "median":
        return lambda c: pl.col(c).median()
    if strategy == "most_frequent":
        # sklearn/scipy break ties by picking the smallest value; polars' .mode()
        # has no guaranteed tie-break order, so sort ascending before taking first.
        return lambda c: pl.col(c).mode().sort().first()
    raise ValueError(f"Unknown strategy: {strategy}")


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
    import polars as pl

    raw = X_pl.select([pl.col(c).null_count() for c in cols]).to_dict(as_series=False)
    counts = {c: int(raw[c][0]) for c in cols}
    return counts, sum(counts.values())


def _sklearn_transform_subset(X: Any, cols: list[str], imputer: Any, is_polars: bool) -> Any:
    """Run a fitted sklearn imputer over X[cols] and write back into a copy of X.

    Used by KNN + Iterative imputers; both share the exact same transform shape.
    Returns the transformed frame (Polars or Pandas, matching the input).
    """
    if is_polars:
        import polars as pl

        X_subset = X.select(cols)
        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        X_transformed = imputer.transform(X_np)
        if hasattr(X_transformed, "values"):
            X_transformed = X_transformed.values
        new_cols = [pl.Series(col, X_transformed[:, i]) for i, col in enumerate(cols)]
        return X.with_columns(new_cols)

    X_out = X.copy()
    X_subset = X_out[cols].copy()
    X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
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
