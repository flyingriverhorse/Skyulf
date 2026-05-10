"""Imputation nodes (Simple / KNN / Iterative).

Each Applier/Calculator dispatches on engine via ``apply_dual_engine`` /
``fit_dual_engine`` (see ``dispatcher.py``). Per-engine logic lives in
small ``_apply_polars`` / ``_apply_pandas`` / ``_fit_polars`` /
``_fit_pandas`` static helpers so the public ``apply`` / ``fit`` methods
stay at CCN 1 (Codacy ``lizard_ccn-medium``).
"""

import logging
from typing import Any, Dict, List, Tuple, cast

from sklearn.ensemble import ExtraTreesRegressor

# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from ..utils import (
    detect_numeric_columns,
    resolve_columns,
    user_picked_no_columns,
)
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine, fit_dual_engine
from ._artifacts import IterativeImputerArtifact, KNNImputerArtifact, SimpleImputerArtifact
from ._helpers import resolve_valid_columns
from ._schema import SkyulfSchema
from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..engines.sklearn_bridge import SklearnBridge

logger = logging.getLogger(__name__)


# =============================================================================
# Shared helpers
# =============================================================================


def _resolve_simple_columns(X: Any, config: Dict[str, Any], strategy: str) -> List[str]:
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
        return lambda c: pl.col(c).mode().first()
    raise ValueError(f"Unknown strategy: {strategy}")


def _compute_polars_fill_values(
    X_pl: Any, cols: List[str], strategy: str, fill_value: Any
) -> Dict[str, Any]:
    """Compute {col: fill_value} for Polars across all SimpleImputer strategies."""
    if strategy == "constant":
        default = fill_value if fill_value is not None else 0
        return {c: default for c in cols}

    expr_builder = _polars_stat_for_strategy(strategy, fill_value)
    stats = X_pl.select([expr_builder(c) for c in cols]).to_dict(as_series=False)
    return {c: stats[c][0] for c in cols}


def _polars_missing_counts(X_pl: Any, cols: List[str]) -> Tuple[Dict[str, int], int]:
    import polars as pl

    raw = X_pl.select([pl.col(c).null_count() for c in cols]).to_dict(as_series=False)
    counts = {c: int(raw[c][0]) for c in cols}
    return counts, sum(counts.values())


def _sklearn_transform_subset(X: Any, cols: List[str], imputer: Any, is_polars: bool) -> Any:
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


# =============================================================================
# Simple Imputer
# =============================================================================


class SimpleImputerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        fill_values = params.get("fill_values", {})
        if not cols:
            return X, _y

        exprs: List[Any] = []
        for col in X.columns:
            if col in cols and col in fill_values:
                exprs.append(pl.col(col).fill_null(fill_values[col]).alias(col))
            else:
                exprs.append(pl.col(col))

        # Restore columns that were present at fit time but missing in input X.
        for col in cols:
            if col not in X.columns and col in fill_values:
                exprs.append(pl.lit(fill_values[col]).alias(col))

        return X.select(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        fill_values = params.get("fill_values", {})
        if not cols:
            return X, _y

        X_out = X.copy()
        for col in cols:
            val = fill_values.get(col)
            if val is None:
                continue
            if col not in X_out.columns:
                X_out[col] = val
            else:
                X_out[col] = X_out[col].fillna(val)
        return X_out, _y


@NodeRegistry.register("SimpleImputer", SimpleImputerApplier)
@node_meta(
    id="SimpleImputer",
    name="Simple Imputer",
    category="Preprocessing",
    description="Imputes missing values using mean, median, or constant.",
    params={"strategy": "mean", "fill_value": None, "columns": []},
)
class SimpleImputerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Imputers fill NaNs in place; column set and order are preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> SimpleImputerArtifact:
        if user_picked_no_columns(config):
            return {}

        strategy = config.get("strategy", "mean")
        if strategy == "mode":
            strategy = "most_frequent"
        fill_value = config.get("fill_value", None)

        cols = _resolve_simple_columns(X, config, strategy)
        if not cols:
            return {}

        # Stash resolved-once values into params so dispatched fits don't redo work.
        merged = dict(config)
        merged["_resolved_strategy"] = strategy
        merged["_resolved_cols"] = cols
        merged["_resolved_fill_value"] = fill_value

        return cast(
            SimpleImputerArtifact,
            fit_dual_engine(X, merged, self._fit_polars, self._fit_pandas),
        )

    @staticmethod
    def _fit_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        cols: List[str] = params["_resolved_cols"]
        strategy: str = params["_resolved_strategy"]
        fill_value = params["_resolved_fill_value"]

        fill_values = _compute_polars_fill_values(X, cols, strategy, fill_value)
        missing_counts, total_missing = _polars_missing_counts(X, cols)

        return {
            "type": "simple_imputer",
            "strategy": strategy,
            "fill_values": fill_values,
            "columns": cols,
            "missing_counts": missing_counts,
            "total_missing": total_missing,
        }

    @staticmethod
    def _fit_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        cols: List[str] = params["_resolved_cols"]
        strategy: str = params["_resolved_strategy"]
        fill_value = params["_resolved_fill_value"]

        # Mean/median: extra safety filter to numeric columns only.
        if strategy in ("mean", "median"):
            numeric = set(detect_numeric_columns(X))
            cols = [c for c in cols if c in numeric]
            if not cols:
                return {}

        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        imputer.fit(X[cols])

        statistics = imputer.statistics_.tolist()
        fill_values = dict(zip(cols, statistics))
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


# =============================================================================
# KNN Imputer
# =============================================================================


class KNNImputerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        imputer = params.get("imputer_object")
        if not resolve_valid_columns(X, cols) or not imputer:
            return X, _y
        try:
            return _sklearn_transform_subset(X, cols, imputer, is_polars=True), _y
        except Exception as e:
            logger.error(f"KNN Imputation failed: {e}")
            return X, _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        imputer = params.get("imputer_object")
        if not resolve_valid_columns(X, cols) or not imputer:
            return X, _y
        try:
            return _sklearn_transform_subset(X, cols, imputer, is_polars=False), _y
        except Exception as e:
            logger.error(f"KNN Imputation failed: {e}")
            return X, _y


@NodeRegistry.register("KNNImputer", KNNImputerApplier)
@node_meta(
    id="KNNImputer",
    name="KNN Imputer",
    category="Preprocessing",
    description="Impute missing values using k-Nearest Neighbors.",
    params={"n_neighbors": 5, "weights": "uniform", "columns": []},
)
class KNNImputerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # KNN imputation fills NaNs in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> KNNImputerArtifact:
        if user_picked_no_columns(config):
            return {}

        n_neighbors = config.get("n_neighbors", 5)
        weights = config.get("weights", "uniform")
        cols = resolve_columns(X, config, detect_numeric_columns)
        if not cols:
            return {}

        # KNN/Iterative imputers always fit through the sklearn bridge which
        # operates on numpy — engine choice doesn't affect the fit math, just
        # which subset selector we use.
        X_subset = X.select(cols) if hasattr(X, "select") and not hasattr(X, "loc") else X[cols]
        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputer.fit(X_np)

        return {
            "type": "knn_imputer",
            "imputer_object": imputer,  # Not JSON serializable
            "columns": cols,
            "n_neighbors": n_neighbors,
            "weights": weights,
        }


# =============================================================================
# Iterative Imputer (MICE)
# =============================================================================


class IterativeImputerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        imputer = params.get("imputer_object")
        if not resolve_valid_columns(X, cols) or not imputer:
            return X, _y
        try:
            return _sklearn_transform_subset(X, cols, imputer, is_polars=True), _y
        except Exception as e:
            logger.error(f"Iterative Imputation failed: {e}")
            return X, _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        imputer = params.get("imputer_object")
        if not resolve_valid_columns(X, cols) or not imputer:
            return X, _y
        try:
            return _sklearn_transform_subset(X, cols, imputer, is_polars=False), _y
        except Exception as e:
            logger.error(f"Iterative Imputation failed: {e}")
            return X, _y


@NodeRegistry.register("IterativeImputer", IterativeImputerApplier)
@node_meta(
    id="IterativeImputer",
    name="Iterative Imputer (MICE)",
    category="Preprocessing",
    description="Multivariate imputation using chained equations.",
    params={"max_iter": 10, "random_state": 0, "estimator": "bayesian_ridge", "columns": []},
)
class IterativeImputerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # MICE imputation fills NaNs in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> IterativeImputerArtifact:
        if user_picked_no_columns(config):
            return {}

        max_iter = config.get("max_iter", 10)
        estimator_name = config.get("estimator", "BayesianRidge")
        cols = resolve_columns(X, config, detect_numeric_columns)
        if not cols:
            return {}

        estimator = _build_iterative_estimator(estimator_name)
        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=0)

        X_subset = X.select(cols) if hasattr(X, "select") and not hasattr(X, "loc") else X[cols]
        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        imputer.fit(X_np)

        return {
            "type": "iterative_imputer",
            "imputer_object": imputer,  # Not JSON serializable
            "columns": cols,
            "estimator": estimator_name,
        }
