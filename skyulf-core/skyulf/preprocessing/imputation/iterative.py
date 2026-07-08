"""Iterative imputer node (MICE / chained equations)."""

import logging
from typing import Any

# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns, user_picked_no_columns
from .._artifacts import IterativeImputerArtifact
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _build_iterative_estimator, _sklearn_transform_subset

logger = logging.getLogger(__name__)


class IterativeImputerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # MICE imputation fills NaNs in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> IterativeImputerArtifact:  # pylint: disable=arguments-differ
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
