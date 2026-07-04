"""KNN imputer node (k-Nearest Neighbors)."""

import logging
from typing import Any, Dict, Tuple

from sklearn.impute import KNNImputer

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns, user_picked_no_columns
from .._artifacts import KNNImputerArtifact
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _sklearn_transform_subset

logger = logging.getLogger(__name__)


class KNNImputerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
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
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> KNNImputerArtifact:  # pylint: disable=arguments-differ
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
