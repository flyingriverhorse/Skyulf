"""Iterative imputer node (MICE / chained equations)."""

from typing import Any

# Side-effect import (F401 by design): activates sklearn's experimental
# IterativeImputer so the import below succeeds.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, user_picked_no_columns
from .._artifacts import IterativeImputerArtifact
from .._helpers import resolve_columns_then_to_numpy, resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _build_iterative_estimator, _sklearn_transform_subset


class IterativeImputerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(
            X, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        cols = params.get("columns", [])
        imputer = params.get("imputer_object")
        if not resolve_valid_columns(X, cols) or not imputer:
            return X, _y
        return _sklearn_transform_subset(X, cols, imputer, is_polars=True), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        cols = params.get("columns", [])
        imputer = params.get("imputer_object")
        if not resolve_valid_columns(X, cols) or not imputer:
            return X, _y
        return _sklearn_transform_subset(X, cols, imputer, is_polars=False), _y


@NodeRegistry.register("IterativeImputer", IterativeImputerApplier)
@node_meta(
    id="IterativeImputer",
    name="Iterative Imputer (MICE)",
    category="Preprocessing",
    description="Multivariate imputation using chained equations.",
    params={"max_iter": 10, "random_state": 0, "estimator": "BayesianRidge", "columns": []},
    learns_from_data=True,
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
        random_state = config.get("random_state", 0)

        # KNN/Iterative imputers always fit through numpy — engine choice
        # doesn't affect the fit math, so we skip the Pandas hop entirely.
        X_np, cols = resolve_columns_then_to_numpy(X, config, detect_numeric_columns)
        if not cols:
            return {}

        estimator = _build_iterative_estimator(estimator_name)
        imputer = IterativeImputer(
            estimator=estimator, max_iter=max_iter, random_state=random_state
        )
        imputer.fit(X_np)

        return {
            "type": "iterative_imputer",
            "imputer_object": imputer,  # Not JSON serializable
            "columns": cols,
            "estimator": estimator_name,
        }
