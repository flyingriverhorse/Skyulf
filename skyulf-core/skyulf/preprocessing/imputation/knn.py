"""KNN imputer node (k-Nearest Neighbors)."""

from typing import Any

import numpy as np
from sklearn.impute import KNNImputer

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, user_picked_no_columns
from .._artifacts import KNNImputerArtifact
from .._fitted_validation import local_state_fields
from .._helpers import (
    promote_configured_columns_to_float64,
    resolve_columns_then_to_numpy,
    resolve_valid_columns,
)
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import (
    _sklearn_transform_subset,
    _validate_imputer_array,
    _validate_local_imputer,
    drop_all_missing_columns,
)


def _validate_knn_neighbors(imputer: KNNImputer, width: int) -> None:
    """Inspect learned donor rows and their masks without recomputing neighbors."""
    donors = getattr(imputer, "_fit_X", None)
    if not isinstance(donors, np.ndarray) or donors.ndim != 2 or not len(donors):
        raise ValueError("Fitted KNN donor rows must be a nonempty matrix.")
    _validate_imputer_array(donors, (len(donors), width), "f")
    _validate_imputer_array(getattr(imputer, "_mask_fit_X", None), donors.shape, "b")
    valid = _validate_imputer_array(getattr(imputer, "_valid_mask", None), (width,), "b")
    if not valid.all() and not imputer.keep_empty_features:
        raise ValueError("Fitted KNN donor columns would change the saved output width.")


class KNNImputerApplier(BaseApplier):
    """Fill missing values from the fitted k-nearest-neighbors imputer."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect the saved neighbor estimator without refitting or normalizing it."""
        fields = {"type", "imputer_object", "columns", "n_neighbors", "weights"}
        if local_state_fields(raw, "knn_imputer", fields, allow_empty=True):
            imputer = _validate_local_imputer(raw, KNNImputer)
            _validate_knn_neighbors(imputer, len(raw["columns"]))
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe native neighbor replay; arbitrary metrics or weights stay undeclared."""
        if engine not in {"pandas", "polars"}:
            return None
        KNNImputerApplier.validate_inference_state(state)
        if state:
            imputer = state["imputer_object"]
            if callable(imputer.weights) or callable(imputer.metric):
                return None
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Impute ``X`` with the stored sklearn imputer; ``y`` passes through."""
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


@NodeRegistry.register("KNNImputer", KNNImputerApplier)
@node_meta(
    id="KNNImputer",
    name="KNN Imputer",
    category="Preprocessing",
    description="Impute missing values using k-Nearest Neighbors.",
    params={"n_neighbors": 5, "weights": "uniform", "columns": []},
    learns_from_data=True,
)
class KNNImputerCalculator(BaseCalculator):
    """Fit a sklearn ``KNNImputer`` on the selected numeric columns."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return a schema with selected columns promoted to ``float64``."""
        return promote_configured_columns_to_float64(input_schema, config)

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> KNNImputerArtifact:  # pylint: disable=arguments-differ
        """Fit the neighbor imputer, dropping all-missing columns sklearn cannot impute."""
        if user_picked_no_columns(config):
            return {}

        n_neighbors = config.get("n_neighbors", 5)
        weights = config.get("weights", "uniform")

        # KNN/Iterative imputers always fit through numpy — engine choice
        # doesn't affect the fit math, so we skip the Pandas hop entirely.
        X_np, cols = resolve_columns_then_to_numpy(X, config, detect_numeric_columns)
        if not cols:
            return {}

        # sklearn silently drops all-missing columns from transform() output,
        # which desyncs the artifact's column list from the imputer's width.
        X_np, cols = drop_all_missing_columns(X_np, cols, "KNNImputerCalculator")
        if not cols:
            return {}

        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputer.fit(X_np)

        return {
            "type": "knn_imputer",
            "imputer_object": imputer,  # Not JSON serializable
            "columns": cols,
            "n_neighbors": n_neighbors,
            "weights": weights,
        }
