"""Iterative imputer node (MICE / chained equations)."""

from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

# Side-effect import (F401 by design): activates sklearn's experimental
# IterativeImputer so the import below succeeds.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, user_picked_no_columns
from .._artifacts import IterativeImputerArtifact
from .._fitted_validation import local_boolean, local_state_fields
from .._helpers import (
    promote_configured_columns_to_float64,
    resolve_columns_then_to_numpy,
    resolve_valid_columns,
)
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import (
    _build_iterative_estimator,
    _sklearn_transform_subset,
    _validate_imputer_array,
    _validate_local_imputer,
    drop_all_missing_columns,
)

_LOCAL_ESTIMATORS = (BayesianRidge, DecisionTreeRegressor, ExtraTreesRegressor, KNeighborsRegressor)


def _validate_iterative_initial(imputer: IterativeImputer, width: int) -> None:
    """Require the saved initial statistics and output width used by transform."""
    initial = getattr(imputer, "initial_imputer_", None)
    if type(initial) is not SimpleImputer or getattr(initial, "n_features_in_", None) != width:
        raise ValueError("Fitted iterative initial imputer is missing or has the wrong width.")
    _validate_imputer_array(getattr(initial, "statistics_", None), (width,), "fiu")
    empty = _validate_imputer_array(getattr(imputer, "_is_empty_feature", None), (width,), "b")
    if empty.any() and not imputer.keep_empty_features:
        raise ValueError("Fitted iterative empty columns would change the saved output width.")
    local_boolean(imputer.sample_posterior, "sample_posterior")


def _validate_iterative_sequence(imputer: IterativeImputer, width: int) -> None:
    """Inspect saved rounds and predictor widths without executing learned estimators."""
    rounds = getattr(imputer, "n_iter_", None)
    if not isinstance(rounds, (int, np.integer)) or rounds < 0:
        raise ValueError("Fitted iterative rounds must be a nonnegative integer.")
    if rounds:
        for bound in ("_min_value", "_max_value"):
            _validate_imputer_array(getattr(imputer, bound, None), (width,), "fiu")
    sequence = getattr(imputer, "imputation_sequence_", None)
    if not isinstance(sequence, (list, tuple)):
        raise ValueError("Fitted iterative estimator sequence is missing.")
    for triplet in sequence:
        _validate_iterative_triplet(triplet, width)


def _validate_iterative_triplet(triplet: Any, width: int) -> None:
    """Keep selected feature indices aligned with their fitted prediction estimator."""
    feature = getattr(triplet, "feat_idx", None)
    if not isinstance(feature, (int, np.integer)) or not 0 <= feature < width:
        raise ValueError("Fitted iterative feature index is invalid.")
    neighbors = getattr(triplet, "neighbor_feat_idx", None)
    if not isinstance(neighbors, np.ndarray) or neighbors.ndim != 1:
        raise ValueError("Fitted iterative neighbors must be an index vector.")
    _validate_imputer_array(neighbors, neighbors.shape, "iu")
    if np.any(neighbors >= width) or np.any(neighbors < 0):
        raise ValueError("Fitted iterative neighbor index is invalid.")
    estimator = getattr(triplet, "estimator", None)
    if getattr(estimator, "n_features_in_", None) != len(neighbors):
        raise ValueError("Fitted iterative predictor width disagrees with its neighbors.")


class IterativeImputerApplier(BaseApplier):
    """Fill missing values from the fitted MICE (chained-equations) imputer."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect learned initial values and estimator sequence without executing them."""
        fields = {"type", "imputer_object", "columns", "estimator"}
        if local_state_fields(raw, "iterative_imputer", fields, allow_empty=True):
            imputer = _validate_local_imputer(raw, IterativeImputer)
            width = len(raw["columns"])
            _validate_iterative_initial(imputer, width)
            _validate_iterative_sequence(imputer, width)
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Retain the all-missing request shortcut and abstain from stochastic replay."""
        if engine not in {"pandas", "polars"}:
            return None
        IterativeImputerApplier.validate_inference_state(state)
        context = "row"
        if state:
            imputer = state["imputer_object"]
            if imputer.sample_posterior or any(
                type(step.estimator) not in _LOCAL_ESTIMATORS
                for step in imputer.imputation_sequence_
            ):
                return None
            if imputer.n_iter_:
                context = "global"
        return ExecutionCapability(engine, "apply", "local", "preserve", context)

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
    """Fit a sklearn ``IterativeImputer`` with the configured estimator."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return a schema with selected columns promoted to ``float64``."""
        return promote_configured_columns_to_float64(input_schema, config)

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> IterativeImputerArtifact:  # pylint: disable=arguments-differ
        """Fit chained-equations imputation, dropping all-missing columns sklearn cannot impute."""
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

        # sklearn silently drops all-missing columns from transform() output,
        # which desyncs the artifact's column list from the imputer's width.
        X_np, cols = drop_all_missing_columns(X_np, cols, "IterativeImputerCalculator")
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
