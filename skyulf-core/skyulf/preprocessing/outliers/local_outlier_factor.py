"""LocalOutlierFactor training exclusion and saved novelty scoring."""

from typing import Any

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import pack_pipeline_output, unpack_pipeline_input, user_picked_no_columns
from .._helpers import select_rows_by_position
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine, fit_transform_train_dual_engine
from ._multivariate import (
    filter_detector_rows,
    fit_detector,
    fitted_array,
    reject_boolean_options,
    validate_detector,
)


def _fit(X: Any, y: Any, config: dict) -> dict:
    """Fit a novelty-capable model while retaining its distinct training outlier scores."""
    reject_boolean_options(config, ("n_neighbors", "contamination"))
    detector = LocalOutlierFactor(
        n_neighbors=config.get("n_neighbors", 20),
        contamination=config.get("contamination", 0.01),
        novelty=True,
    )
    return fit_detector(X, y, config, detector, "local_outlier_factor")


def _fit_train(X: Any, y: Any, config: dict) -> tuple[dict, Any, Any]:
    """Use learned training scores rather than incorrectly novelty-scoring training samples."""
    state = _fit(X, y, config)
    output, target = filter_detector_rows(X, y, state, training=True)
    return state, output, target


class LocalOutlierFactorApplier(BaseApplier):
    """Exclude unseen rows using the fitted training population as the neighbor reference."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Require novelty mode and internally consistent learned neighbor statistics."""
        model = validate_detector(raw, "local_outlier_factor", LocalOutlierFactor)
        if model is None:
            return raw
        if model.novelty is not True or model.metric != "minkowski" or model.p != 2:
            raise ValueError("Fitted LocalOutlierFactor requires Euclidean novelty scoring.")
        rows = getattr(model, "n_samples_fit_", 0)
        neighbors = getattr(model, "n_neighbors_", 0)
        if not 1 <= neighbors < rows:
            raise ValueError("Fitted LocalOutlierFactor neighbor count is invalid.")
        fitted_array(model._fit_X, (rows, len(raw["columns"])))
        fitted_array(model._lrd, (rows,), positive=True)
        fitted_array(model.negative_outlier_factor_, (rows,))
        fitted_array(model._distances_fit_X_, (rows, neighbors))
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Declare local row filtering against fixed learned neighbors."""
        if engine not in {"pandas", "polars"}:
            return None
        LocalOutlierFactorApplier.validate_inference_state(state)
        return ExecutionCapability(
            engine, "apply", "local", "filter" if state else "preserve", "row"
        )

    @apply_method
    def apply(self, X: Any, y: Any, params: dict) -> Any:
        """Score new rows only, retaining nonfinite rows for explicit downstream handling."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"pandas": filter_detector_rows, "polars": filter_detector_rows},
        )


@NodeRegistry.register("LocalOutlierFactor", LocalOutlierFactorApplier)
@node_meta(
    id="LocalOutlierFactor",
    name="Local Outlier Factor",
    category="Preprocessing",
    description="Exclude joint local-density outliers; use fixed neighbors for new rows.",
    params={"columns": [], "n_neighbors": 20, "contamination": 0.01},
    learns_from_data=True,
)
class LocalOutlierFactorCalculator(BaseCalculator):
    """Learn LOF training scores and novelty state with separate train/apply representations."""

    def infer_output_schema(self, input_schema: Any, config: dict) -> Any:
        """Filtering rows retains all input column names and dtypes."""
        return input_schema

    @fit_method
    def fit(self, X: Any, y: Any, config: dict) -> dict:
        """Fit a saved detector; use fit_transform_train to obtain filtered training data."""
        if user_picked_no_columns(config):
            return {}
        return fit_dual_engine(
            (X, y) if y is not None else X, config, {"pandas": _fit, "polars": _fit}
        )

    def fit_transform_train(self, df: Any, config: dict) -> tuple[dict, Any]:
        """Filter training rows with negative_outlier_factor_, never novelty predict."""
        if user_picked_no_columns(config):
            return {}, df
        return fit_transform_train_dual_engine(
            df, config, {"pandas": _fit_train, "polars": _fit_train}
        )


def fit_transform_train_weighted(
    calculator: LocalOutlierFactorCalculator, data: Any, config: dict, weights: Any
) -> tuple[dict, Any, Any]:
    """Slice original targets and weights with LOF's exact unsupervised training positions."""
    X, y, paired = unpack_pipeline_input(data)
    state = calculator.fit(data, config)
    result, positions = filter_detector_rows(X, np.arange(len(X)), state, training=True)
    output = pack_pipeline_output(result, select_rows_by_position(y, positions), paired)
    return state, output, weights[positions]
