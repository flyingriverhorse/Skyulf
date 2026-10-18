"""Joint IsolationForest exclusion with fixed learned trees."""

from importlib import import_module
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.tree import ExtraTreeRegressor

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import user_picked_no_columns
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._multivariate import (
    filter_detector_rows,
    fit_detector,
    reject_boolean_options,
    validate_detector,
)

_tree = import_module("sklearn.tree._tree")


def _fit(X: Any, y: Any, config: dict) -> dict:
    """Build the fixed serializable forest recipe and fit the joint feature matrix."""
    reject_boolean_options(config, ("n_estimators", "max_samples", "contamination", "random_state"))
    detector = IsolationForest(
        n_estimators=config.get("n_estimators", 100),
        max_samples=config.get("max_samples", "auto"),
        contamination=config.get("contamination", 0.01),
        random_state=config.get("random_state", 42),
    )
    return fit_detector(X, y, config, detector, "isolation_forest")


def _validate_tree(tree: Any, features: Any, width: int) -> None:
    """Reject nested executable overrides and inconsistent sklearn tree feature axes."""
    if type(tree) is not ExtraTreeRegressor or type(getattr(tree, "tree_", None)) is not _tree.Tree:
        raise ValueError("Fitted IsolationForest contains an unsupported tree.")
    if any(callable(value) for value in vars(tree).values()):
        raise ValueError("Overridden IsolationForest tree methods are unsupported.")
    if tree.n_features_in_ != width or tree.tree_.n_features != width:
        raise ValueError("Fitted IsolationForest tree feature width is invalid.")
    if not np.array_equal(features, np.arange(width)):
        raise ValueError("Fitted IsolationForest tree feature axes are invalid.")


class IsolationForestApplier(BaseApplier):
    """Exclude rows using the saved joint forest, retaining unscored nonfinite rows."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect the fitted forest and its feature axes without running tree inference."""
        model = validate_detector(raw, "isolation_forest", IsolationForest)
        if model is None:
            return raw
        trees = getattr(model, "estimators_", None)
        axes = getattr(model, "estimators_features_", None)
        if (
            not isinstance(trees, list)
            or not trees
            or not isinstance(axes, list)
            or len(trees) != len(axes)
        ):
            raise ValueError("Fitted IsolationForest requires matching tree and feature lists.")
        for tree, features in zip(trees, axes, strict=True):
            _validate_tree(tree, features, len(raw["columns"]))
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Declare local row filtering only after validating the fitted detector."""
        if engine not in {"pandas", "polars"}:
            return None
        IsolationForestApplier.validate_inference_state(state)
        return ExecutionCapability(
            engine, "apply", "local", "filter" if state else "preserve", "row"
        )

    @apply_method
    def apply(self, X: Any, y: Any, params: dict) -> Any:
        """Apply one positional mask to features and targets on either native engine."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"pandas": filter_detector_rows, "polars": filter_detector_rows},
        )


@NodeRegistry.register("IsolationForest", IsolationForestApplier)
@node_meta(
    id="IsolationForest",
    name="Isolation Forest",
    category="Preprocessing",
    description="Exclude multivariate outliers using a learned joint isolation forest.",
    params={
        "columns": [],
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": 0.01,
        "random_state": 42,
    },
    learns_from_data=True,
)
class IsolationForestCalculator(BaseCalculator):
    """Fit one forest across selected numeric features, excluding the implicit target."""

    def infer_output_schema(self, input_schema: Any, config: dict) -> Any:
        """Row exclusion preserves input column names and dtypes."""
        return input_schema

    @fit_method
    def fit(self, X: Any, y: Any, config: dict) -> dict:
        """Learn only from finite training rows; malformed options and fit errors propagate."""
        if user_picked_no_columns(config):
            return {}
        return fit_dual_engine(
            (X, y) if y is not None else X, config, {"pandas": _fit, "polars": _fit}
        )
