"""Model-based (importance-weight) feature selector."""

import logging
from typing import Any, Dict, cast

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from .._artifacts import ModelBasedSelectionArtifact
from .._helpers import to_pandas
from ._common import (
    _build_model_selector,
    _drop_selected_pandas,
    _drop_selected_polars,
    _extract_target,
    _model_feature_importances,
    _prepare_sklearn_y,
    _resolve_candidate_columns,
    _resolve_estimator,
    _resolve_problem_type,
)

logger = logging.getLogger(__name__)


class ModelBasedSelectionApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _drop_selected_polars, _drop_selected_pandas)


@NodeRegistry.register("ModelBasedSelection", ModelBasedSelectionApplier)
@node_meta(
    id="ModelBasedSelection",
    name="Model-Based Selection",
    category="Feature Selection",
    description="Select features based on importance weights.",
    params={"estimator": "RandomForest", "threshold": "mean", "max_features": None},
)
class ModelBasedSelectionCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> ModelBasedSelectionArtifact:
        target_col = config.get("target_column")
        X_pd = to_pandas(X)

        y = _extract_target(X_pd, y, target_col)
        if y is None:
            logger.error(
                f"ModelBasedSelection requires target column '{target_col}' "
                "to be present in training data."
            )
            return cast(ModelBasedSelectionArtifact, {})

        cols = _resolve_candidate_columns(X_pd, config, target_col)
        if not cols:
            return cast(ModelBasedSelectionArtifact, {})

        method = config.get("method", "select_from_model")
        estimator_name = config.get("estimator", "auto")
        problem_type = _resolve_problem_type(config.get("problem_type", "auto"), y)

        estimator = _resolve_estimator(estimator_name, problem_type)
        if estimator is None:
            logger.error(
                f"Could not resolve estimator '{estimator_name}' for problem type '{problem_type}'"
            )
            return cast(ModelBasedSelectionArtifact, {})

        selector = _build_model_selector(method, estimator, config)
        if selector is None:
            return cast(ModelBasedSelectionArtifact, {})

        X_np, _ = SklearnBridge.to_sklearn(X_pd[cols].fillna(0))
        selector.fit(X_np, _prepare_sklearn_y(y, problem_type))
        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support) if s]

        return cast(
            ModelBasedSelectionArtifact,
            {
                "type": "model_based_selection",
                "selected_columns": selected_cols,
                "candidate_columns": cols,
                "method": method,
                "drop_columns": config.get("drop_columns", True),
                "feature_importances": _model_feature_importances(selector, cols),
            },
        )
