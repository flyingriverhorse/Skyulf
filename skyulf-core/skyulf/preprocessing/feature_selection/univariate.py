"""Univariate statistical feature selector."""

import logging
from typing import Any, Dict, cast

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from .._artifacts import UnivariateSelectionArtifact
from .._helpers import to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import (
    _build_univariate_selector,
    _drop_selected_pandas,
    _drop_selected_polars,
    _extract_target,
    _maybe_chi2_rescale,
    _prepare_sklearn_y,
    _resolve_candidate_columns,
    _resolve_problem_type,
    _resolve_score_function,
    _univariate_no_target_artifact,
    _univariate_score_dicts,
)

logger = logging.getLogger(__name__)


class UnivariateSelectionApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _drop_selected_polars, _drop_selected_pandas)


@NodeRegistry.register("UnivariateSelection", UnivariateSelectionApplier)
@node_meta(
    id="UnivariateSelection",
    name="Univariate Selection",
    category="Feature Selection",
    description="Select best features based on univariate statistical tests.",
    params={"method": "SelectKBest", "score_func": "f_classif", "k": 10},
)
class UnivariateSelectionCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> UnivariateSelectionArtifact:
        target_col = config.get("target_column")
        X_pd = to_pandas(X)

        y = _extract_target(X_pd, y, target_col)
        if y is None and not config.get("allow_missing_target", False):
            logger.error(
                f"UnivariateSelection requires target column '{target_col}' "
                "to be present in training data."
            )
            return cast(UnivariateSelectionArtifact, {})

        cols = _resolve_candidate_columns(X_pd, config, target_col)
        if not cols:
            return cast(UnivariateSelectionArtifact, {})

        method = config.get("method", "select_k_best")
        score_func_name = config.get("score_func")
        problem_type = _resolve_problem_type(config.get("problem_type", "auto"), y)
        score_func = _resolve_score_function(score_func_name, problem_type)

        selector = _build_univariate_selector(method, score_func, config)
        if selector is None:
            return cast(UnivariateSelectionArtifact, {})

        X_np, _ = SklearnBridge.to_sklearn(X_pd[cols].fillna(0))
        X_np = _maybe_chi2_rescale(X_np, score_func_name)

        if y is None:
            return _univariate_no_target_artifact(cols, method, config)

        selector.fit(X_np, _prepare_sklearn_y(y, problem_type))
        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support) if s]
        scores, pvalues = _univariate_score_dicts(selector, cols)

        return cast(
            UnivariateSelectionArtifact,
            {
                "type": "univariate_selection",
                "selected_columns": selected_cols,
                "candidate_columns": cols,
                "method": method,
                "drop_columns": config.get("drop_columns", True),
                "feature_scores": scores,
                "p_values": pvalues,
            },
        )
