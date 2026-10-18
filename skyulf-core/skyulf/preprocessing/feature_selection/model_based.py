"""Model-based (importance-weight) feature selector."""

from typing import Any, cast

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from .._artifacts import ModelBasedSelectionArtifact
from .._fitted_validation import local_state_fields
from .._helpers import select_then_to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import (
    _build_model_selector,
    _drop_selected_pandas,
    _drop_selected_polars,
    _extract_target,
    _fillna_zero_with_warning,
    _local_selection_columns,
    _local_selection_drop_flag,
    _model_feature_importances,
    _prepare_sklearn_y,
    _resolve_candidate_columns,
    _resolve_estimator,
    _resolve_problem_type,
)


class ModelBasedSelectionApplier(BaseApplier):
    """Remove the columns an estimator's importances rejected."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved membership while leaving unused fit reports unchanged."""
        fields = {
            "type",
            "selected_columns",
            "candidate_columns",
            "method",
            "drop_columns",
            "feature_importances",
        }
        if local_state_fields(raw, "model_based_selection", fields, allow_empty=True):
            selected = _local_selection_columns(raw["selected_columns"])
            candidates = _local_selection_columns(raw["candidate_columns"])
            if not set(selected).issubset(candidates):
                raise ValueError("Fitted selected columns must belong to candidate columns.")
            _local_selection_drop_flag(raw["drop_columns"])
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe saved column selection without rerunning the training estimator."""
        if engine not in {"pandas", "polars"}:
            return None
        ModelBasedSelectionApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Drop ``candidate_columns - selected_columns`` from ``X``.

        Honours the artifact's ``drop_columns`` flag; the recorded
        ``feature_importances`` are reporting metadata and take no part in the
        transform.
        """
        return apply_dual_engine(
            X, params, {"polars": _drop_selected_polars, "pandas": _drop_selected_pandas}
        )


@NodeRegistry.register("ModelBasedSelection", ModelBasedSelectionApplier)
@node_meta(
    id="ModelBasedSelection",
    name="Model-Based Selection",
    category="Feature Selection",
    description="Select features using importance weights, RFE, RFECV or sequential CV.",
    params={
        "method": "select_from_model",
        "estimator": "RandomForest",
        "threshold": "mean",
        "max_features": None,
        "step": 1,
        "cv": 5,
        "scoring": None,
        "min_features_to_select": 1,
        "direction": "forward",
    },
    tags=["rfe", "rfecv", "sequential_feature_selection"],
    learns_from_data=True,
)
class ModelBasedSelectionCalculator(BaseCalculator):
    """Fit importance, recursive, or sequential selection around a resolved estimator.

    RFECV and sequential selection use integer unshuffled cross-validation,
    defaulting to five folds. Small datasets must choose a valid lower fold
    count; folds are not silently capped. Saved replay only uses feature names.
    """

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> ModelBasedSelectionArtifact:  # pylint: disable=arguments-differ
        """Return the features the fitted estimator's support keeps.

        This selector is supervised: with no target there is nothing to score
        against, so a missing target raises. Unsupported methods, estimators and
        problem types also raise, and estimator fit errors propagate unchanged.
        No candidate columns still produces an empty no-op artifact.
        """
        target_col = config.get("target_column")
        # Resolve candidate columns natively on the raw frame first (works on
        # Polars without conversion), then convert only the columns actually
        # needed -- candidates plus the target column when it must be pulled
        # out of X itself (y is None) -- instead of the whole input frame.
        cols = _resolve_candidate_columns(X, config, target_col)
        if not cols:
            return cast(ModelBasedSelectionArtifact, {})

        needed_cols = cols if y is not None or not target_col else [*cols, target_col]
        X_pd = select_then_to_pandas(X, needed_cols)

        y = _extract_target(X_pd, y, target_col)
        if y is None:
            raise ValueError(
                f"ModelBasedSelection requires target column '{target_col}' "
                "to be present in training data."
            )

        method = config.get("method", "select_from_model")
        estimator_name = config.get("estimator", "auto")
        problem_type = _resolve_problem_type(config.get("problem_type", "auto"), y)

        estimator = _resolve_estimator(estimator_name, problem_type)
        selector = _build_model_selector(method, estimator, config)

        X_np, _ = SklearnBridge.to_sklearn(_fillna_zero_with_warning(X_pd, cols))
        selector.fit(X_np, _prepare_sklearn_y(y, problem_type))
        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support, strict=True) if s]

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
