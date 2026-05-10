"""Feature-selection nodes (Variance / Correlation / Univariate / Model-based).

Each Applier dispatches on engine via :func:`apply_dual_engine` (see
``dispatcher.py``); per-engine logic lives in small ``_apply_polars`` /
``_apply_pandas`` static helpers. Calculator ``fit`` paths are sklearn-bound,
so they use :func:`to_pandas` once at the top instead of going through
``fit_dual_engine``.
"""

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..utils import (
    detect_numeric_columns,
    resolve_columns,
)
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine
from ._artifacts import (
    CorrelationThresholdArtifact,
    ModelBasedSelectionArtifact,
    UnivariateSelectionArtifact,
    VarianceThresholdArtifact,
)
from ._helpers import to_pandas
from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..engines.sklearn_bridge import SklearnBridge

logger = logging.getLogger(__name__)

# --- Helpers ---
SCORE_FUNCTIONS: Dict[str, Callable] = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
    "chi2": chi2,
    "r_regression": r_regression,
}


def _infer_problem_type(series: pd.Series) -> str:
    if series.empty:
        return "classification"
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
        return "classification"
    unique_values = series.dropna().unique()
    if len(unique_values) <= 10:
        return "classification"
    return "regression"


def _resolve_score_function(name: Optional[str], problem_type: str) -> Any:
    if name and name in SCORE_FUNCTIONS:
        return SCORE_FUNCTIONS[name]

    if problem_type == "classification":
        return f_classif
    return f_regression


def _resolve_estimator(key: Optional[str], problem_type: str) -> Any:
    key = (key or "auto").lower()
    if problem_type == "classification":
        if key in {"auto", "logistic_regression", "logisticregression"}:
            return LogisticRegression(max_iter=1000)
        if key in {"random_forest", "randomforest"}:
            return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        if key in {"linear_regression", "linearregression"}:
            return LinearRegression()  # Odd for classification but allowed in V1 logic
    else:
        if key in {"auto", "linear_regression", "linearregression"}:
            return LinearRegression()
        if key in {"random_forest", "randomforest"}:
            return RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    return None


# -----------------------------------------------------------------------------
# Shared apply helpers (drop unselected columns)
# -----------------------------------------------------------------------------


def _resolve_drop_list(params: Dict[str, Any], existing_cols: List[str]) -> List[str]:
    """Compute the column-drop list from selected/candidate params."""
    selected = params.get("selected_columns")
    candidates = params.get("candidate_columns", [])
    if selected is None:
        return []
    return [c for c in (set(candidates) - set(selected)) if c in existing_cols]


def _drop_selected_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Polars apply path for selectors that drop ``candidate \\ selected`` columns."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = _resolve_drop_list(params, list(X.columns))
    if to_drop:
        X = X.drop(to_drop)
    return X, y


def _drop_selected_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Pandas apply path for selectors that drop ``candidate \\ selected`` columns."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = _resolve_drop_list(params, list(X.columns))
    if to_drop:
        X = X.drop(columns=to_drop)
    return X, y


# -----------------------------------------------------------------------------
# Shared fit helpers (target extraction + sklearn subset prep)
# -----------------------------------------------------------------------------


def _extract_target(X_pd: pd.DataFrame, y: Any, target_col: Optional[str]) -> Optional[pd.Series]:
    """Return ``y`` if provided; else pull ``target_col`` from the (pandas) frame."""
    if y is not None:
        return y
    if not target_col or target_col not in X_pd.columns:
        return None
    return X_pd[target_col]


def _prepare_sklearn_y(y: Any, problem_type: str) -> np.ndarray:
    """Convert ``y`` to a numpy array, factorising non-numeric classification targets."""
    y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)
    if problem_type == "classification" and not np.issubdtype(y_np.dtype, np.number):
        y_factorized, _ = pd.factorize(y_np)
        return y_factorized
    return y_np


def _resolve_problem_type(declared: str, y: Any) -> str:
    """Resolve ``"auto"`` problem-type using ``y``; defaults to classification."""
    if declared != "auto":
        return declared
    if y is None:
        return "classification"
    return _infer_problem_type(y)


_GENERIC_PARAM_KEYS: Dict[str, Tuple[str, Any]] = {
    "k_best": ("k", 10),
    "percentile": ("percentile", 10),
}


def _resolve_generic_param(config: Dict[str, Any]) -> Any:
    """Pick the GenericUnivariateSelect ``param`` from explicit or mode-derived config."""
    if "param" in config:
        return config.get("param")
    mode = config.get("mode", "k_best")
    key, default = _GENERIC_PARAM_KEYS.get(mode, ("alpha", 0.05))
    return config.get(key, default)


_UNIVARIATE_SELECTOR_BUILDERS: Dict[str, Callable[[Any, Dict[str, Any]], Any]] = {
    "select_k_best": lambda sf, cfg: SelectKBest(score_func=sf, k=cfg.get("k", 10)),
    "select_percentile": lambda sf, cfg: SelectPercentile(
        score_func=sf, percentile=cfg.get("percentile", 10)
    ),
    "select_fpr": lambda sf, cfg: SelectFpr(score_func=sf, alpha=cfg.get("alpha", 0.05)),
    "select_fdr": lambda sf, cfg: SelectFdr(score_func=sf, alpha=cfg.get("alpha", 0.05)),
    "select_fwe": lambda sf, cfg: SelectFwe(score_func=sf, alpha=cfg.get("alpha", 0.05)),
    "generic_univariate_select": lambda sf, cfg: GenericUnivariateSelect(
        score_func=sf, mode=cfg.get("mode", "k_best"), param=_resolve_generic_param(cfg)
    ),
}


def _build_univariate_selector(
    method: str, score_func: Any, config: Dict[str, Any]
) -> Optional[Any]:
    """Construct the sklearn univariate selector named by ``method``."""
    builder = _UNIVARIATE_SELECTOR_BUILDERS.get(method)
    return builder(score_func, config) if builder else None


def _build_model_selector(method: str, estimator: Any, config: Dict[str, Any]) -> Optional[Any]:
    """Construct the sklearn model-based selector named by ``method``."""
    if method == "select_from_model":
        threshold = config.get("threshold", "mean")
        if isinstance(threshold, str):
            try:
                threshold = float(threshold)
            except ValueError:
                pass  # Keep as string (e.g. "mean", "1.25*mean")
        return SelectFromModel(
            estimator=estimator,
            threshold=threshold,
            max_features=config.get("max_features", None),
        )
    if method == "rfe":
        return RFE(
            estimator=estimator,
            n_features_to_select=config.get("n_features_to_select", None),
            step=config.get("step", 1),
        )
    return None


def _maybe_chi2_rescale(X_np: np.ndarray, score_func_name: Optional[str]) -> np.ndarray:
    """Apply MinMax rescale when chi2 is requested but features contain negatives."""
    if score_func_name != "chi2" or not (X_np < 0).any():
        return X_np
    logger.warning(
        "Chi-squared statistic requires non-negative feature values. "
        "Applying MinMaxScaler to features for selection."
    )
    from sklearn.preprocessing import MinMaxScaler

    return MinMaxScaler().fit_transform(X_np)


def _resolve_candidate_columns(
    X_pd: pd.DataFrame, config: Dict[str, Any], target_col: Optional[str]
) -> List[str]:
    """Return numeric candidate columns minus the target."""
    cols = resolve_columns(X_pd, config, lambda d: detect_numeric_columns(d, exclude_binary=False))
    return [c for c in cols if c != target_col]


def _univariate_score_dicts(
    selector: Any, cols: List[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Pull (scores, pvalues) off a fitted univariate selector, NaN-safe."""
    scores: Dict[str, float] = {}
    pvalues: Dict[str, float] = {}
    if hasattr(selector, "scores_"):
        safe_scores = np.nan_to_num(selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        scores = dict(zip(cols, safe_scores.tolist()))
    if hasattr(selector, "pvalues_"):
        safe_pvalues = np.nan_to_num(cast(Any, selector.pvalues_), nan=1.0)
        pvalues = dict(zip(cols, safe_pvalues.tolist()))
    return scores, pvalues


def _univariate_no_target_artifact(
    cols: List[str], method: str, config: Dict[str, Any]
) -> "UnivariateSelectionArtifact":
    """Artifact returned when the selector ran without a target (passthrough)."""
    return cast(
        UnivariateSelectionArtifact,
        {
            "type": "univariate_selection",
            "selected_columns": cols,
            "candidate_columns": cols,
            "method": method,
            "drop_columns": config.get("drop_columns", True),
            "scores": {},
            "pvalues": {},
        },
    )


def _model_feature_importances(selector: Any, cols: List[str]) -> Dict[str, float]:
    """Pull importances or |coef| off a fitted model-based selector."""
    estimator = getattr(selector, "estimator_", None)
    if estimator is None:
        return {}
    if hasattr(estimator, "feature_importances_"):
        return dict(zip(cols, estimator.feature_importances_.tolist()))
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return dict(zip(cols, np.abs(coef).tolist()))
    return {}


# --- Variance Threshold ---


class VarianceThresholdApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _drop_selected_polars, _drop_selected_pandas)


@NodeRegistry.register("VarianceThreshold", VarianceThresholdApplier)
@node_meta(
    id="VarianceThreshold",
    name="Variance Threshold",
    category="Feature Selection",
    description="Remove features with low variance.",
    params={"threshold": 0.0},
)
class VarianceThresholdCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> VarianceThresholdArtifact:
        threshold = config.get("threshold", 0.0)
        drop_columns = config.get("drop_columns", True)

        X_pd = to_pandas(X)
        cols = resolve_columns(
            X_pd,
            config,
            lambda d: detect_numeric_columns(d, exclude_binary=False, exclude_constant=False),
        )
        if not cols:
            return cast(VarianceThresholdArtifact, {})

        selector = VarianceThreshold(threshold=threshold)
        X_np, _ = SklearnBridge.to_sklearn(X_pd[cols])
        selector.fit(X_np)

        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support) if s]
        variances = (
            dict(zip(cols, selector.variances_.tolist())) if hasattr(selector, "variances_") else {}
        )
        return cast(
            VarianceThresholdArtifact,
            {
                "type": "variance_threshold",
                "selected_columns": selected_cols,
                "candidate_columns": cols,
                "threshold": threshold,
                "drop_columns": drop_columns,
                "variances": variances,
            },
        )


# --- Correlation Threshold ---


def _corr_drop_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Polars apply path: drop the precomputed ``columns_to_drop`` list."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = [c for c in params.get("columns_to_drop", []) if c in X.columns]
    if to_drop:
        X = X.drop(to_drop)
    return X, y


def _corr_drop_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Pandas apply path: drop the precomputed ``columns_to_drop`` list."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = [c for c in params.get("columns_to_drop", []) if c in X.columns]
    if to_drop:
        X = X.drop(columns=to_drop)
    return X, y


class CorrelationThresholdApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _corr_drop_polars, _corr_drop_pandas)


@NodeRegistry.register("CorrelationThreshold", CorrelationThresholdApplier)
@node_meta(
    id="CorrelationThreshold",
    name="Correlation Threshold",
    category="Feature Selection",
    description="Remove features highly correlated with others.",
    params={"threshold": 0.95, "method": "pearson"},
)
class CorrelationThresholdCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> CorrelationThresholdArtifact:
        X_pd = to_pandas(X)

        threshold = config.get("threshold", 0.95)
        drop_columns = config.get("drop_columns", True)
        # Prefer "correlation_method" — falling back to "method" can collide with the
        # facade's own "method" key (e.g. "correlation_threshold").
        method = config.get("correlation_method", "pearson")

        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        if len(cols) < 2:
            return cast(CorrelationThresholdArtifact, {})

        corr_matrix = X_pd[cols].corr(method=method).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        return cast(
            CorrelationThresholdArtifact,
            {
                "type": "correlation_threshold",
                "columns_to_drop": to_drop,
                "threshold": threshold,
                "method": method,
                "drop_columns": drop_columns,
            },
        )


# --- Univariate Selection ---


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


# --- Model Based Selection ---


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


# --- Unified Feature Selection (Facade) ---
class FeatureSelectionApplier(BaseApplier):
    def apply(
        self,
        df: Any,
        params: Dict[str, Any],
    ) -> Any:
        # The params returned by the specific calculator carry a "type" tag
        # that selects the right concrete applier.
        type_name = params.get("type")

        applier: Optional[BaseApplier] = None
        if type_name == "variance_threshold":
            applier = VarianceThresholdApplier()
        elif type_name == "correlation_threshold":
            applier = CorrelationThresholdApplier()
        elif type_name == "univariate_selection":
            applier = UnivariateSelectionApplier()
        elif type_name == "model_based_selection":
            applier = ModelBasedSelectionApplier()

        if applier:
            return applier.apply(df, params)
        # Identity passthrough when no concrete applier matches.
        return df


_FS_CALCULATORS: Dict[str, Callable[[], BaseCalculator]] = {
    "variance_threshold": VarianceThresholdCalculator,
    "correlation_threshold": CorrelationThresholdCalculator,
    "select_k_best": UnivariateSelectionCalculator,
    "select_percentile": UnivariateSelectionCalculator,
    "generic_univariate_select": UnivariateSelectionCalculator,
    "select_fpr": UnivariateSelectionCalculator,
    "select_fdr": UnivariateSelectionCalculator,
    "select_fwe": UnivariateSelectionCalculator,
    "select_from_model": ModelBasedSelectionCalculator,
    "rfe": ModelBasedSelectionCalculator,
}


@NodeRegistry.register("feature_selection", FeatureSelectionApplier)
@node_meta(
    id="feature_selection",
    name="Feature Selection (Wrapper)",
    category="Feature Selection",
    description="General wrapper for feature selection strategies.",
    params={"method": "variance", "threshold": 0.0},
)
class FeatureSelectionCalculator(BaseCalculator):
    def fit(
        self,
        df: Any,
        config: Dict[str, Any],
    ) -> Mapping[str, Any]:
        method = config.get("method", "select_k_best")
        ctor = _FS_CALCULATORS.get(method)
        if ctor is None:
            logger.warning(f"Unknown feature selection method: {method}")
            return {}
        return ctor().fit(df, config)
