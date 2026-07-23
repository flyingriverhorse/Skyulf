"""Shared helpers for feature-selection nodes."""

import contextlib
import logging
from collections.abc import Callable
from typing import Any, cast

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
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from ...utils import detect_numeric_columns, resolve_columns
from .._artifacts import UnivariateSelectionArtifact

logger = logging.getLogger(__name__)

SCORE_FUNCTIONS: dict[str, Callable] = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
    "chi2": chi2,
    "r_regression": r_regression,
}


# Heuristic threshold: a numeric target with this many or fewer distinct
# values is treated as classification (e.g. a small integer-coded label).
# This is a coarse heuristic, not a config knob - a genuine regression
# target that happens to take <= this many distinct values (e.g. a discrete
# count) will be misclassified as classification. Logged at inference time
# so this silent assumption is at least visible in diagnostics.
_MAX_UNIQUE_VALUES_FOR_CLASSIFICATION = 10


def _infer_problem_type(series: pd.Series) -> str:
    if series.empty:
        return "classification"
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
        return "classification"
    unique_values = series.dropna().unique()
    if len(unique_values) <= _MAX_UNIQUE_VALUES_FOR_CLASSIFICATION:
        logger.debug(
            "Inferred problem_type='classification' for target with %d distinct "
            "numeric values (<= %d cutoff heuristic); pass problem_type explicitly "
            "if this is actually a regression target.",
            len(unique_values),
            _MAX_UNIQUE_VALUES_FOR_CLASSIFICATION,
        )
        return "classification"
    return "regression"


def _resolve_score_function(name: str | None, problem_type: str) -> Any:
    if name and name in SCORE_FUNCTIONS:
        return SCORE_FUNCTIONS[name]

    if problem_type == "classification":
        return f_classif
    return f_regression


def _resolve_estimator(key: str | None, problem_type: str) -> Any:
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


def _resolve_drop_list(params: dict[str, Any], existing_cols: list[str]) -> list[str]:
    """Compute the column-drop list from selected/candidate params."""
    selected = params.get("selected_columns")
    candidates = params.get("candidate_columns", [])
    if selected is None:
        return []
    return [c for c in (set(candidates) - set(selected)) if c in existing_cols]


def _drop_selected_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Polars apply path for selectors that drop ``candidate \\ selected`` columns."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = _resolve_drop_list(params, list(X.columns))
    if to_drop:
        X = X.drop(to_drop)
    return X, y


def _drop_selected_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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


def _extract_target(X_pd: pd.DataFrame, y: Any, target_col: str | None) -> pd.Series | None:
    """Return ``y`` if provided; else pull ``target_col`` from the (pandas) frame.

    ``y`` may arrive as a Polars Series when the pipeline runs on the Polars
    engine; the rest of this module (``_infer_problem_type``, ``pd.factorize``,
    etc.) assumes a pandas Series, so normalize here rather than in every caller.
    """
    if y is not None:
        return y.to_pandas() if hasattr(y, "to_pandas") else y
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


_GENERIC_PARAM_KEYS: dict[str, tuple[str, Any]] = {
    "k_best": ("k", 10),
    "percentile": ("percentile", 10),
}


def _resolve_generic_param(config: dict[str, Any]) -> Any:
    """Pick the GenericUnivariateSelect ``param`` from explicit or mode-derived config."""
    if "param" in config:
        return config.get("param")
    mode = config.get("mode", "k_best")
    key, default = _GENERIC_PARAM_KEYS.get(mode, ("alpha", 0.05))
    return config.get(key, default)


_UNIVARIATE_SELECTOR_BUILDERS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
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

# The node's documented default (``params={"method": "SelectKBest", ...}`` in
# @node_meta) uses sklearn's own PascalCase class names, but the builder dict
# above is keyed by snake_case aliases. Without this alias map, the
# documented default silently no-ops (unknown method -> no selector built).
_UNIVARIATE_METHOD_ALIASES: dict[str, str] = {
    "selectkbest": "select_k_best",
    "selectpercentile": "select_percentile",
    "selectfpr": "select_fpr",
    "selectfdr": "select_fdr",
    "selectfwe": "select_fwe",
    "genericunivariateselect": "generic_univariate_select",
}


def _normalize_univariate_method(method: str) -> str:
    """Map either snake_case or sklearn PascalCase method names to the internal key."""
    key = method.lower().replace("_", "")
    return _UNIVARIATE_METHOD_ALIASES.get(key, method)


def _build_univariate_selector(method: str, score_func: Any, config: dict[str, Any]) -> Any | None:
    """Construct the sklearn univariate selector named by ``method``."""
    builder = _UNIVARIATE_SELECTOR_BUILDERS.get(_normalize_univariate_method(method))
    return builder(score_func, config) if builder else None


def _build_model_selector(method: str, estimator: Any, config: dict[str, Any]) -> Any | None:
    """Construct the sklearn model-based selector named by ``method``."""
    if method == "select_from_model":
        threshold = config.get("threshold", "mean")
        if isinstance(threshold, str):
            # Keep as string (e.g. "mean", "1.25*mean")
            with contextlib.suppress(ValueError):
                threshold = float(threshold)
        return SelectFromModel(
            estimator=estimator,
            threshold=threshold,
            max_features=config.get("max_features"),
        )
    if method == "rfe":
        return RFE(
            estimator=estimator,
            n_features_to_select=config.get("n_features_to_select"),
            step=config.get("step", 1),
        )
    return None


def _fillna_zero_with_warning(X_pd: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill missing values with 0 before scoring, warning when this actually
    changes data (unlike ``_maybe_chi2_rescale``'s already-existing warning
    pattern, a silent ``fillna(0)`` can bias univariate/model-based feature
    scores whenever 0 is itself a meaningful value, or missingness is
    correlated with the target)."""
    subset = X_pd[cols]
    if subset.isna().any().any():
        logger.warning(
            "Feature selection: filling missing values with 0 before scoring "
            "for columns %s. This may bias scores/importances if 0 is a "
            "meaningful value for these columns or if missingness itself "
            "correlates with the target; consider imputing upstream instead.",
            [c for c in cols if subset[c].isna().any()],
        )
    return subset.fillna(0)


def _maybe_chi2_rescale(X_np: np.ndarray, score_func_name: str | None) -> np.ndarray:
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
    X_pd: pd.DataFrame, config: dict[str, Any], target_col: str | None
) -> list[str]:
    """Return numeric candidate columns minus the target."""
    cols = resolve_columns(X_pd, config, lambda d: detect_numeric_columns(d, exclude_binary=False))
    return [c for c in cols if c != target_col]


def _univariate_score_dicts(
    selector: Any, cols: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Pull (scores, pvalues) off a fitted univariate selector, NaN-safe."""
    scores: dict[str, float] = {}
    pvalues: dict[str, float] = {}
    if hasattr(selector, "scores_"):
        safe_scores = np.nan_to_num(selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        scores = dict(zip(cols, safe_scores.tolist(), strict=True))
    if hasattr(selector, "pvalues_"):
        safe_pvalues = np.nan_to_num(cast(Any, selector.pvalues_), nan=1.0)
        pvalues = dict(zip(cols, safe_pvalues.tolist(), strict=True))
    return scores, pvalues


def _univariate_no_target_artifact(
    cols: list[str], method: str, config: dict[str, Any]
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


def _rfe_selected_names(selector: Any, cols: list[str]) -> list[str]:
    """Narrow `cols` to the RFE-selected subset per `support_`, if available."""
    if isinstance(selector, RFE):
        support = getattr(selector, "support_", None)
        if support is not None:
            return [c for c, s in zip(cols, support, strict=True) if s]
    return cols


def _importances_from_estimator(estimator: Any, names: list[str]) -> dict[str, float]:
    """Extract feature_importances_ or |coef_| from a fitted estimator, keyed by name."""
    if hasattr(estimator, "feature_importances_"):
        return dict(zip(names, estimator.feature_importances_.tolist(), strict=True))
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return dict(zip(names, np.abs(coef).tolist(), strict=True))
    return {}


def _model_feature_importances(selector: Any, cols: list[str]) -> dict[str, float]:
    """Pull importances or |coef| off a fitted model-based selector."""
    estimator = getattr(selector, "estimator_", None)
    if estimator is None:
        return {}
    # For RFE, `estimator_` is refit on only the surviving feature subset, so
    # its importances/coefs align with `support_`-selected columns, not the
    # full candidate list. For SelectFromModel (and anything else exposing
    # `support_`), the estimator is fit on the full candidate set, so no
    # narrowing is needed.
    names = _rfe_selected_names(selector, cols)
    return _importances_from_estimator(estimator, names)
