"""Feature selection utilities for Skyulf-MLFlow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...core.base import BaseTransformer
from ...core.exceptions import FeatureEngineeringError
from ...core.types import TransformerState

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency guard
    from sklearn.feature_selection import (
        GenericUnivariateSelect,
        RFE,
        SelectFdr,
        SelectFpr,
        SelectFwe,
        SelectFromModel,
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
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except Exception:  # pragma: no cover - defensive guard
    GenericUnivariateSelect = None  # type: ignore[assignment]
    RFE = None  # type: ignore[assignment]
    SelectFdr = None  # type: ignore[assignment]
    SelectFpr = None  # type: ignore[assignment]
    SelectFwe = None  # type: ignore[assignment]
    SelectFromModel = None  # type: ignore[assignment]
    SelectKBest = None  # type: ignore[assignment]
    SelectPercentile = None  # type: ignore[assignment]
    VarianceThreshold = None  # type: ignore[assignment]
    chi2 = None  # type: ignore[assignment]
    f_classif = None  # type: ignore[assignment]
    f_regression = None  # type: ignore[assignment]
    mutual_info_classif = None  # type: ignore[assignment]
    mutual_info_regression = None  # type: ignore[assignment]
    r_regression = None  # type: ignore[assignment]
    LinearRegression = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    RandomForestClassifier = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment]


SUPPORTED_METHODS: Tuple[str, ...] = (
    "select_k_best",
    "select_percentile",
    "generic_univariate_select",
    "select_fpr",
    "select_fdr",
    "select_fwe",
    "select_from_model",
    "variance_threshold",
    "rfe",
)
DEFAULT_K_FALLBACK = 10
DEFAULT_PERCENTILE = 10.0
DEFAULT_ALPHA = 0.05
DEFAULT_THRESHOLD = 0.0

SCORE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
    "chi2": chi2,
    "r_regression": r_regression,
}


@dataclass
class FeatureSelectionStatistics:
    """Summaries for each evaluated feature."""

    column: str
    selected: bool
    score: Optional[float] = None
    p_value: Optional[float] = None
    rank: Optional[int] = None
    importance: Optional[float] = None
    note: Optional[str] = None


class FeatureSelector(BaseTransformer):
    """Wrapper around scikit-learn feature selection strategies."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        auto_detect: bool = True,
        method: str = "select_k_best",
        score_func: Optional[str] = None,
        mode: Optional[str] = None,
        problem_type: str = "auto",
        target_column: Optional[str] = None,
        k: Optional[int] = None,
        percentile: Optional[float] = None,
        alpha: Optional[float] = None,
        threshold: Optional[float] = None,
        drop_unselected: bool = True,
        estimator: Optional[str] = None,
        step: Optional[float] = None,
        min_features: Optional[int] = None,
        max_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(columns=columns, **kwargs)
        self.auto_detect = bool(auto_detect)
        self.method = (method or "select_k_best").strip().lower()
        self.score_func = score_func.strip().lower() if isinstance(score_func, str) else None
        self.mode = mode.strip().lower() if isinstance(mode, str) else None
        self.problem_type = problem_type.strip().lower() if isinstance(problem_type, str) else "auto"
        self.target_column = target_column
        self.k = _coerce_optional_int(k)
        self.percentile = _coerce_optional_float(percentile)
        self.alpha = _coerce_optional_float(alpha) if alpha is not None else DEFAULT_ALPHA
        self.threshold = _coerce_optional_float(threshold)
        self.drop_unselected = bool(drop_unselected)
        self.estimator = estimator.strip().lower() if isinstance(estimator, str) else None
        self.step = _coerce_optional_float(step)
        self.min_features = _coerce_optional_int(min_features)
        self.max_features = _coerce_optional_int(max_features)

        if self.method not in SUPPORTED_METHODS:
            self.method = "select_k_best"
        if self.mode not in {"k_best", "percentile", "fpr", "fdr", "fwe", None}:
            self.mode = None
        if self.problem_type not in {"classification", "regression", "auto"}:
            self.problem_type = "auto"

        self.selector_: Any = None
        self.candidate_columns_: List[str] = []
        self.selected_columns_: List[str] = []
        self.dropped_columns_: List[str] = []
        self.statistics_: List[FeatureSelectionStatistics] = []

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        if SelectKBest is None:
            raise FeatureEngineeringError(
                "scikit-learn feature selection modules are not available. "
                "Install scikit-learn to use FeatureSelector."
            )

        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")

        target_series = self._resolve_target_series(X, y)

        candidate_columns = self._resolve_candidate_columns(X)
        if not candidate_columns:
            raise FeatureEngineeringError("FeatureSelector requires at least one candidate column")

        numeric_frame = X[candidate_columns].apply(pd.to_numeric, errors="coerce")
        if numeric_frame.dropna(how="all").empty:
            raise FeatureEngineeringError("Candidate columns do not contain numeric data")

        if self.method != "variance_threshold" and target_series is None:
            raise FeatureEngineeringError("FeatureSelector requires a target column for the selected method")

        if self.method == "variance_threshold":
            prepared_y = None
            y_metadata: Dict[str, Any] = {}
            problem_type = "regression"
        else:
            inferred_problem = self._infer_problem_type(target_series)
            prepared_y, y_metadata = _prepare_target_series(target_series, inferred_problem)
            problem_type = inferred_problem

        score_func, score_name = _resolve_score_function(self.score_func, problem_type)
        selector, method_label = _build_selector(
            method=self.method,
            problem_type=problem_type,
            config=self,
            score_function=score_func,
            score_name=score_name,
        )
        if selector is None:
            raise FeatureEngineeringError("Unsupported feature selection configuration or missing dependencies")

        if score_name == "chi2":
            numeric_frame = _sanitize_chi2_input(numeric_frame.fillna(0.0))
        else:
            numeric_frame = numeric_frame.fillna(0.0)

        if self.method == "variance_threshold":
            X_fit = numeric_frame.fillna(0.0)
            y_fit = None
        else:
            X_fit, y_fit = _prepare_matrix_and_target(numeric_frame, prepared_y, problem_type)

        try:
            if y_fit is not None and self.method not in {"variance_threshold"}:
                selector.fit(X_fit, y_fit)
            else:
                selector.fit(X_fit)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Feature selection fit failed", exc_info=exc)
            raise FeatureEngineeringError(f"Feature selection fit failed: {exc}") from exc

        metadata, summaries = _build_metadata(
            selector=selector,
            config=self,
            candidate_columns=candidate_columns,
            method_label=method_label,
            score_name=score_name,
            target_metadata=y_metadata,
        )

        self.selector_ = selector
        self.metadata = metadata
        self.candidate_columns_ = candidate_columns
        self.selected_columns_ = metadata.get("selected_columns", [])
        self.dropped_columns_ = metadata.get("dropped_columns", [])
        self.statistics_ = summaries
        self.state = TransformerState.FITTED
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()

        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")

        self._validate_columns_present(X, self.candidate_columns_)

        if not self.dropped_columns_ or not self.drop_unselected:
            return X.copy()

        remaining = [col for col in X.columns if col not in self.dropped_columns_]
        return X[remaining].copy()

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return super().fit_transform(X, y)

    # ------------------------------------------------------------------
    def get_support(self) -> List[bool]:
        self._check_is_fitted()
        support = self.metadata.get("support_mask")
        if isinstance(support, Iterable):
            return [bool(value) for value in support]
        return [col in self.selected_columns_ for col in self.candidate_columns_]

    # ------------------------------------------------------------------
    def get_selected_columns(self) -> List[str]:
        self._check_is_fitted()
        return list(self.selected_columns_)

    # ------------------------------------------------------------------
    def get_dropped_columns(self) -> List[str]:
        self._check_is_fitted()
        return list(self.dropped_columns_)

    # ------------------------------------------------------------------
    def get_feature_summaries(self) -> List[FeatureSelectionStatistics]:
        self._check_is_fitted()
        return list(self.statistics_)

    # ------------------------------------------------------------------
    def _validate_columns_present(self, X: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = [col for col in columns if col not in X.columns]
        if missing:
            raise FeatureEngineeringError(
                f"Columns not found during transform: {', '.join(missing[:5])}" + ("..." if len(missing) > 5 else "")
            )

    # ------------------------------------------------------------------
    def _resolve_candidate_columns(self, X: pd.DataFrame) -> List[str]:
        explicit = []
        if self.columns:
            explicit.extend(self.columns)

        candidate_columns: List[str] = []
        seen = set()

        for column in explicit:
            if column in X.columns and column not in seen:
                seen.add(column)
                candidate_columns.append(column)

        if self.auto_detect:
            for column in _auto_detect_numeric_columns(X):
                if column not in seen and column != self.target_column:
                    seen.add(column)
                    candidate_columns.append(column)
        return candidate_columns

    # ------------------------------------------------------------------
    def _resolve_target_series(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Optional[pd.Series]:
        if self.method == "variance_threshold":
            return None

        if y is not None:
            return pd.Series(y, index=X.index)

        if not self.target_column:
            return None
        if self.target_column not in X.columns:
            raise FeatureEngineeringError(
                f"Target column '{self.target_column}' not found in input dataframe"
            )
        return X[self.target_column]

    # ------------------------------------------------------------------
    def _infer_problem_type(self, target: Optional[pd.Series]) -> str:
        if self.problem_type and self.problem_type != "auto":
            return self.problem_type
        if target is None or target.empty:
            return "classification"
        if pd.api.types.is_bool_dtype(target) or pd.api.types.is_object_dtype(target):
            return "classification"
        unique_values = target.dropna().unique()
        if len(unique_values) <= 10:
            return "classification"
        return "regression"


# ======================================================================
def _auto_detect_numeric_columns(frame: pd.DataFrame) -> List[str]:
    return frame.select_dtypes(include=["number", "bool"]).columns.tolist()


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        if numeric <= 0:
            return None
        return int(round(numeric))
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return int(round(numeric))


def _prepare_target_series(target: Optional[pd.Series], problem_type: str) -> Tuple[pd.Series, Dict[str, Any]]:
    if target is None:
        raise FeatureEngineeringError("Target series is required for supervised feature selection")

    metadata: Dict[str, Any] = {}
    if problem_type == "classification":
        factorized, uniques = pd.factorize(target, sort=True)
        metadata["class_labels"] = [str(item) for item in uniques]
        encoded = pd.Series(factorized, index=target.index)
        return encoded, metadata

    numeric = pd.to_numeric(target, errors="coerce")
    return numeric, metadata


def _prepare_matrix_and_target(
    matrix: pd.DataFrame,
    target: pd.Series,
    problem_type: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    X_fit = matrix.fillna(0.0)
    if problem_type == "classification":
        valid_mask = pd.notna(target) & (target != -1)
    else:
        valid_mask = pd.notna(target)

    X_fit = X_fit.loc[valid_mask]
    y_fit = target.loc[valid_mask]
    if y_fit.empty:
        raise FeatureEngineeringError("Target series became empty after removing NaNs")
    return X_fit, y_fit


def _resolve_score_function(name: Optional[str], problem_type: str) -> Tuple[Optional[Callable[..., Any]], str]:
    if name and name in SCORE_FUNCTIONS and SCORE_FUNCTIONS[name] is not None:
        return SCORE_FUNCTIONS[name], name
    if problem_type == "classification" and SCORE_FUNCTIONS.get("f_classif") is not None:
        return SCORE_FUNCTIONS["f_classif"], "f_classif"
    if problem_type == "regression" and SCORE_FUNCTIONS.get("f_regression") is not None:
        return SCORE_FUNCTIONS["f_regression"], "f_regression"
    fallback = "mutual_info_classif" if problem_type == "classification" else "mutual_info_regression"
    if SCORE_FUNCTIONS.get(fallback) is not None:
        return SCORE_FUNCTIONS[fallback], fallback
    return None, name or ""


def _resolve_estimator(key: Optional[str], problem_type: str) -> Tuple[Optional[Any], Optional[str]]:
    label = None
    choice = (key or "auto").lower()

    if problem_type == "classification":
        if choice in {"auto", "logistic_regression"} and LogisticRegression is not None:
            label = "logistic_regression"
            return LogisticRegression(max_iter=1000), label
        if choice in {"random_forest", "auto"} and RandomForestClassifier is not None:
            label = "random_forest_classifier"
            return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1), label
        if choice == "linear_regression" and LinearRegression is not None:
            label = "linear_regression"
            return LinearRegression(), label
    else:
        if choice in {"auto", "linear_regression"} and LinearRegression is not None:
            label = "linear_regression"
            return LinearRegression(), label
        if choice in {"random_forest", "auto"} and RandomForestRegressor is not None:
            label = "random_forest_regressor"
            return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1), label
    return None, label


def _sanitize_chi2_input(matrix: pd.DataFrame) -> pd.DataFrame:
    min_value = matrix.min().min()
    if pd.notna(min_value) and min_value < 0:
        return matrix - float(min_value)
    return matrix


def _build_selector(
    *,
    method: str,
    problem_type: str,
    config: FeatureSelector,
    score_function: Optional[Callable[..., Any]],
    score_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if method == "variance_threshold":
        if VarianceThreshold is None:
            return None, None
        threshold = config.threshold if config.threshold is not None else DEFAULT_THRESHOLD
        method_label = f"VarianceThreshold (threshold={threshold})"
        return VarianceThreshold(threshold=threshold), method_label

    if method == "select_from_model":
        if SelectFromModel is None:
            return None, None
        estimator, estimator_label = _resolve_estimator(config.estimator, problem_type)
        if estimator is None:
            return None, None
        threshold = config.threshold if config.threshold is not None else "median"
        method_label = f"SelectFromModel ({estimator_label or 'estimator'}, threshold={threshold})"
        selector = SelectFromModel(estimator=estimator, threshold=threshold)
        return selector, method_label

    if method == "rfe":
        if RFE is None:
            return None, None
        estimator, estimator_label = _resolve_estimator(config.estimator, problem_type)
        if estimator is None:
            return None, None
        n_features = config.k
        step = config.step if config.step is not None else 1
        method_label = f"RFE ({estimator_label or 'estimator'}, n_features={n_features or 'auto'})"
        selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
        return selector, method_label

    if score_function is None:
        return None, None

    if method == "select_k_best":
        if SelectKBest is None:
            return None, None
        k_value = config.k if config.k is not None else DEFAULT_K_FALLBACK
        method_label = f"SelectKBest ({score_name}, k={k_value})"
        selector = SelectKBest(score_func=score_function, k=k_value)
        return selector, method_label

    if method == "select_percentile":
        if SelectPercentile is None:
            return None, None
        percentile = config.percentile if config.percentile is not None else DEFAULT_PERCENTILE
        method_label = f"SelectPercentile ({score_name}, {percentile}%)"
        selector = SelectPercentile(score_func=score_function, percentile=percentile)
        return selector, method_label

    if method in {"select_fpr", "select_fdr", "select_fwe"}:
        alpha = config.alpha if config.alpha is not None else DEFAULT_ALPHA
        if method == "select_fpr" and SelectFpr is not None:
            method_label = f"SelectFpr ({score_name}, alpha={alpha})"
            return SelectFpr(score_func=score_function, alpha=alpha), method_label
        if method == "select_fdr" and SelectFdr is not None:
            method_label = f"SelectFdr ({score_name}, alpha={alpha})"
            return SelectFdr(score_func=score_function, alpha=alpha), method_label
        if method == "select_fwe" and SelectFwe is not None:
            method_label = f"SelectFwe ({score_name}, alpha={alpha})"
            return SelectFwe(score_func=score_function, alpha=alpha), method_label
        return None, None

    if method == "generic_univariate_select":
        if GenericUnivariateSelect is None:
            return None, None
        mode = config.mode or "k_best"
        if mode == "k_best":
            param = config.k if config.k is not None else DEFAULT_K_FALLBACK
        elif mode == "percentile":
            param = config.percentile if config.percentile is not None else DEFAULT_PERCENTILE
        else:
            param = config.alpha if config.alpha is not None else DEFAULT_ALPHA
        method_label = f"GenericUnivariateSelect ({score_name}, mode={mode}, param={param})"
        selector = GenericUnivariateSelect(score_func=score_function, mode=mode, param=param)
        return selector, method_label

    return None, None


def _build_metadata(
    *,
    selector: Any,
    config: FeatureSelector,
    candidate_columns: Sequence[str],
    method_label: Optional[str],
    score_name: str,
    target_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[FeatureSelectionStatistics]]:
    support = selector.get_support() if hasattr(selector, "get_support") else [True] * len(candidate_columns)

    selected_columns = [str(col) for col, keep in zip(candidate_columns, support) if keep]
    dropped_columns = [str(col) for col, keep in zip(candidate_columns, support) if not keep]

    scores = getattr(selector, "scores_", None)
    p_values = getattr(selector, "pvalues_", None)
    ranking = getattr(selector, "ranking_", None)

    importances = None
    if hasattr(selector, "estimator_"):
        estimator = getattr(selector, "estimator_")
        if hasattr(estimator, "feature_importances_"):
            importances = getattr(estimator, "feature_importances_")
        elif hasattr(estimator, "coef_"):
            coef = getattr(estimator, "coef_")
            if isinstance(coef, np.ndarray):
                importances = np.abs(coef) if coef.ndim == 1 else np.abs(coef).mean(axis=0)

    notes: Dict[str, str] = {}
    if config.method == "variance_threshold" and hasattr(selector, "variances_"):
        variances = getattr(selector, "variances_")
        for column, variance in zip(candidate_columns, variances):
            notes[str(column)] = f"variance={_safe_float(variance) if variance is not None else 'NA'}"

    summaries = _build_feature_summaries(
        columns=candidate_columns,
        support=support,
        scores=scores,
        p_values=p_values,
        ranking=ranking,
        importances=importances,
        notes=notes,
    )

    metadata: Dict[str, Any] = {
        "candidate_columns": [str(col) for col in candidate_columns],
        "selected_columns": selected_columns,
        "dropped_columns": dropped_columns,
        "support_mask": [bool(value) for value in support],
        "scores": [_safe_float(value) for value in scores] if scores is not None else None,
        "p_values": [_safe_float(value) for value in p_values] if p_values is not None else None,
        "ranking": [_safe_int(value) for value in ranking] if ranking is not None else None,
        "method": config.method,
        "method_label": method_label,
        "score_func": score_name,
        "mode": config.mode,
        "estimator": config.estimator,
        "problem_type": config.problem_type,
        "target_column": config.target_column,
        "k": config.k,
        "percentile": config.percentile,
        "alpha": config.alpha,
        "threshold": config.threshold,
        "drop_unselected": config.drop_unselected,
        "summary": f"Kept {len(selected_columns)} of {len(candidate_columns)} columns",
    }

    if summaries:
        metadata["feature_summaries"] = [summary.__dict__ for summary in summaries]
    if config.method == "select_from_model" and hasattr(selector, "estimator_"):
        metadata.setdefault("estimator_params", getattr(selector.estimator_, "get_params", lambda: {})())
    if target_metadata:
        metadata["target_metadata"] = target_metadata

    return metadata, summaries


def _build_feature_summaries(
    *,
    columns: Sequence[str],
    support: Sequence[bool],
    scores: Optional[Sequence[Any]] = None,
    p_values: Optional[Sequence[Any]] = None,
    ranking: Optional[Sequence[Any]] = None,
    importances: Optional[Sequence[Any]] = None,
    notes: Optional[Dict[str, str]] = None,
) -> List[FeatureSelectionStatistics]:
    summaries: List[FeatureSelectionStatistics] = []
    for idx, column in enumerate(columns):
        selected = bool(support[idx]) if idx < len(support) else False
        score = _safe_float(scores[idx]) if scores is not None and idx < len(scores) else None
        p_val = _safe_float(p_values[idx]) if p_values is not None and idx < len(p_values) else None
        rank = _safe_int(ranking[idx]) if ranking is not None and idx < len(ranking) else None
        importance = _safe_float(importances[idx]) if importances is not None and idx < len(importances) else None
        note = notes.get(str(column)) if notes else None
        summaries.append(
            FeatureSelectionStatistics(
                column=str(column),
                selected=selected,
                score=score,
                p_value=p_val,
                rank=rank,
                importance=importance,
                note=note,
            )
        )
    return summaries


__all__ = ["FeatureSelector", "FeatureSelectionStatistics"]
