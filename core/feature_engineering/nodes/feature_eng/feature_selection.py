"""Feature selection node helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
from core.feature_engineering.schemas import (
    FeatureSelectionFeatureSummary,
    FeatureSelectionNodeSignal,
)
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store

from .utils import (
    _auto_detect_numeric_columns,
    _coerce_config_boolean,
    _coerce_string_list,
)

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

logger = logging.getLogger(__name__)


def _check_f_classif_zero_variance(X: np.ndarray, y: np.ndarray) -> Tuple[bool, List[int]]:
    """
    Check if any features have zero within-class variance, which causes
    divide-by-zero warnings in f_classif.

    Returns:
        (has_zero_variance, problematic_feature_indices)
    """
    if X.shape[0] == 0 or X.shape[1] == 0:
        return False, []

    problematic_features = []
    unique_classes = np.unique(y)

    if len(unique_classes) < 2:
        return False, []

    for feature_idx in range(X.shape[1]):
        feature_values = X[:, feature_idx]

        # Check variance within each class
        has_zero_var_in_class = False
        for class_label in unique_classes:
            class_mask = (y == class_label)
            class_values = feature_values[class_mask]

            if len(class_values) > 0:
                # Check if all values are identical (zero variance)
                if np.var(class_values) == 0.0:
                    has_zero_var_in_class = True
                    break

        if has_zero_var_in_class:
            problematic_features.append(feature_idx)

    return len(problematic_features) > 0, problematic_features


TRANSFORMER_NAME = "feature_selection"
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
class NormalizedFeatureSelectionConfig:
    columns: List[str]
    auto_detect: bool
    method: str
    score_func: Optional[str]
    mode: Optional[str]
    problem_type: Optional[str]
    target_column: Optional[str]
    k: Optional[int]
    percentile: Optional[float]
    alpha: Optional[float]
    threshold: Optional[float]
    drop_unselected: bool
    estimator: Optional[str]
    step: Optional[float]
    min_features: Optional[int]
    max_features: Optional[int]


@dataclass
class SelectionInputs:
    config: NormalizedFeatureSelectionConfig
    numeric_frame: pd.DataFrame
    valid_columns: List[str]
    target_vector: Optional[pd.Series]
    problem_type: str
    score_name: str
    selector: Any
    method_label: Optional[str]
    y_metadata: Dict[str, Any]


@dataclass
class SelectionExecutionResult:
    metadata: Dict[str, Any]
    feature_summaries: List[FeatureSelectionFeatureSummary]
    transform_mode: str
    storage: Optional[Any]
    split_counts: Dict[str, int]
    method_label: Optional[str]


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        converted_float = float(value)
        if np.isnan(converted_float):
            return None
        return int(round(converted_float))
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        converted = float(value)
        if np.isnan(converted):  # type: ignore[arg-type]
            return None
        return converted
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort conversion that drops non-finite values."""

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


def _normalize_feature_selection_config(raw_config: Any) -> NormalizedFeatureSelectionConfig:
    if not isinstance(raw_config, dict):
        raw_config = {}

    columns = _coerce_string_list(raw_config.get("columns"))
    auto_detect = _coerce_config_boolean(raw_config.get("auto_detect"), default=True)

    method = str(raw_config.get("method") or "select_k_best").strip().lower()
    if method not in SUPPORTED_METHODS:
        method = "select_k_best"

    score_func = raw_config.get("score_func")
    score_func_name = str(score_func).strip().lower() if isinstance(score_func, str) else None
    if score_func_name and score_func_name not in SCORE_FUNCTIONS:
        score_func_name = None

    mode = raw_config.get("mode")
    mode_name = str(mode).strip().lower() if isinstance(mode, str) else None
    if mode_name not in {"k_best", "percentile", "fpr", "fdr", "fwe", None}:
        mode_name = None

    problem_type = raw_config.get("problem_type")
    problem_type_name = str(problem_type).strip().lower() if isinstance(problem_type, str) else None
    if problem_type_name not in {"classification", "regression", "auto", None}:
        problem_type_name = None

    target_column = str(raw_config.get("target_column") or "").strip() or None

    k_value = _coerce_optional_int(raw_config.get("k"))
    if k_value is not None and k_value <= 0:
        k_value = None

    percentile_value = _coerce_optional_float(raw_config.get("percentile"))
    if percentile_value is not None:
        percentile_value = max(0.0, min(100.0, percentile_value))

    alpha_value = _coerce_optional_float(raw_config.get("alpha")) or DEFAULT_ALPHA
    threshold_value = _coerce_optional_float(raw_config.get("threshold"))

    drop_unselected = _coerce_config_boolean(raw_config.get("drop_unselected"), default=True)

    estimator = raw_config.get("estimator")
    estimator_name = str(estimator).strip().lower() if isinstance(estimator, str) else None

    step_value = _coerce_optional_float(raw_config.get("step"))
    if step_value is not None and step_value <= 0:
        step_value = None

    min_features = _coerce_optional_int(raw_config.get("min_features"))
    max_features = _coerce_optional_int(raw_config.get("max_features"))

    return NormalizedFeatureSelectionConfig(
        columns=columns,
        auto_detect=auto_detect,
        method=method,
        score_func=score_func_name,
        mode=mode_name,
        problem_type=problem_type_name or "auto",
        target_column=target_column,
        k=k_value,
        percentile=percentile_value,
        alpha=alpha_value,
        threshold=threshold_value,
        drop_unselected=drop_unselected,
        estimator=estimator_name,
        step=step_value,
        min_features=min_features,
        max_features=max_features,
    )


def _infer_problem_type(series: pd.Series, configured: Optional[str]) -> str:
    if configured and configured != "auto":
        return configured
    if series.empty:
        return "classification"
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
        return "classification"
    unique_values = series.dropna().unique()
    if len(unique_values) <= 10:
        return "classification"
    return "regression"


def _prepare_target(series: pd.Series, problem_type: str) -> Tuple[pd.Series, Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    if problem_type == "classification":
        factorized, uniques = pd.factorize(series, sort=True)
        metadata["class_labels"] = [str(item) for item in uniques]
        return pd.Series(factorized, index=series.index), metadata
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric, metadata


def _resolve_score_function(name: Optional[str], problem_type: str) -> Tuple[Optional[Callable[..., Any]], str]:
    if name and name in SCORE_FUNCTIONS and SCORE_FUNCTIONS[name] is not None:
        return SCORE_FUNCTIONS[name], name
    if problem_type == "classification":
        if SCORE_FUNCTIONS.get("f_classif") is not None:
            return SCORE_FUNCTIONS["f_classif"], "f_classif"
    else:
        if SCORE_FUNCTIONS.get("f_regression") is not None:
            return SCORE_FUNCTIONS["f_regression"], "f_regression"
    # fallback to mutual information if available
    fallback = "mutual_info_classif" if problem_type == "classification" else "mutual_info_regression"
    if SCORE_FUNCTIONS.get(fallback) is not None:
        return SCORE_FUNCTIONS[fallback], fallback
    return None, name or ""


def _resolve_estimator(key: Optional[str], problem_type: str) -> Tuple[Optional[Any], Optional[str]]:
    label = None
    if problem_type == "classification":
        choice = (key or "auto").lower()
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
        choice = (key or "auto").lower()
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


def _build_variance_threshold_selector(
    config: NormalizedFeatureSelectionConfig,
    *_: Any,
) -> Tuple[Optional[Any], Optional[str]]:
    if VarianceThreshold is None:
        return None, None
    threshold = config.threshold if config.threshold is not None else DEFAULT_THRESHOLD
    method_label = f"VarianceThreshold (threshold={threshold})"
    return VarianceThreshold(threshold=threshold), method_label


def _build_select_from_model_selector(
    config: NormalizedFeatureSelectionConfig,
    problem_type: str,
    *_: Any,
) -> Tuple[Optional[Any], Optional[str]]:
    if SelectFromModel is None:
        return None, None
    estimator, estimator_label = _resolve_estimator(config.estimator, problem_type)
    if estimator is None:
        return None, None
    threshold = config.threshold if config.threshold is not None else "median"
    method_label = f"SelectFromModel ({estimator_label or 'estimator'}, threshold={threshold})"
    selector = SelectFromModel(estimator=estimator, threshold=threshold)
    return selector, method_label


def _build_rfe_selector(
    config: NormalizedFeatureSelectionConfig,
    problem_type: str,
    *_: Any,
) -> Tuple[Optional[Any], Optional[str]]:
    if RFE is None:
        return None, None
    estimator, estimator_label = _resolve_estimator(config.estimator, problem_type)
    if estimator is None:
        return None, None
    n_features = config.k
    step = config.step if config.step is not None else 1
    method_label = f"RFE ({estimator_label or 'estimator'}, n_features={n_features or 'auto'})"
    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=step,
    )
    return selector, method_label


def _build_select_k_best_selector(
    config: NormalizedFeatureSelectionConfig,
    _problem_type: str,
    score_function: Optional[Callable[..., Any]],
    score_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if SelectKBest is None or score_function is None:
        return None, None
    k = config.k if config.k is not None else DEFAULT_K_FALLBACK
    method_label = f"SelectKBest ({score_name}, k={k})"
    selector = SelectKBest(score_func=score_function, k=k)
    return selector, method_label


def _build_select_percentile_selector(
    config: NormalizedFeatureSelectionConfig,
    _problem_type: str,
    score_function: Optional[Callable[..., Any]],
    score_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if SelectPercentile is None or score_function is None:
        return None, None
    percentile = config.percentile if config.percentile is not None else DEFAULT_PERCENTILE
    method_label = f"SelectPercentile ({score_name}, {percentile}%)"
    selector = SelectPercentile(score_func=score_function, percentile=percentile)
    return selector, method_label


def _build_select_alpha_variant_selector(
    config: NormalizedFeatureSelectionConfig,
    _problem_type: str,
    score_function: Optional[Callable[..., Any]],
    score_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if score_function is None:
        return None, None
    alpha = config.alpha if config.alpha is not None else DEFAULT_ALPHA
    if config.method == "select_fpr" and SelectFpr is not None:
        return SelectFpr(score_func=score_function, alpha=alpha), f"SelectFpr ({score_name}, alpha={alpha})"
    if config.method == "select_fdr" and SelectFdr is not None:
        return SelectFdr(score_func=score_function, alpha=alpha), f"SelectFdr ({score_name}, alpha={alpha})"
    if config.method == "select_fwe" and SelectFwe is not None:
        return SelectFwe(score_func=score_function, alpha=alpha), f"SelectFwe ({score_name}, alpha={alpha})"
    return None, None


def _build_generic_univariate_selector(
    config: NormalizedFeatureSelectionConfig,
    _problem_type: str,
    score_function: Optional[Callable[..., Any]],
    score_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if GenericUnivariateSelect is None or score_function is None:
        return None, None
    mode = config.mode or "k_best"
    if mode == "k_best":
        param: Any = config.k if config.k is not None else DEFAULT_K_FALLBACK
    elif mode == "percentile":
        param = config.percentile if config.percentile is not None else DEFAULT_PERCENTILE
    else:
        param = config.alpha if config.alpha is not None else DEFAULT_ALPHA
    method_label = f"GenericUnivariateSelect ({score_name}, mode={mode}, param={param})"
    selector = GenericUnivariateSelect(score_func=score_function, mode=mode, param=param)
    return selector, method_label


SelectorBuilder = Callable[
    [NormalizedFeatureSelectionConfig, str, Optional[Callable[..., Any]], str],
    Tuple[Optional[Any], Optional[str]],
]


_SELECTOR_BUILDERS: Dict[str, SelectorBuilder] = {
    "variance_threshold": _build_variance_threshold_selector,
    "select_from_model": _build_select_from_model_selector,
    "rfe": _build_rfe_selector,
    "select_k_best": _build_select_k_best_selector,
    "select_percentile": _build_select_percentile_selector,
    "select_fpr": _build_select_alpha_variant_selector,
    "select_fdr": _build_select_alpha_variant_selector,
    "select_fwe": _build_select_alpha_variant_selector,
    "generic_univariate_select": _build_generic_univariate_selector,
}


def _build_selector(
    config: NormalizedFeatureSelectionConfig,
    problem_type: str,
    score_function: Optional[Callable[..., Any]],
    score_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    handler = _SELECTOR_BUILDERS.get(config.method)
    if handler is None:
        return None, None
    return handler(config, problem_type, score_function, score_name)


def _build_feature_summaries(
    columns: Sequence[str],
    support: Sequence[bool],
    *,
    scores: Optional[Sequence[Any]] = None,
    p_values: Optional[Sequence[Any]] = None,
    ranking: Optional[Sequence[Any]] = None,
    importances: Optional[Sequence[Any]] = None,
    notes: Optional[Dict[str, str]] = None,
) -> List[FeatureSelectionFeatureSummary]:
    summaries: List[FeatureSelectionFeatureSummary] = []
    for idx, column in enumerate(columns):
        selected = bool(support[idx]) if idx < len(support) else False
        score = _safe_float(scores[idx]) if scores is not None and idx < len(scores) else None
        p_value = _safe_float(p_values[idx]) if p_values is not None and idx < len(p_values) else None
        rank = _safe_int(ranking[idx]) if ranking is not None and idx < len(ranking) else None
        importance = _safe_float(importances[idx]) if importances is not None and idx < len(importances) else None
        note = notes.get(column) if notes else None
        summaries.append(
            FeatureSelectionFeatureSummary(
                column=str(column),
                selected=selected,
                score=score,
                p_value=p_value,
                rank=rank,
                importance=importance,
                note=note,
            )
        )
    return summaries


def _initialize_signal_from_config(
    signal: FeatureSelectionNodeSignal,
    config: NormalizedFeatureSelectionConfig,
) -> None:
    signal.method = config.method
    signal.score_func = config.score_func
    signal.mode = config.mode
    signal.estimator = config.estimator
    signal.problem_type = config.problem_type
    signal.target_column = config.target_column
    signal.drop_unselected = config.drop_unselected
    signal.auto_detect = config.auto_detect
    signal.k = config.k
    signal.percentile = config.percentile
    signal.alpha = config.alpha
    signal.threshold = config.threshold


def _collect_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedFeatureSelectionConfig,
) -> List[str]:
    candidate_columns: List[str] = []
    seen: Set[str] = set()

    for column in config.columns:
        normalized = str(column or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidate_columns.append(normalized)

    if config.auto_detect:
        for column in _auto_detect_numeric_columns(frame):
            if column not in seen:
                seen.add(column)
                candidate_columns.append(column)

    return candidate_columns


def _partition_candidate_columns(
    frame: pd.DataFrame,
    candidate_columns: Sequence[str],
    signal: FeatureSelectionNodeSignal,
) -> Tuple[List[str], Optional[str]]:
    missing_columns = [col for col in candidate_columns if col not in frame.columns]
    if missing_columns:
        signal.notes.append(f"Missing columns excluded: {', '.join(missing_columns)}")

    valid_columns = [col for col in candidate_columns if col in frame.columns]
    if not valid_columns:
        message = "Feature selection: candidate columns missing from frame"
        signal.notes.append(message)
        return [], message

    return valid_columns, None


def _prepare_numeric_matrix(
    frame: pd.DataFrame,
    valid_columns: Sequence[str],
    config: NormalizedFeatureSelectionConfig,
    signal: FeatureSelectionNodeSignal,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    numeric_frame = frame[list(valid_columns)].apply(pd.to_numeric, errors="coerce")

    if numeric_frame.dropna(how="all").empty:
        message = "Feature selection: candidate columns lack numeric data"
        signal.notes.append(message)
        return None, message

    if config.score_func == "chi2":
        numeric_frame = _sanitize_chi2_input(numeric_frame.fillna(0.0))
    else:
        numeric_frame = numeric_frame.fillna(0.0)

    return numeric_frame, None


def _resolve_target_vector(
    frame: pd.DataFrame,
    config: NormalizedFeatureSelectionConfig,
    signal: FeatureSelectionNodeSignal,
) -> Tuple[Optional[pd.Series], str, Dict[str, Any], Optional[str]]:
    if config.method == "variance_threshold":
        problem_type = (
            config.problem_type
            if config.problem_type and config.problem_type != "auto"
            else "classification"
        )
        return None, problem_type, {}, None

    if not config.target_column:
        message = "Feature selection: target column required for selected method"
        signal.notes.append(message)
        return None, "classification", {}, message

    if config.target_column not in frame.columns:
        message = f"Feature selection: target column '{config.target_column}' missing"
        signal.notes.append(message)
        return None, "classification", {}, message

    raw_target = frame[config.target_column]
    problem_type = _infer_problem_type(raw_target, config.problem_type)
    prepared_target, metadata = _prepare_target(raw_target, problem_type)
    return prepared_target, problem_type, metadata, None


def _build_selection_inputs(
    frame: pd.DataFrame,
    config: NormalizedFeatureSelectionConfig,
    signal: FeatureSelectionNodeSignal,
) -> Tuple[Optional[SelectionInputs], Optional[str]]:
    candidate_columns = _collect_candidate_columns(frame, config)

    signal.configured_columns = list(config.columns)
    signal.evaluated_columns = list(candidate_columns)

    if not candidate_columns:
        message = "Feature selection: no candidate columns available"
        signal.notes.append(message)
        return None, message

    valid_columns, error = _partition_candidate_columns(frame, candidate_columns, signal)
    if error:
        return None, error

    numeric_frame, error = _prepare_numeric_matrix(frame, valid_columns, config, signal)
    if error:
        return None, error
    assert numeric_frame is not None

    target_vector, problem_type, y_metadata, error = _resolve_target_vector(frame, config, signal)
    if error:
        return None, error

    score_function, score_name = _resolve_score_function(config.score_func, problem_type)
    selector, method_label = _build_selector(config, problem_type, score_function, score_name)

    if selector is None:
        message = "Feature selection: unsupported configuration or missing dependencies"
        signal.notes.append(message)
        return None, message

    signal.problem_type = problem_type
    signal.score_func = score_name

    return SelectionInputs(
        config=config,
        numeric_frame=numeric_frame,
        valid_columns=list(valid_columns),
        target_vector=target_vector,
        problem_type=problem_type,
        score_name=score_name,
        selector=selector,
        method_label=method_label,
        y_metadata=y_metadata,
    ), None


def _filter_training_rows(
    matrix: pd.DataFrame,
    target_vector: Optional[pd.Series],
    problem_type: str,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if target_vector is None:
        return matrix, None

    valid_mask = pd.notna(target_vector)
    if problem_type == "classification":
        valid_mask &= target_vector != -1

    return matrix.loc[valid_mask], target_vector.loc[valid_mask]


def _maybe_warn_zero_variance(
    valid_columns: Sequence[str],
    matrix: np.ndarray,
    target_vector: Optional[pd.Series],
    inputs: SelectionInputs,
    signal: FeatureSelectionNodeSignal,
) -> None:
    if (
        target_vector is None
        or inputs.config.method == "variance_threshold"
        or inputs.score_name != "f_classif"
        or inputs.problem_type != "classification"
    ):
        return

    has_zero_var, problematic_indices = _check_f_classif_zero_variance(matrix, target_vector.values)
    if not has_zero_var:
        return

    problematic_columns = [valid_columns[i] for i in problematic_indices if i < len(valid_columns)]
    warning_msg = (
        f"Feature selection: {len(problematic_columns)} feature(s) have zero within-class variance "
        "and may produce unreliable f_classif scores. Consider removing constant features first."
    )
    logger.warning(warning_msg)
    signal.notes.append(warning_msg)


def _perform_feature_selection(
    frame: pd.DataFrame,
    inputs: SelectionInputs,
    signal: FeatureSelectionNodeSignal,
    pipeline_id: Optional[str],
    node_id_str: Optional[str],
) -> Tuple[Optional[SelectionExecutionResult], Optional[str]]:
    storage: Optional[Any] = None
    split_counts: Dict[str, int] = {}
    selector = inputs.selector
    metadata: Dict[str, Any] = {}
    feature_summaries: List[FeatureSelectionFeatureSummary] = []
    transform_mode = "fit"

    has_splits = SPLIT_TYPE_COLUMN in frame.columns

    if pipeline_id and has_splits:
        storage = get_pipeline_store()
        split_counts = frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
        train_mask = frame[SPLIT_TYPE_COLUMN] == "train"
        X_train = inputs.numeric_frame.loc[train_mask]
        y_train = inputs.target_vector.loc[train_mask] if inputs.target_vector is not None else None
        X_train, y_train = _filter_training_rows(X_train, y_train, inputs.problem_type)

        train_rows = int(split_counts.get("train", 0))
        fit_rows = len(X_train)

        stored_transformer = storage.get_transformer(
            pipeline_id=pipeline_id,
            node_id=node_id_str or "",
            transformer_name=TRANSFORMER_NAME,
        )
        stored_metadata = storage.get_metadata(
            pipeline_id=pipeline_id,
            node_id=node_id_str or "",
            transformer_name=TRANSFORMER_NAME,
        ) or {}

        if train_rows > 0 and fit_rows > 0 and (y_train is None or len(y_train) > 0):
            try:
                if y_train is not None:
                    _maybe_warn_zero_variance(
                        inputs.valid_columns,
                        X_train.values,
                        y_train,
                        inputs,
                        signal,
                    )
                if y_train is not None and inputs.config.method != "variance_threshold":
                    selector.fit(X_train.values, y_train.values)
                else:
                    selector.fit(X_train.values)
                transform_mode = "fit"
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Feature selection fit failed", exc_info=exc)
                warning = f"Feature selection fit failed: {exc}"
                signal.notes.append(warning)
                return None, warning

            metadata, feature_summaries = _build_metadata(
                selector=selector,
                valid_columns=inputs.valid_columns,
                config=inputs.config,
                method_label=inputs.method_label,
                score_name=inputs.score_name,
                y_metadata=inputs.y_metadata,
            )

            storage.store_transformer(
                pipeline_id=pipeline_id,
                node_id=node_id_str or "",
                transformer_name=TRANSFORMER_NAME,
                transformer=selector,
                metadata=metadata,
            )
        elif stored_transformer is not None:
            selector = stored_transformer
            transform_mode = "reuse"
            metadata = stored_metadata
            feature_summaries = _load_feature_summaries_from_metadata(metadata)
            if inputs.method_label is None:
                inputs.method_label = metadata.get("method_label")
        else:
            warning = "Feature selection skipped: no training rows available"
            signal.notes.append(warning)
            logger.info(warning)
            return None, warning
    else:
        X_fit, y_fit = _filter_training_rows(
            inputs.numeric_frame,
            inputs.target_vector,
            inputs.problem_type,
        )
        if X_fit.empty:
            message = "Feature selection: insufficient rows after filtering target"
            signal.notes.append(message)
            return None, message
        try:
            if y_fit is not None:
                _maybe_warn_zero_variance(
                    inputs.valid_columns,
                    X_fit.values,
                    y_fit,
                    inputs,
                    signal,
                )
            if y_fit is not None and inputs.config.method != "variance_threshold":
                selector.fit(X_fit.values, y_fit.values)
            else:
                selector.fit(X_fit.values)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Feature selection fit failed", exc_info=exc)
            warning = f"Feature selection fit failed: {exc}"
            signal.notes.append(warning)
            return None, warning

        metadata, feature_summaries = _build_metadata(
            selector=selector,
            valid_columns=inputs.valid_columns,
            config=inputs.config,
            method_label=inputs.method_label,
            score_name=inputs.score_name,
            y_metadata=inputs.y_metadata,
        )

    method_label = inputs.method_label or metadata.get("method_label")

    return SelectionExecutionResult(
        metadata=metadata,
        feature_summaries=feature_summaries,
        transform_mode=transform_mode,
        storage=storage,
        split_counts=split_counts,
        method_label=method_label,
    ), None


def _compose_selection_summary(
    kept_count: int,
    total_count: int,
    method_label: Optional[str],
) -> str:
    summary = f"Feature selection: kept {kept_count} of {total_count} columns"
    if method_label:
        summary += f" using {method_label}"
    return summary


def apply_feature_selection(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    *,
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, FeatureSelectionNodeSignal]:
    """Apply feature selection techniques leveraging scikit-learn selectors."""

    node_id = node.get("id") if isinstance(node, dict) else None
    node_id_str = str(node_id) if node_id is not None else None

    signal = FeatureSelectionNodeSignal(node_id=node_id_str)

    if frame.empty:
        message = "Feature selection: no data available"
        signal.notes.append(message)
        return frame, message, signal

    if SelectKBest is None:
        warning = "Feature selection skipped: scikit-learn is not installed"
        logger.warning(warning)
        signal.notes.append(warning)
        return frame, warning, signal

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_feature_selection_config(config_payload)

    _initialize_signal_from_config(signal, config)

    inputs, error = _build_selection_inputs(frame, config, signal)
    if error or inputs is None:
        summary = error or "Feature selection failed"
        return frame, summary, signal

    execution_result, error = _perform_feature_selection(
        frame,
        inputs,
        signal,
        pipeline_id,
        node_id_str,
    )

    if error or execution_result is None:
        summary = error or "Feature selection failed"
        return frame, summary, signal

    metadata = execution_result.metadata
    feature_summaries = execution_result.feature_summaries or _load_feature_summaries_from_metadata(metadata)
    selected_columns = metadata.get("selected_columns", [])
    dropped_columns = metadata.get("dropped_columns", [])

    signal.selected_columns = [str(col) for col in selected_columns]
    signal.dropped_columns = [str(col) for col in dropped_columns]
    signal.feature_summaries = feature_summaries
    signal.transform_mode = execution_result.transform_mode

    working_frame = frame.copy()
    if config.drop_unselected and dropped_columns:
        drop_candidates = [col for col in dropped_columns if col in working_frame.columns]
        if drop_candidates:
            working_frame = working_frame.drop(columns=drop_candidates)

    if execution_result.storage is not None and execution_result.split_counts:
        storage = execution_result.storage
        split_counts = execution_result.split_counts
        train_rows = int(split_counts.get("train", 0))
        storage.record_split_activity(
            pipeline_id=pipeline_id,
            node_id=node_id_str or "",
            transformer_name=TRANSFORMER_NAME,
            split_name="train",
            action="fit_transform" if execution_result.transform_mode == "fit" and train_rows > 0 else "transform",
            row_count=train_rows if train_rows > 0 else None,
        )
        for split_name in ("validation", "test"):
            rows = int(split_counts.get(split_name, 0))
            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=node_id_str or "",
                transformer_name=TRANSFORMER_NAME,
                split_name=split_name,
                action="transform" if rows > 0 else "not_available",
                row_count=rows if rows > 0 else None,
            )

    kept_count = len(selected_columns)
    total = len(metadata.get("input_columns", inputs.valid_columns))
    summary = _compose_selection_summary(kept_count, total, execution_result.method_label)
    signal.notes.append(summary)

    return working_frame, summary, signal


def _build_metadata(
    *,
    selector: Any,
    valid_columns: Sequence[str],
    config: NormalizedFeatureSelectionConfig,
    method_label: Optional[str],
    score_name: str,
    y_metadata: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[FeatureSelectionFeatureSummary]]:
    support: Sequence[bool]
    if hasattr(selector, "get_support"):
        support = selector.get_support()
    else:
        support = [True] * len(valid_columns)

    selected_columns = [str(col) for col, keep in zip(valid_columns, support) if keep]
    dropped_columns = [str(col) for col, keep in zip(valid_columns, support) if not keep]

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
                if coef.ndim == 1:
                    importances = np.abs(coef)
                else:
                    importances = np.abs(coef).mean(axis=0)

    notes: Dict[str, str] = {}
    if config.method == "variance_threshold" and hasattr(selector, "variances_"):
        variances = getattr(selector, "variances_")
        for column, variance in zip(valid_columns, variances):
            notes[str(column)] = f"variance={float(variance):.4f}"

    feature_summaries = _build_feature_summaries(
        columns=valid_columns,
        support=support,
        scores=scores,
        p_values=p_values,
        ranking=ranking,
        importances=importances,
        notes=notes,
    )

    metadata: Dict[str, Any] = {
        "input_columns": [str(col) for col in valid_columns],
        "selected_columns": selected_columns,
        "dropped_columns": dropped_columns,
        "support_mask": [bool(value) for value in support],
        "scores": [_safe_float(score) for score in scores] if scores is not None else None,
        "p_values": [_safe_float(value) for value in p_values] if p_values is not None else None,
        "ranking": [_safe_int(value) for value in ranking] if ranking is not None else None,
        "feature_summaries": [summary.model_dump() for summary in feature_summaries],
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
        "column_summary": ", ".join(selected_columns),
    }
    if y_metadata:
        metadata["target_metadata"] = y_metadata
    return metadata, feature_summaries


def _load_feature_summaries_from_metadata(metadata: Optional[Dict[str, Any]]) -> List[FeatureSelectionFeatureSummary]:
    summaries: List[FeatureSelectionFeatureSummary] = []
    if not isinstance(metadata, dict):
        return summaries
    payload = metadata.get("feature_summaries")
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                try:
                    summaries.append(FeatureSelectionFeatureSummary(**entry))
                except TypeError:  # pragma: no cover - defensive
                    logger.debug("Unable to parse feature selection summary", extra={"entry": entry})
    return summaries


__all__ = ["apply_feature_selection"]
