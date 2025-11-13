"""Celery tasks that execute long-running model training jobs."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import logging
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from celery import Celery
from pandas.api import types as pd_types
try:  # pragma: no cover - ensure workers start even if sklearn adjusts exports
    from sklearn.exceptions import ConvergenceWarning  # type: ignore
except Exception:  # pragma: no cover - fallback when import path changes
    ConvergenceWarning = Warning  # type: ignore

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize

from config import get_settings
from core.database.engine import create_tables, init_db
from core.database.models import get_database_session
from core.feature_engineering.schemas import TrainingJobStatus
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store

from .dataset_split import SPLIT_TYPE_COLUMN
from .model_training_jobs import get_training_job, update_job_status
from .model_training_registry import get_model_spec

_imblearn_metrics = None
try:
    _imblearn_metrics = importlib.import_module("imblearn.metrics")
except ModuleNotFoundError:  # pragma: no cover - optional dependency safeguard
    pass

geometric_mean_score = None
if _imblearn_metrics is not None:
    geometric_mean_score = getattr(_imblearn_metrics, "geometric_mean_score", None)

logger = logging.getLogger(__name__)


def _extract_warning_messages(caught: Optional[Iterable[warnings.WarningMessage]]) -> List[str]:
    messages: List[str] = []
    if not caught:
        return messages

    seen: set[str] = set()
    for entry in caught:
        text = str(getattr(entry, "message", "")).strip()
        if not text:
            continue
        category = getattr(entry, "category", None)
        category_name = getattr(category, "__name__", None) if category else None
        formatted = f"{category_name}: {text}" if category_name else text
        if formatted in seen:
            continue
        seen.add(formatted)
        messages.append(formatted)
    return messages


def _safe_float(value: Any) -> Any:
    try:
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
            value = value[0]
        return float(value)
    except Exception:
        return value


def _log_scaler_debug(transformer_type: Optional[str], transformer_obj: Any) -> None:
    if transformer_obj is None or transformer_type not in {"MinMaxScaler", "RobustScaler", "StandardScaler"}:
        return
    attrs = dir(transformer_obj)
    relevant_attrs = [a for a in attrs if any(x in a for x in ["min", "max", "mean", "scale", "center"])]
    logger.info("[FIT_PARAMS_DEBUG] Available attributes: %s", relevant_attrs)


def _handle_label_encoder(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "LabelEncoder" or not hasattr(transformer_obj, "classes_"):
        return False
    classes = transformer_obj.classes_
    params["mapping"] = {str(val): int(idx) for idx, val in enumerate(classes)}
    params["classes"] = [str(val) for val in classes]
    return True


def _handle_ordinal_encoder(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "OrdinalEncoder" or not hasattr(transformer_obj, "categories_"):
        return False
    if len(transformer_obj.categories_) > 0:
        categories = transformer_obj.categories_[0]
        params["ordering"] = [str(val) for val in categories]
    return True


def _handle_standard_scaler(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "StandardScaler" and metadata.get("method") != "standard":
        return False
    _log_scaler_debug(transformer_type, transformer_obj)
    if hasattr(transformer_obj, "mean_"):
        mean_val = (
            float(transformer_obj.mean_[0])
            if hasattr(transformer_obj.mean_, "__getitem__")
            else float(transformer_obj.mean_)
        )
        params["mean"] = mean_val
        logger.info("[FIT_PARAMS_DEBUG] Extracted mean=%s", mean_val)
    if hasattr(transformer_obj, "scale_"):
        std_val = (
            float(transformer_obj.scale_[0])
            if hasattr(transformer_obj.scale_, "__getitem__")
            else float(transformer_obj.scale_)
        )
        params["std"] = std_val
        logger.info("[FIT_PARAMS_DEBUG] Extracted std=%s", std_val)
    params.setdefault("method", metadata.get("method") or "standard")
    return True


def _handle_minmax_scaler(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "MinMaxScaler" and metadata.get("method") != "minmax":
        return False
    _log_scaler_debug(transformer_type, transformer_obj)
    if hasattr(transformer_obj, "data_min_"):
        min_val = (
            float(transformer_obj.data_min_[0])
            if hasattr(transformer_obj.data_min_, "__getitem__")
            else float(transformer_obj.data_min_)
        )
        params["min"] = min_val
        logger.info("[FIT_PARAMS_DEBUG] Extracted min=%s", min_val)
    if hasattr(transformer_obj, "data_max_"):
        max_val = (
            float(transformer_obj.data_max_[0])
            if hasattr(transformer_obj.data_max_, "__getitem__")
            else float(transformer_obj.data_max_)
        )
        params["max"] = max_val
        logger.info("[FIT_PARAMS_DEBUG] Extracted max=%s", max_val)
    params.setdefault("method", metadata.get("method") or "minmax")
    return True


def _handle_robust_scaler(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "RobustScaler" and metadata.get("method") != "robust":
        return False
    _log_scaler_debug(transformer_type, transformer_obj)
    if hasattr(transformer_obj, "center_"):
        median_val = (
            float(transformer_obj.center_[0])
            if hasattr(transformer_obj.center_, "__getitem__")
            else float(transformer_obj.center_)
        )
        params["median"] = median_val
        logger.info("[FIT_PARAMS_DEBUG] Extracted median=%s", median_val)
    if hasattr(transformer_obj, "scale_"):
        iqr_val = (
            float(transformer_obj.scale_[0])
            if hasattr(transformer_obj.scale_, "__getitem__")
            else float(transformer_obj.scale_)
        )
        params["iqr"] = iqr_val
        logger.info("[FIT_PARAMS_DEBUG] Extracted iqr=%s", iqr_val)
    params.setdefault("method", metadata.get("method") or "robust")
    return True


def _handle_max_abs_scaler(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "MaxAbsScaler" and metadata.get("method") != "maxabs":
        return False
    if hasattr(transformer_obj, "max_abs_"):
        max_abs = (
            float(transformer_obj.max_abs_[0])
            if hasattr(transformer_obj.max_abs_, "__getitem__")
            else float(transformer_obj.max_abs_)
        )
        params["max_abs"] = max_abs
    return True


def _handle_kbins_discretizer(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "KBinsDiscretizer" or not hasattr(transformer_obj, "bin_edges_"):
        return False
    if len(transformer_obj.bin_edges_) > 0:
        edges = transformer_obj.bin_edges_[0]
        params["bin_edges"] = [float(e) for e in edges]
        params["n_bins"] = len(edges) - 1
    return True


def _handle_pandas_binning(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    is_dict_payload = isinstance(transformer_obj, dict) and transformer_obj.get("type") == "pandas_binning"
    if transformer_type != "pandas_binning" and not is_dict_payload:
        return False
    bin_edges = None
    categories = None
    if isinstance(transformer_obj, dict):
        bin_edges = transformer_obj.get("bin_edges")
        categories = transformer_obj.get("categories")
    else:
        bin_edges = getattr(transformer_obj, "bin_edges_", None)
        categories = getattr(transformer_obj, "categories_", None)
    if bin_edges:
        params["bin_edges"] = [float(e) for e in bin_edges]
    if categories:
        params["categories"] = [str(c) for c in categories]
    return True


def _handle_power_transformer(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "PowerTransformer":
        return False
    if hasattr(transformer_obj, "lambdas_"):
        params["lambda"] = (
            float(transformer_obj.lambdas_[0])
            if hasattr(transformer_obj.lambdas_, "__getitem__")
            else float(transformer_obj.lambdas_)
        )
    method = getattr(transformer_obj, "method", "yeo-johnson")
    params["method"] = method
    return True


def _handle_one_hot_encoder(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "OneHotEncoder" or not hasattr(transformer_obj, "categories_"):
        return False
    if len(transformer_obj.categories_) > 0:
        categories = transformer_obj.categories_[0]
        params["categories"] = [str(val) for val in categories]
    return True


def _handle_target_encoder_object(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "TargetEncoder" or not hasattr(transformer_obj, "mapping"):
        return False
    mapping_dict = transformer_obj.mapping
    if mapping_dict:
        params["encoding"] = {str(k): _safe_float(v) for k, v in mapping_dict.items()}
    global_mean = getattr(transformer_obj, "global_mean", None)
    if global_mean is not None:
        params["default_value"] = _safe_float(global_mean)
    placeholder = getattr(transformer_obj, "placeholder", None)
    if placeholder is not None:
        params["placeholder"] = str(placeholder)
    target_column = getattr(transformer_obj, "target_column", None)
    if target_column is not None:
        params["target_column"] = str(target_column)
    return True


def _handle_target_encoder_dict(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if not isinstance(transformer_obj, dict):
        return False
    if transformer_type not in {"dict", "target_encoder", "TargetEncoder"} and not metadata.get("encoded_column"):
        return False
    mapping_dict = transformer_obj.get("mapping") or transformer_obj.get("encoding")
    if isinstance(mapping_dict, dict) and mapping_dict:
        params["encoding"] = {str(k): _safe_float(v) for k, v in mapping_dict.items()}
    global_mean = transformer_obj.get("global_mean") or metadata.get("global_mean")
    if global_mean is not None:
        params["default_value"] = _safe_float(global_mean)
    placeholder = transformer_obj.get("placeholder") or metadata.get("placeholder")
    if placeholder is not None:
        params["placeholder"] = str(placeholder)
    if metadata.get("target_column"):
        params["target_column"] = metadata.get("target_column")
    return True


def _handle_simple_imputer_dict(
    transformer_obj: Any,
    _transformer_type: Optional[str],
    _column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if not isinstance(transformer_obj, dict):
        return False
    method = metadata.get("method") or transformer_obj.get("method")
    if method not in {"mean", "median", "mode", "constant"}:
        return False
    params["strategy"] = str(method or "mean")
    replacement = transformer_obj.get("value")
    if replacement is None:
        replacement = metadata.get("replacement_value")
    if replacement is not None:
        params["replacement_value"] = _safe_float(replacement)
    return True


def _handle_iterative_imputer(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "IterativeImputer":
        return False
    try:
        payload = pickle.dumps(transformer_obj)
        params["serialized"] = base64.b64encode(payload).decode("ascii")
        params["transformer_type"] = "IterativeImputer"
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "[FIT_PARAMS_ERROR] Failed to serialize IterativeImputer for %s: %s",
            column_name,
            exc,
            exc_info=True,
        )
    return True


def _handle_knn_imputer(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "KNNImputer":
        return False
    try:
        payload = pickle.dumps(transformer_obj)
        params["serialized"] = base64.b64encode(payload).decode("ascii")
        params["transformer_type"] = "KNNImputer"
        neighbors = getattr(transformer_obj, "n_neighbors", None)
        if neighbors is not None:
            params["n_neighbors"] = int(neighbors)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "[FIT_PARAMS_ERROR] Failed to serialize KNNImputer for %s: %s",
            column_name,
            exc,
            exc_info=True,
        )
    return True


def _handle_hash_encoder(
    transformer_obj: Any,
    transformer_type: Optional[str],
    _column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    is_hash_type = transformer_type == "hash_encoder" or metadata.get("method") == "hash"
    contains_buckets = isinstance(transformer_obj, dict) and "n_buckets" in transformer_obj
    if not (is_hash_type or contains_buckets):
        return False
    n_buckets = None
    if isinstance(transformer_obj, dict):
        n_buckets = transformer_obj.get("n_buckets")
    if n_buckets is None:
        n_buckets = metadata.get("n_buckets")
    if n_buckets is not None:
        params["n_components"] = int(n_buckets)
        params["n_buckets"] = int(n_buckets)
    encode_missing = False
    if isinstance(transformer_obj, dict):
        encode_missing = transformer_obj.get("encode_missing", False)
    if not encode_missing:
        encode_missing = metadata.get("encode_missing", False)
    params["encode_missing"] = bool(encode_missing)
    params["deterministic"] = True
    params["hash_algorithm"] = "md5"
    return True


def _handle_elliptic_envelope(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "EllipticEnvelope":
        return False
    try:
        payload = pickle.dumps(transformer_obj)
        params["serialized"] = base64.b64encode(payload).decode("ascii")
        params["transformer_type"] = "EllipticEnvelope"
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            f"[FIT_PARAMS_ERROR] Failed to serialize EllipticEnvelope for {column_name}: {exc}",
            exc_info=True,
        )
        params["transformer_type"] = "EllipticEnvelope"
    return True


def _handle_generic_attributes(
    transformer_obj: Any,
    _transformer_type: Optional[str],
    _column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> None:
    for attr in ["classes_", "categories_", "mean_", "scale_", "center_", "min_", "max_"]:
        if hasattr(transformer_obj, attr):
            value = getattr(transformer_obj, attr)
            if isinstance(value, (list, np.ndarray)):
                params[attr] = [
                    float(x) if isinstance(x, (int, float, np.number)) else str(x)
                    for x in value
                ]
            elif isinstance(value, (int, float, np.number)):
                params[attr] = float(value)


def _extract_fitted_parameters(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract fitted parameters from sklearn transformers for export system.

    Args:
        transformer_obj: The fitted sklearn transformer object
        transformer_type: Type/class name of transformer
        column_name: Column the transformer was fit on
        metadata: Additional metadata from storage

    Returns:
        Dictionary of fitted parameters needed for inference
    """
    if transformer_obj is None:
        return {}

    params: Dict[str, Any] = {}
    metadata = metadata or {}

    handlers: Tuple[Callable[[Any, Optional[str], Optional[str], Dict[str, Any], Dict[str, Any]], bool], ...] = (
        _handle_label_encoder,
        _handle_ordinal_encoder,
        _handle_standard_scaler,
        _handle_minmax_scaler,
        _handle_robust_scaler,
        _handle_max_abs_scaler,
        _handle_kbins_discretizer,
        _handle_pandas_binning,
        _handle_power_transformer,
        _handle_one_hot_encoder,
        _handle_target_encoder_object,
        _handle_target_encoder_dict,
        _handle_simple_imputer_dict,
        _handle_iterative_imputer,
        _handle_knn_imputer,
        _handle_hash_encoder,
        _handle_elliptic_envelope,
    )

    try:
        # Debug logging
        logger.info(
            "[FIT_PARAMS_DEBUG] Extracting for column=%s, type=%s, obj=%s",
            column_name,
            transformer_type,
            transformer_obj is not None,
        )
        handled = False
        for handler in handlers:
            if handler(transformer_obj, transformer_type, column_name, metadata, params):
                handled = True
                break
        if not handled:
            _handle_generic_attributes(transformer_obj, transformer_type, column_name, metadata, params)

        method_override = metadata.get("method")
        if method_override and "method" not in params:
            params["method"] = method_override

    except Exception as e:
        # Don't fail training if parameter extraction fails
        logger.error(
            "[FIT_PARAMS_ERROR] Failed to extract parameters from %s for column %s: %s",
            transformer_type,
            column_name,
            e,
            exc_info=True,
        )

    logger.info(
        "[FIT_PARAMS_DEBUG] Returning params for %s: %s",
        column_name,
        params,
    )
    return params


_ALLOWED_CV_STRATEGIES = {"auto", "kfold", "stratified_kfold"}
_ALLOWED_REFIT_STRATEGIES = {"train_only", "train_plus_validation"}
_DEFAULT_CV_FOLDS = 5


@dataclass(frozen=True)
class CrossValidationConfig:
    enabled: bool
    strategy: str
    folds: int
    shuffle: bool
    random_state: Optional[int]
    refit_strategy: str


def _parse_cross_validation_config(config: Dict[str, Any]) -> CrossValidationConfig:
    enabled = bool(config.get("cv_enabled", False))

    raw_strategy = str(config.get("cv_strategy", "auto")).strip().lower()
    strategy = raw_strategy if raw_strategy in _ALLOWED_CV_STRATEGIES else "auto"

    raw_folds = config.get("cv_folds", _DEFAULT_CV_FOLDS)
    try:
        folds = int(raw_folds)
    except (TypeError, ValueError):
        folds = _DEFAULT_CV_FOLDS
    folds = max(2, folds)

    shuffle = bool(config.get("cv_shuffle", True))

    random_state_value = config.get("cv_random_state")
    random_state: Optional[int]
    if isinstance(random_state_value, int):
        random_state = random_state_value
    elif isinstance(random_state_value, str):
        try:
            random_state = int(random_state_value.strip())
        except ValueError:
            random_state = None
    else:
        random_state = None

    raw_refit = str(config.get("cv_refit_strategy", "train_plus_validation")).strip().lower()
    refit_strategy = raw_refit if raw_refit in _ALLOWED_REFIT_STRATEGIES else "train_plus_validation"

    if not enabled:
        return CrossValidationConfig(False, strategy, folds, shuffle, random_state, refit_strategy)

    return CrossValidationConfig(True, strategy, folds, shuffle, random_state, refit_strategy)


def _build_cv_splitter(
    problem_type: str,
    cv_config: CrossValidationConfig,
    y_train: pd.Series,
):
    shuffle = cv_config.shuffle
    random_state = cv_config.random_state if shuffle else None

    strategy = cv_config.strategy
    if strategy == "stratified_kfold" or (strategy == "auto" and problem_type == "classification"):
        return StratifiedKFold(
            n_splits=cv_config.folds,
            shuffle=shuffle,
            random_state=random_state,
        )

    return KFold(
        n_splits=cv_config.folds,
        shuffle=shuffle,
        random_state=random_state,
    )


def _aggregate_cv_metrics(fold_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_values: Dict[str, List[float]] = {}
    for entry in fold_entries:
        metrics = entry.get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                metric_values.setdefault(key, []).append(float(value))

    if not metric_values:
        return {"mean": {}, "std": {}}

    mean_summary = {key: float(np.mean(values)) for key, values in metric_values.items()}
    std_summary = {key: float(np.std(values)) for key, values in metric_values.items()}
    return {"mean": mean_summary, "std": std_summary}


def _run_cross_validation(
    spec,
    params: Dict[str, Any],
    resolved_problem_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_config: CrossValidationConfig,
) -> Optional[Dict[str, Any]]:
    if not cv_config.enabled:
        return None

    if X_train.shape[0] < cv_config.folds:
        return {
            "status": "skipped",
            "reason": "insufficient_rows",
            "requested_folds": cv_config.folds,
            "available_rows": int(X_train.shape[0]),
        }

    try:
        splitter = _build_cv_splitter(resolved_problem_type, cv_config, y_train)
    except ValueError as exc:  # pragma: no cover - defensive guard
        logger.warning("Cross-validation splitter could not be constructed: %s", exc)
        return {
            "status": "skipped",
            "reason": "invalid_configuration",
            "message": str(exc),
        }

    if resolved_problem_type == "classification":
        y_array = y_train.astype(int).to_numpy()
    else:
        y_array = y_train.astype(float).to_numpy()

    fold_entries: List[Dict[str, Any]] = []

    try:
        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_array), start=1):
            if val_idx.size == 0:
                continue

            model = spec.factory(**params)
            model.fit(X_train.iloc[train_idx], y_array[train_idx])

            validation_features = X_train.iloc[val_idx]
            validation_target = y_array[val_idx]

            if resolved_problem_type == "classification":
                metrics = _classification_metrics(model, validation_features, validation_target)
            else:
                metrics = _regression_metrics(model, validation_features, validation_target)

            fold_entries.append(
                {
                    "fold": fold_index,
                    "row_count": int(val_idx.size),
                    "metrics": metrics,
                }
            )
    except ValueError as exc:  # pragma: no cover - defensive guard
        logger.warning("Cross-validation split failed: %s", exc)
        return {
            "status": "skipped",
            "reason": "split_failed",
            "message": str(exc),
        }

    if not fold_entries:
        return {
            "status": "skipped",
            "reason": "no_valid_folds",
        }

    summary = _aggregate_cv_metrics(fold_entries)

    return {
        "status": "completed",
        "strategy": cv_config.strategy,
        "folds": len(fold_entries),
        "shuffle": cv_config.shuffle,
        "random_state": cv_config.random_state,
        "refit_strategy": cv_config.refit_strategy,
        "metrics": summary,
        "folds_detail": fold_entries,
    }


_settings = get_settings()

celery_app = Celery(
    "mlops_training",
    broker=_settings.CELERY_BROKER_URL,
    backend=_settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_default_queue=_settings.CELERY_TASK_DEFAULT_QUEUE,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
)

_db_ready = False
_db_lock: Optional[asyncio.Lock] = None


async def _ensure_database_ready() -> None:
    global _db_ready
    if _db_ready:
        return
    global _db_lock
    if _db_lock is None:
        _db_lock = asyncio.Lock()
    async with _db_lock:
        if _db_ready:
            return
        await init_db()
        await create_tables()
        _db_ready = True
        logger.info("Celery worker database initialized")


def _prepare_feature_matrix(frame: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Return a numeric feature matrix suitable for model fitting."""

    matrix = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        series = matrix[column]
        if pd_types.is_bool_dtype(series):
            matrix[column] = series.astype(int)
        elif pd_types.is_numeric_dtype(series):
            matrix[column] = pd.to_numeric(series, errors="coerce")
        else:
            categorical = pd.Categorical(series)
            matrix[column] = pd.Series(categorical.codes, index=series.index)

    matrix = matrix.replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0)
    return matrix.astype(float)


def _encode_target_series(
    series: pd.Series,
    *,
    categories: Optional[List[Any]] = None,
) -> Tuple[pd.Series, pd.Index, Dict[str, Any]]:
    """Encode target series into numeric form, returning valid index and metadata."""

    meta: Dict[str, Any] = {}
    if pd_types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        valid_idx = numeric[numeric.notna()].index
        meta["dtype"] = str(numeric.dtype)
        return numeric.loc[valid_idx], valid_idx, meta

    categorical = pd.Categorical(series, categories=categories, ordered=False)
    codes = pd.Series(categorical.codes, index=series.index)
    valid_idx = codes[codes != -1].index
    meta["dtype"] = "categorical"
    meta["categories"] = list(categorical.categories)
    return codes.loc[valid_idx], valid_idx, meta


def _prepare_training_data(
    frame: pd.DataFrame,
    target_column: str,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    Optional[pd.DataFrame],
    Optional[pd.Series],
    Optional[pd.DataFrame],
    Optional[pd.Series],
    List[str],
    Dict[str, Any],
]:
    """Split frame into train/validation/test matrices and encoded targets."""

    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in pipeline output")

    working = frame.copy()
    if SPLIT_TYPE_COLUMN in working.columns:
        train_df = working[working[SPLIT_TYPE_COLUMN] == "train"].copy()
        test_df = working[working[SPLIT_TYPE_COLUMN] == "test"].copy()
        validation_df = working[working[SPLIT_TYPE_COLUMN] == "validation"].copy()
    else:
        train_df = working.copy()
        test_df = pd.DataFrame()
        validation_df = pd.DataFrame()

    if train_df.empty:
        raise ValueError("Training split is empty after preprocessing")

    train_df = train_df.dropna(subset=[target_column])
    if train_df.empty:
        raise ValueError("Training data has no rows after dropping missing targets")

    target_series, train_index, target_meta = _encode_target_series(train_df[target_column])
    train_df = train_df.loc[train_index]

    feature_columns = [col for col in train_df.columns if col not in {target_column, SPLIT_TYPE_COLUMN}]
    if not feature_columns:
        raise ValueError("No feature columns available for training")

    X_train = _prepare_feature_matrix(train_df, feature_columns)
    y_train = target_series.loc[train_df.index]

    X_validation: Optional[pd.DataFrame] = None
    y_validation: Optional[pd.Series] = None

    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None

    if not validation_df.empty:
        validation_df = validation_df.dropna(subset=[target_column])
        if not validation_df.empty:
            validation_series, validation_index, _ = _encode_target_series(
                validation_df[target_column], categories=target_meta.get("categories")
            )
            validation_df = validation_df.loc[validation_index]
            if not validation_df.empty and not validation_series.empty:
                X_validation = _prepare_feature_matrix(validation_df, feature_columns)
                y_validation = validation_series.loc[validation_df.index]

    if not test_df.empty:
        test_df = test_df.dropna(subset=[target_column])
        if not test_df.empty:
            test_series, test_index, _ = _encode_target_series(
                test_df[target_column], categories=target_meta.get("categories")
            )
            test_df = test_df.loc[test_index]
            if not test_df.empty and not test_series.empty:
                X_test = _prepare_feature_matrix(test_df, feature_columns)
                y_test = test_series.loc[test_df.index]

    if target_meta.get("dtype") == "categorical":
        y_train = y_train.astype(int)
        if y_validation is not None:
            y_validation = y_validation.astype(int)
        if y_test is not None:
            y_test = y_test.astype(int)
    else:
        y_train = y_train.astype(float)
        if y_validation is not None:
            y_validation = y_validation.astype(float)
        if y_test is not None:
            y_test = y_test.astype(float)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
        target_meta,
    )


def _classification_metrics(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics for predictions."""

    predictions = model.predict(X)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_weighted": float(precision_score(y, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y, predictions, average="weighted", zero_division=0)),
    }

    if geometric_mean_score is not None:
        try:
            metrics["g_score"] = float(geometric_mean_score(y, predictions, average="weighted"))
        except Exception:
            # Some classifiers or class distributions can make geometric mean undefined
            pass

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                class_count = proba.shape[1]
                try:
                    if class_count == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
                        metrics["pr_auc"] = float(average_precision_score(y, proba[:, 1]))
                    else:
                        metrics["roc_auc_weighted"] = float(
                            roc_auc_score(y, proba, multi_class="ovr", average="weighted")
                        )
                        classes = getattr(model, "classes_", None)
                        if classes is None or len(classes) != class_count:
                            classes = np.arange(class_count)
                        y_indicator = label_binarize(y, classes=classes)
                        metrics["pr_auc_weighted"] = float(
                            average_precision_score(y_indicator, proba, average="weighted")
                        )
                except Exception:
                    pass
    except Exception:
        pass

    return metrics


def _regression_metrics(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics for predictions."""

    predictions = model.predict(X)
    mse_value = mean_squared_error(y, predictions)
    metrics: Dict[str, float] = {
        "mae": float(mean_absolute_error(y, predictions)),
        "mse": float(mse_value),
        "rmse": float(math.sqrt(mse_value)),
    }

    try:
        metrics["r2"] = float(r2_score(y, predictions))
    except Exception:
        metrics["r2"] = float("nan")

    try:
        metrics["mape"] = float(mean_absolute_percentage_error(y, predictions))
    except Exception:
        metrics["mape"] = float("nan")

    return metrics


def _resolve_problem_type(problem_type_hint: Optional[str], default: str) -> str:
    if problem_type_hint in {"classification", "regression"}:
        return problem_type_hint
    return default


def _merge_model_params(
    default_params: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    params = dict(default_params)
    if overrides:
        params.update(overrides)
    return params


def _normalize_model_params(spec, params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(params)

    if spec.key == "logistic_regression":
        max_iter_default = int(spec.default_params.get("max_iter", 1000))
        solver_value = normalized.get("solver")
        solver_normalized = str(solver_value).strip().lower() if isinstance(solver_value, str) else None
        requires_longer_runs = solver_normalized in {"lbfgs", "sag", "saga", "newton-cg"}

        max_iter_value = normalized.get("max_iter")
        if isinstance(max_iter_value, (int, float)) and max_iter_value > 0:
            coerced_iter = int(max_iter_value)
            if requires_longer_runs and coerced_iter < max_iter_default:
                normalized["max_iter"] = max_iter_default
            else:
                normalized["max_iter"] = coerced_iter
        else:
            normalized["max_iter"] = max_iter_default

        multi_class_value = normalized.get("multi_class")
        if isinstance(multi_class_value, str):
            lowered_multi = multi_class_value.strip().lower()
            if lowered_multi in {"ovr", "multinomial"}:
                normalized["multi_class"] = lowered_multi
            else:
                normalized.pop("multi_class", None)

    return normalized


def _prepare_refit_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: Optional[pd.DataFrame],
    y_validation: Optional[pd.Series],
    cv_config: CrossValidationConfig,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    use_validation = (
        cv_config.enabled
        and cv_config.refit_strategy == "train_plus_validation"
        and X_validation is not None
        and y_validation is not None
        and not X_validation.empty
        and y_validation.shape[0] > 0
    )

    if use_validation:
        fit_features = pd.concat([X_train, X_validation], axis=0)
        fit_target = pd.concat([y_train, y_validation], axis=0)
    else:
        fit_features = X_train
        fit_target = y_train

    return fit_features, fit_target, use_validation


def _prepare_target_array(target: pd.Series, problem_type: str) -> np.ndarray:
    if problem_type == "classification":
        return target.astype(int).to_numpy()
    return target.astype(float).to_numpy()


def _initialize_metrics(
    feature_columns: List[str],
    target_column: str,
    model_type: str,
    version: int,
    refit_strategy: str,
    used_validation: bool,
    cv_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "row_counts": {},
        "feature_columns": feature_columns,
        "target_column": target_column,
        "model_type": model_type,
        "version": version,
        "refit_strategy": refit_strategy,
        "validation_used_for_training": used_validation,
    }
    if cv_summary is not None:
        metrics["cross_validation"] = cv_summary
    return metrics


def _add_split_metrics(
    metrics: Dict[str, Any],
    split_name: str,
    model,
    X: Optional[pd.DataFrame],
    y: Optional[pd.Series],
    problem_type: str,
) -> None:
    if X is None or y is None or X.empty or y.shape[0] <= 0:
        metrics.setdefault("row_counts", {})[split_name] = 0
        return

    y_array = _prepare_target_array(y, problem_type)
    metrics["row_counts"][split_name] = int(y_array.shape[0])
    metrics[split_name] = (
        _classification_metrics(model, X, y_array)
        if problem_type == "classification"
        else _regression_metrics(model, X, y_array)
    )


def _collect_transformers(pipeline_id: str) -> List[Dict[str, Any]]:
    if not pipeline_id:
        return []

    transformers: List[Dict[str, Any]] = []
    storage = get_pipeline_store()
    records = storage.list_transformers(pipeline_id=pipeline_id)
    for record in records:
        node_id_value = record.get("node_id")
        transformer_name = record.get("transformer_name")
        if not transformer_name or node_id_value is None:
            continue

        column_name = record.get("column_name")
        transformer_obj = storage.get_transformer(
            pipeline_id=pipeline_id,
            node_id=str(node_id_value),
            transformer_name=str(transformer_name),
            column_name=str(column_name) if column_name is not None else None,
        )

        if transformer_obj is None:
            continue

        transformers.append(
            {
                "node_id": str(node_id_value),
                "transformer_name": str(transformer_name),
                "column_name": str(column_name) if column_name is not None else None,
                "transformer": transformer_obj,
                "metadata": record.get("metadata") or {},
                "split_activity": record.get("split_activity") or {},
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
            }
        )

    return transformers


def _build_transformer_plan(
    transformers: List[Dict[str, Any]],
    upstream_node_order: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if not transformers:
        return []

    by_node: Dict[str, List[Dict[str, Any]]] = {}
    for entry in transformers:
        by_node.setdefault(entry["node_id"], []).append(entry)

    ordered_nodes = upstream_node_order or []
    planned_nodes = ordered_nodes if ordered_nodes else sorted(by_node.keys())

    transformer_plan: List[Dict[str, Any]] = []
    for node_id in planned_nodes:
        node_entries = by_node.get(node_id)
        if not node_entries:
            continue

        step_transformers: List[Dict[str, Any]] = []
        for entry in node_entries:
            transformer_obj = entry.get("transformer")
            transformer_type = None
            if transformer_obj is not None:
                transformer_type = transformer_obj.__class__.__name__
            elif isinstance(entry.get("metadata"), dict):
                transformer_type = entry["metadata"].get("method")

            entry_metadata = entry.get("metadata") or {}
            logger.info(
                "[FIT_PARAMS_DEBUG] Before extraction: col=%s, type=%s, metadata_method=%s, obj_type=%s",
                entry.get("column_name"),
                transformer_type,
                entry_metadata.get("method"),
                type(transformer_obj).__name__ if transformer_obj else None,
            )

            if transformer_obj is not None:
                has_attrs = {
                    attr: hasattr(transformer_obj, attr)
                    for attr in ["mean_", "scale_", "data_min_", "data_max_", "center_"]
                }
                logger.info("[FIT_PARAMS_DEBUG] Transformer attributes: %s", has_attrs)

            fitted_params = _extract_fitted_parameters(
                transformer_obj=transformer_obj,
                transformer_type=transformer_type,
                column_name=entry.get("column_name"),
                metadata=entry_metadata,
            )
            logger.info(
                "[FIT_PARAMS_DEBUG] After extraction: col=%s, params=%s",
                entry.get("column_name"),
                fitted_params,
            )

            step_transformers.append(
                {
                    "transformer_name": entry.get("transformer_name"),
                    "column_name": entry.get("column_name"),
                    "transformer_type": transformer_type,
                    "metadata": entry.get("metadata") or {},
                    "fitted_params": fitted_params,
                }
            )

        if step_transformers:
            transformer_plan.append({"node_id": node_id, "transformers": step_transformers})

    return transformer_plan


def _dump_fitted_params_debug(
    transformers: List[Dict[str, Any]],
    transformer_plan: List[Dict[str, Any]],
) -> None:
    try:
        debug_dir = Path("logs")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / "fitted_params_debug_latest.json"
        with debug_file.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "transformer_plan": transformer_plan,
                    "transformers_overview": [
                        {
                            "node_id": (t.get("node_id") if isinstance(t, dict) else None),
                            "transformer_name": (
                                t.get("transformer_name") if isinstance(t, dict) else None
                            ),
                            "column_name": (
                                t.get("column_name") if isinstance(t, dict) else None
                            ),
                            "transformer_type": (
                                t.get("transformer").__class__.__name__
                                if isinstance(t, dict) and t.get("transformer") is not None
                                else None
                            ),
                            "metadata": (t.get("metadata") if isinstance(t, dict) else {}),
                        }
                        for t in transformers
                    ],
                },
                fh,
                indent=2,
                default=str,
            )
        logger.info("Wrote fitted-params debug file: %s", debug_file)
    except Exception:
        logger.exception("Failed to write fitted-params debug file")


def _persist_artifact(
    artifact_root: str,
    pipeline_id: str,
    job_id: str,
    version: int,
    artifact_data: Dict[str, Any],
) -> Path:
    artifact_dir = Path(artifact_root) / pipeline_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{job_id}_v{version}.joblib"
    joblib.dump(artifact_data, artifact_path, compress=("gzip", 3))
    return artifact_path


def _train_and_save_model(
    model_type: str,
    hyperparameters: Optional[Dict[str, Any]],
    problem_type_hint: Optional[str],
    target_column: str,
    feature_columns: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: Optional[pd.DataFrame],
    y_validation: Optional[pd.Series],
    X_test: Optional[pd.DataFrame],
    y_test: Optional[pd.Series],
    artifact_root: str,
    pipeline_id: str,
    job_id: str,
    version: int,
    cv_config: CrossValidationConfig,
    upstream_node_order: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any], str, List[str]]:
    """Fit model synchronously, persist artifacts/metrics, and collect warnings."""

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        warnings.simplefilter("always", ConvergenceWarning)

        spec = get_model_spec(model_type)
        resolved_problem_type = _resolve_problem_type(problem_type_hint, spec.problem_type)
        params = _merge_model_params(spec.default_params, hyperparameters)
        params = _normalize_model_params(spec, params)

        cv_summary = _run_cross_validation(spec, params, resolved_problem_type, X_train, y_train, cv_config)

        fit_features, fit_target, used_validation = _prepare_refit_dataset(
            X_train,
            y_train,
            X_validation,
            y_validation,
            cv_config,
        )

        final_model = spec.factory(**params)
        fit_target_array = _prepare_target_array(fit_target, resolved_problem_type)
        final_model.fit(fit_features, fit_target_array)

        metrics = _initialize_metrics(
            feature_columns,
            target_column,
            spec.key,
            version,
            cv_config.refit_strategy,
            used_validation,
            cv_summary,
        )
        metrics["row_counts"]["train"] = int(fit_target_array.shape[0])
        metrics["train"] = (
            _classification_metrics(final_model, fit_features, fit_target_array)
            if resolved_problem_type == "classification"
            else _regression_metrics(final_model, fit_features, fit_target_array)
        )

        _add_split_metrics(metrics, "validation", final_model, X_validation, y_validation, resolved_problem_type)
        _add_split_metrics(metrics, "test", final_model, X_test, y_test, resolved_problem_type)

        transformers = _collect_transformers(pipeline_id)
        transformer_plan = _build_transformer_plan(transformers, upstream_node_order)

        artifact_data = {
            "model": final_model,
            "model_type": spec.key,
            "problem_type": resolved_problem_type,
            "feature_columns": feature_columns,
            "transformers": transformers,
            "transformer_plan": transformer_plan,
            "transformer_bundle_version": 1,
            "version": version,
        }

        _dump_fitted_params_debug(transformers, transformer_plan)

        artifact_path = _persist_artifact(
            artifact_root,
            pipeline_id,
            job_id,
            version,
            artifact_data,
        )

        metrics["artifact_uri"] = str(artifact_path)

    warning_messages = _extract_warning_messages(caught_warnings)
    if warning_messages:
        metrics.setdefault("warnings", warning_messages)

    return str(artifact_path), metrics, resolved_problem_type, warning_messages


async def _resolve_training_inputs(session, job) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], List[str]]:
    """Load dataset and apply pipeline transformations up to the training node."""

    from core.feature_engineering import routes as fe_routes  # Lazy import to avoid circular deps

    graph_payload = job.graph or {}
    graph_nodes = graph_payload.get("nodes") or []
    graph_edges = graph_payload.get("edges") or []

    node_map_raw = fe_routes._sanitize_graph_nodes(graph_nodes)
    edges = fe_routes._sanitize_graph_edges(graph_edges)
    node_map = cast(Dict[str, Dict[str, Any]], fe_routes._ensure_dataset_node(node_map_raw))

    execution_order = fe_routes._execution_order(node_map, edges, job.node_id)
    if not execution_order:
        raise ValueError("Unable to resolve pipeline execution order for training node")

    upstream_order = execution_order[:-1] if execution_order[-1] == job.node_id else execution_order

    dataset_frame, dataset_meta = await fe_routes._load_dataset_frame(
        session,
        job.dataset_source_id,
        sample_size=0,
        execution_mode="full",
    )

    if upstream_order:
        dataset_frame, _, _, _ = fe_routes._run_pipeline_execution(
            dataset_frame,
            upstream_order,
            node_map,
            pipeline_id=job.pipeline_id,
            collect_signals=False,
            preserve_split_column=True,
        )

    # Attempt to infer a target column from upstream node configs when the
    # training node doesn't explicitly provide one. This mirrors the UI
    # behaviour which prefers a `feature_target_split` node, falling back
    # to `train_test_split` if present upstream.
    try:
        resolved_config = (node_map.get(job.node_id, {}).get("data") or {}).get("config") or {}
    except Exception:
        resolved_config = {}

    job_metadata = job.job_metadata or {}
    has_explicit_target = bool(
        resolved_config.get("target_column")
        or resolved_config.get("targetColumn")
        or job_metadata.get("target_column")
    )

    if not has_explicit_target and upstream_order:
        feature_target_col: Optional[str] = None
        train_test_col: Optional[str] = None
        for upstream_id in upstream_order:
            upstream_node = node_map.get(upstream_id)
            if not upstream_node:
                continue
            data = upstream_node.get("data") or {}
            catalog = str(data.get("catalogType") or "").lower().strip()
            cfg = data.get("config") or {}
            if catalog == "feature_target_split":
                tc = cfg.get("target_column") or cfg.get("targetColumn")
                if isinstance(tc, str) and tc.strip():
                    # mirror frontend: keep the last occurrence if multiple
                    feature_target_col = tc.strip()
            elif catalog == "train_test_split":
                tc = cfg.get("target_column") or cfg.get("targetColumn")
                if isinstance(tc, str) and tc.strip():
                    train_test_col = tc.strip()

        inferred = feature_target_col or train_test_col
        if inferred:
            # ensure node_config is a dict and set the inferred value so
            # downstream code (training/tuning) can pick it up from node_config
            try:
                node_config_obj = (node_map.get(job.node_id, {}).get("data") or {}).get("config")
                if isinstance(node_config_obj, dict):
                    node_config_obj.setdefault("target_column", inferred)
                else:
                    # fallback: ensure returned node_config contains the inferred value
                    resolved_config = resolved_config if isinstance(resolved_config, dict) else {}
                    resolved_config["target_column"] = inferred
            except Exception:
                # best-effort only; don't fail training resolution on logging
                logger.debug("Failed to attach inferred target column to node_config")

    training_node = node_map.get(job.node_id)
    if training_node is None:
        raise ValueError("Training node configuration missing from graph payload")

    raw_config = (training_node.get("data") or {}).get("config")
    node_config: Dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}
    return dataset_frame, node_config, dataset_meta or {}, upstream_order


async def _run_training_workflow(job_id: str) -> None:
    await _ensure_database_ready()

    async with get_database_session(expire_on_commit=False) as session:
        job = await get_training_job(session, job_id)
        if job is None:
            logger.warning("Training job %s not found; skipping", job_id)
            return

        try:
            job = await update_job_status(session, job, status=TrainingJobStatus.RUNNING)
            frame, resolved_node_config, dataset_meta, upstream_order = await _resolve_training_inputs(session, job)
            node_config: Dict[str, Any] = cast(Dict[str, Any], resolved_node_config)

            def _config_value(key: str) -> Any:
                return node_config[key] if key in node_config else None

            job_metadata: Dict[str, Any] = job.job_metadata if isinstance(job.job_metadata, dict) else {}

            target_column = (
                _config_value("target_column")
                or _config_value("targetColumn")
                or job_metadata.get("target_column")
            )
            if not target_column:
                raise ValueError("Training configuration missing target column")

            hyperparameters: Dict[str, Any] = {}
            hyperparameters_config = _config_value("hyperparameters")
            if isinstance(hyperparameters_config, dict):
                hyperparameters.update(hyperparameters_config)
            if isinstance(job.hyperparameters, dict):
                hyperparameters.update(job.hyperparameters)

            problem_type_hint = _config_value("problem_type") or _config_value("problemType") or "auto"
            raw_model_type = job.model_type
            if raw_model_type is None:
                raise ValueError("Training job missing model_type")

            if isinstance(raw_model_type, str):
                model_type_value = raw_model_type
            else:
                model_type_value = str(raw_model_type)

            raw_pipeline_id = job.pipeline_id
            if raw_pipeline_id is None:
                raise ValueError("Training job missing pipeline_id")
            pipeline_id_value = raw_pipeline_id if isinstance(raw_pipeline_id, str) else str(raw_pipeline_id)

            raw_job_id = job.id
            job_id_value = raw_job_id if isinstance(raw_job_id, str) else str(raw_job_id)

            raw_version = job.version
            if raw_version is None:
                raise ValueError("Training job missing version")
            version_value = int(raw_version) if not isinstance(raw_version, int) else raw_version

            (
                X_train,
                y_train,
                X_validation,
                y_validation,
                X_test,
                y_test,
                feature_columns,
                target_meta,
            ) = _prepare_training_data(
                frame,
                target_column,
            )

            cv_config = _parse_cross_validation_config(node_config)

            artifact_root = _settings.TRAINING_ARTIFACT_DIR
            artifact_uri, metrics, resolved_problem_type, training_warnings = _train_and_save_model(
                model_type=model_type_value,
                hyperparameters=hyperparameters or None,
                problem_type_hint=problem_type_hint,
                target_column=target_column,
                feature_columns=feature_columns,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                X_test=X_test,
                y_test=y_test,
                artifact_root=artifact_root,
                pipeline_id=pipeline_id_value,
                job_id=job_id_value,
                version=version_value,
                cv_config=cv_config,
                upstream_node_order=upstream_order,
            )

            metrics["problem_type"] = resolved_problem_type
            if dataset_meta:
                try:
                    metrics.setdefault("dataset", {}).update(dataset_meta)
                except AttributeError:
                    metrics["dataset"] = dataset_meta
            metadata_update = {
                "resolved_problem_type": resolved_problem_type,
                "target_column": target_column,
                "target_encoding": target_meta,
                "feature_columns": feature_columns,
                "cross_validation": {
                    "enabled": cv_config.enabled,
                    "strategy": cv_config.strategy,
                    "folds": cv_config.folds,
                    "shuffle": cv_config.shuffle,
                    "random_state": cv_config.random_state,
                    "refit_strategy": cv_config.refit_strategy,
                },
            }
            if dataset_meta:
                metadata_update["dataset"] = dataset_meta

            if training_warnings:
                metadata_update["warnings"] = training_warnings

            job = await update_job_status(
                session,
                job,
                status=TrainingJobStatus.SUCCEEDED,
                metrics=metrics,
                artifact_uri=artifact_uri,
                metadata=metadata_update,
            )

            logger.info(
                "Training job %s succeeded (model=%s, version=%s, pipeline=%s, node=%s)",
                job.id,
                model_type_value,
                job.version,
                job.pipeline_id,
                job.node_id,
            )
        except Exception as exc:  # pragma: no cover - defensive guard for worker runtime
            logger.exception("Training job %s failed", job_id)
            await update_job_status(
                session,
                job,
                status=TrainingJobStatus.FAILED,
                error_message=str(exc),
            )


@celery_app.task(name="core.feature_engineering.nodes.modeling.model_training.train_model")
def train_model(job_id: str) -> None:
    """Celery entrypoint for training jobs."""

    asyncio.run(_run_training_workflow(job_id))


def dispatch_training_job(job_id: str) -> None:
    """Queue a Celery task for the given job identifier."""

    train_model.delay(job_id)
