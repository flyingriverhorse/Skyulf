"""Shared helpers used by both training and hyperparameter tuning flows."""

from __future__ import annotations

import asyncio
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from celery import Celery
from pandas.api import types as pd_types

try:  # pragma: no cover - ensure workers start even if sklearn adjusts exports
    from sklearn.exceptions import ConvergenceWarning  # type: ignore
except Exception:  # pragma: no cover - fallback when import path changes
    ConvergenceWarning = Warning  # type: ignore

from sklearn.model_selection import KFold, StratifiedKFold

from config import get_settings
from core.database.engine import create_tables, init_db
from core.feature_engineering.preprocessing.split import SPLIT_TYPE_COLUMN

logger = logging.getLogger(__name__)

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


def _resolve_problem_type_hint(problem_type_hint: Optional[str], default: str) -> str:
    normalized = (problem_type_hint or "").strip().lower()
    if normalized in {"classification", "regression"}:
        return normalized
    return default


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
                    feature_target_col = tc.strip()
            elif catalog == "train_test_split":
                tc = cfg.get("target_column") or cfg.get("targetColumn")
                if isinstance(tc, str) and tc.strip():
                    train_test_col = tc.strip()

        inferred = feature_target_col or train_test_col
        if inferred:
            try:
                node_config_obj = (node_map.get(job.node_id, {}).get("data") or {}).get("config")
                if isinstance(node_config_obj, dict):
                    node_config_obj.setdefault("target_column", inferred)
                else:
                    resolved_config = resolved_config if isinstance(resolved_config, dict) else {}
                    resolved_config["target_column"] = inferred
            except Exception:
                logger.debug("Failed to attach inferred target column to node_config")

    training_node = node_map.get(job.node_id)
    if training_node is None:
        raise ValueError("Training node configuration missing from graph payload")

    raw_config = (training_node.get("data") or {}).get("config")
    node_config: Dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}
    return dataset_frame, node_config, dataset_meta or {}, upstream_order


__all__ = [
    "celery_app",
    "ConvergenceWarning",
    "CrossValidationConfig",
    "_build_cv_splitter",
    "_classification_metrics",
    "_ensure_database_ready",
    "_extract_warning_messages",
    "_parse_cross_validation_config",
    "_prepare_training_data",
    "_regression_metrics",
    "_resolve_problem_type_hint",
    "_resolve_training_inputs",
]
