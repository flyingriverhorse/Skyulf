"""Transformer extraction helpers for training workflows."""

from __future__ import annotations

import base64
import logging
import pickle
from typing import Any, Dict, List, Optional

from core.feature_engineering.pipeline_store_singleton import get_pipeline_store

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> Any:
    try:
        if isinstance(value, (list, tuple)) and len(value) == 1:
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
    except Exception:
        logger.exception("Failed to serialize IterativeImputer for column %s", column_name)
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
    except Exception:
        logger.exception("Failed to serialize KNNImputer for column %s", column_name)
    return True


def _handle_hash_encoder(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    _metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_type != "HashingEncoder":
        return False
    try:
        payload = pickle.dumps(transformer_obj)
        params["serialized"] = base64.b64encode(payload).decode("ascii")
    except Exception:
        logger.exception("Failed to serialize HashingEncoder for column %s", column_name)
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
    except Exception:
        logger.exception("Failed to serialize EllipticEnvelope for column %s", column_name)
    return True


def _handle_generic_attributes(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    if transformer_obj is None:
        if isinstance(metadata, dict) and metadata:
            params.update(metadata)
        return bool(params)

    for attr in ["mean_", "scale_", "data_min_", "data_max_", "center_", "quantiles_"]:
        if hasattr(transformer_obj, attr):
            value = getattr(transformer_obj, attr)
            params[attr] = _safe_float(value)
    if hasattr(transformer_obj, "n_bins_"):
        params["n_bins"] = int(getattr(transformer_obj, "n_bins_"))
    if hasattr(transformer_obj, "bin_edges_"):
        params["bin_edges"] = [float(e) for e in getattr(transformer_obj, "bin_edges_")]
    params.setdefault("transformer_type", transformer_type)
    return True


def _extract_fitted_parameters(
    transformer_obj: Any,
    transformer_type: Optional[str],
    column_name: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    handlers = [
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
        _handle_generic_attributes,
    ]

    try:
        handlers = [
            handler
            for handler in handlers
            if handler is _handle_generic_attributes or transformer_obj is not None
        ]
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

    except Exception as exc:
        logger.error(
            "[FIT_PARAMS_ERROR] Failed to extract parameters from %s for column %s: %s",
            transformer_type,
            column_name,
            exc,
            exc_info=True,
        )

    logger.info("[FIT_PARAMS_DEBUG] Returning params for %s: %s", column_name, params)
    return params


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
            transformer_type: Optional[str] = None
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


__all__ = [
    "_collect_transformers",
    "_build_transformer_plan",
    "_extract_fitted_parameters",
]
