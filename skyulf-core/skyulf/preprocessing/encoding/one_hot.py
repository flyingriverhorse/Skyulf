"""One-Hot Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, List, Mapping, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import OneHotArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _exclude_target_column, detect_categorical_columns

logger = logging.getLogger(__name__)

_MISSING_TOKEN = "__mlops_missing__"


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _validate_apply_params(X: Any, params: Dict[str, Any]) -> Tuple[List[str], Any, Any]:
    """Resolve ``(valid_cols, encoder, feature_names)`` or sentinel values for early-out."""
    if not params or not params.get("columns"):
        return [], None, None
    cols = params["columns"]
    encoder = params.get("encoder_object")
    feature_names = params.get("feature_names")
    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols or not encoder:
        return [], None, None
    return valid_cols, encoder, feature_names


def _to_dense(encoded_array: Any) -> Any:
    """Densify sparse sklearn output and unwrap pandas wrappers."""
    if hasattr(encoded_array, "toarray"):
        return encoded_array.toarray()
    if hasattr(encoded_array, "values"):
        return encoded_array.values
    return encoded_array


def _onehot_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    valid_cols, encoder, feature_names = _validate_apply_params(X, params)
    if not valid_cols:
        return X, y

    drop_original = params.get("drop_original", True)
    include_missing = params.get("include_missing", False)

    try:
        X_subset = X.select(valid_cols)
        if include_missing:
            X_subset = X_subset.fill_null(_MISSING_TOKEN)

        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        encoded = _to_dense(encoder.transform(X_np))

        encoded_df = pl.DataFrame(encoded, schema=feature_names)
        X_out = pl.concat([X, encoded_df], how="horizontal")
        if drop_original:
            X_out = X_out.drop(valid_cols)
        return X_out, y
    except Exception as e:
        logger.error(f"OneHot Encoding failed: {e}")
        return X, y


def _onehot_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    valid_cols, encoder, feature_names = _validate_apply_params(X, params)
    if not valid_cols:
        return X, y

    drop_original = params.get("drop_original", True)
    include_missing = params.get("include_missing", False)
    X_out = X.copy()
    X_subset = X_out[valid_cols]
    if include_missing:
        X_subset = X_subset.fillna(_MISSING_TOKEN)

    try:
        X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
        encoded = _to_dense(encoder.transform(X_input))
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_out.index)
        X_out = pd.concat(cast(Any, [X_out, encoded_df]), axis=1)
        if drop_original:
            X_out = X_out.drop(columns=valid_cols)
    except Exception as e:
        logger.error(f"OneHot Encoding failed: {e}")
    return X_out, y


class OneHotEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=_onehot_apply_polars,
            pandas_func=_onehot_apply_pandas,
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _resolve_fit_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the per-call options out of ``config`` once, with defaults."""
    return {
        "drop": "first" if config.get("drop_first", False) else None,
        "max_categories": config.get("max_categories", 20),
        "handle_unknown": (
            "ignore" if config.get("handle_unknown", "ignore") == "ignore" else "error"
        ),
        "prefix_separator": config.get("prefix_separator", "_"),
        "drop_original": config.get("drop_original", True),
        "include_missing": config.get("include_missing", False),
    }


def _warn_degenerate_categories(encoder: OneHotEncoder, cols: List[str], drop: Any) -> None:
    """Log warnings for empty or single-category columns when relevant."""
    if not hasattr(encoder, "categories_"):
        return
    for i, col in enumerate(cols):
        n_cats = len(encoder.categories_[i])
        if n_cats == 0:
            logger.warning(
                f"OneHotEncoder: Column '{col}' has 0 categories "
                "(empty or all missing). It will be dropped."
            )
        elif drop == "first" and n_cats == 1:
            logger.warning(
                f"OneHotEncoder: Column '{col}' has only 1 category "
                f"('{encoder.categories_[i][0]}') and 'Drop First' is enabled. "
                "This results in 0 encoded features."
            )


def _fit_sklearn_onehot(X_subset: Any, opts: Dict[str, Any], cols: List[str]) -> OneHotEncoder:
    """Run the sklearn ``OneHotEncoder.fit`` step on a prepared subset."""
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    encoder = OneHotEncoder(
        drop=opts["drop"],
        max_categories=opts["max_categories"],
        handle_unknown=opts["handle_unknown"],
        sparse_output=False,
        dtype=np.int8,
    )
    encoder.fit(X_np)
    _warn_degenerate_categories(encoder, cols, opts["drop"])
    return encoder


def _build_onehot_artifact(
    encoder: OneHotEncoder, cols: List[str], opts: Dict[str, Any]
) -> Mapping[str, Any]:
    return {
        "type": "onehot",
        "columns": cols,
        "encoder_object": encoder,
        "feature_names": encoder.get_feature_names_out(cols).tolist(),
        "prefix_separator": opts["prefix_separator"],
        "drop_original": opts["drop_original"],
        "include_missing": opts["include_missing"],
    }


def _onehot_fit_polars(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "OneHotEncoder", y)
    if not cols:
        return {}

    opts = _resolve_fit_options(config)
    X_subset = X.select(cols)
    if opts["include_missing"]:
        X_subset = X_subset.fill_null(_MISSING_TOKEN)

    encoder = _fit_sklearn_onehot(X_subset, opts, cols)
    return _build_onehot_artifact(encoder, cols, opts)


def _onehot_fit_pandas(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "OneHotEncoder", y)
    if not cols:
        return {}

    opts = _resolve_fit_options(config)
    X_subset = X[cols]
    if opts["include_missing"]:
        X_subset = X_subset.fillna(_MISSING_TOKEN)

    encoder = _fit_sklearn_onehot(X_subset, opts, cols)
    return _build_onehot_artifact(encoder, cols, opts)


@NodeRegistry.register("OneHotEncoder", OneHotEncoderApplier)
@node_meta(
    id="OneHotEncoder",
    name="One-Hot Encoder",
    category="Preprocessing",
    description="Encodes categorical features as a one-hot numeric array.",
    params={
        "handle_unknown": "ignore",
        "drop_first": False,
        "max_categories": 20,
        "columns": [],
        "include_missing": False,
    },
)
class OneHotEncoderCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> OneHotArtifact:
        if user_picked_no_columns(config):
            return {}
        return cast(
            OneHotArtifact,
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                polars_func=_onehot_fit_polars,
                pandas_func=_onehot_fit_pandas,
            ),
        )


__all__ = ["OneHotEncoderApplier", "OneHotEncoderCalculator"]
