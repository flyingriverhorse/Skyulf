"""Weight-of-Evidence (WOE) / Information-Value (IV) encoder.

Credit-risk standard: replaces each category of a categorical column with the
log-odds of the binary target for that category (WOE). Also records the
Information Value (IV) per column as artifact metadata — a quick univariate
predictive-power score.

Supervised + binary-target only. The math is engine-agnostic; the fit converts
the relevant columns to pandas to compute the mapping, while ``apply`` stays in
the caller's engine (pandas in/out, polars in/out).
"""

import logging
import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import KFold

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from ...types import DEFAULT_RANDOM_STATE
from ...utils import resolve_columns, user_picked_no_columns
from .._category_keys import category_key_expr, category_keys_pandas, uses_category_keys
from .._helpers import select_then_to_pandas
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine, fit_transform_train_dual_engine
from ._common import _exclude_target_column, _extract_target, detect_categorical_columns

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _resolve_apply_inputs(X: Any, params: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Return ``(valid_cols, mappings)`` or ``([], {})`` if nothing to do."""
    cols = params.get("columns", [])
    mappings = params.get("mappings", {})
    valid_cols = [c for c in cols if c in X.columns and c in mappings]
    if not valid_cols:
        return [], {}
    return valid_cols, mappings


def _string_keys_with_nan(series: Any, canonical_keys: bool = False) -> Any:
    """Coerce a pandas column to string keys, rendering missing as ``"nan"``.

    Mirrors the Polars path's ``fill_null("nan")``: a bare ``astype(str)``
    would render ``None`` in an object column as ``"None"``, so the two
    engines would learn different artifact keys for the same null category
    and cross-engine replay would silently fall back to the default (F-28).
    """
    if canonical_keys:
        return category_keys_pandas(series)
    return series.where(series.notna(), "nan").astype(str)


def _woe_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols, mappings = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    default = float(params.get("default", 0.0))
    # ``fill_null("nan")`` matches the pandas side's ``_string_keys_with_nan``
    # so null rows hit the learned mapping on both engines instead of always
    # falling back to ``default``.
    exprs = [
        (
            category_key_expr(col)
            if uses_category_keys(params)
            else pl.col(col).cast(pl.Utf8).fill_null("nan")
        )
        .replace_strict(mappings[col], default=default, return_dtype=pl.Float64)
        .alias(col)
        for col in valid_cols
    ]
    return X.with_columns(exprs), y


def _woe_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols, mappings = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    default = float(params.get("default", 0.0))
    X_out = X.copy()
    for col in valid_cols:
        mapped = _string_keys_with_nan(X_out[col], uses_category_keys(params)).map(mappings[col])
        X_out[col] = mapped.fillna(default).astype(float)
    return X_out, y


class WOEEncoderApplier(BaseApplier):
    """Replace each category with the Weight-of-Evidence value learned for it.

    Both engines look the category up in the same fitted mapping, keyed by the
    scalar key of the value, and fall back to ``default`` for categories
    never seen at fit time. Nulls are keyed as the literal ``"nan"`` on both
    sides, so a missing category gets its own learned WOE instead of always
    taking the default (F-28). New artifacts distinguish numeric values from
    literal strings; older artifacts preserve their original string keys.
    Encoded columns become float.
    """

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch to the engine-specific WOE lookup, forwarding ``(X, y)`` only when present."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"polars": _woe_apply_polars, "pandas": _woe_apply_pandas},
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _binary_target(y: Any) -> np.ndarray | None:
    """Coerce ``y`` to a 0/1 numpy array, or ``None`` if not binary."""
    arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    if _is_null_mask(arr).any():
        raise ValueError("WOEEncoder requires a complete binary target without missing values.")
    classes = np.unique(arr)
    if len(classes) != 2:
        return None
    positive = classes[-1]
    return (arr == positive).astype(float)


def _is_null_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of NaN/None entries, dtype-safe for object arrays."""
    return np.asarray(pd.isna(arr), dtype=bool)


def _column_woe(
    values: np.ndarray, y_bin: np.ndarray, reg: float
) -> tuple[dict[str, float], float]:
    """Compute the WOE map and IV for a single categorical column."""
    total_pos = float(y_bin.sum())
    total_neg = float(len(y_bin) - total_pos)
    mapping: dict[str, float] = {}
    iv = 0.0
    for cat in np.unique(values):
        mask = values == cat
        pos = float(y_bin[mask].sum())
        neg = float(mask.sum() - pos)
        dist_pos = (pos + reg) / (total_pos + reg)
        dist_neg = (neg + reg) / (total_neg + reg)
        woe = math.log(dist_neg / dist_pos)
        mapping[str(cat)] = woe
        iv += (dist_neg - dist_pos) * woe
    return mapping, iv


def _build_woe_artifact(
    frame: Any, y_bin: np.ndarray, cols: list[str], reg: float
) -> Mapping[str, Any]:
    """Build the WOE artifact for the given pandas frame + binary target."""
    mappings: dict[str, dict[str, float]] = {}
    iv_scores: dict[str, float] = {}
    for col in cols:
        values = _string_keys_with_nan(frame[col], True).to_numpy()
        mappings[col], iv_scores[col] = _column_woe(values, y_bin, reg)
    return {
        "type": "woe_encoder",
        "columns": cols,
        "mappings": mappings,
        "information_value": iv_scores,
        "default": 0.0,
        "category_key_version": 1,
    }


def _woe_fit_common(
    frame: Any, y: Any, cols: list[str], config: dict[str, Any]
) -> Mapping[str, Any]:
    """Shared fit: validate binary target, then build the artifact."""
    y_bin = _binary_target(y)
    if y_bin is None:
        logger.warning("WOEEncoder requires a binary target (exactly 2 classes). Skipping.")
        return {}
    reg = float(config.get("regularization", 0.5))
    return _build_woe_artifact(frame, y_bin, cols, reg)


def _categorical_frame_for_fit(X: Any, cols: list[str]) -> Any:
    """Preserve raw scalar types before the shared fit builds canonical keys."""
    if hasattr(X, "fill_null"):  # polars DataFrame
        return pd.DataFrame(
            {col: pd.Series(X.get_column(col).to_list(), dtype=object) for col in cols}
        )
    return select_then_to_pandas(X, cols)[cols]


def _woe_fit(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    """Fit WOE for either dataframe engine using one narrow Pandas boundary."""
    y = _extract_target(X, y, config.get("target_column"))
    if y is None:
        logger.warning("WOEEncoder requires a target variable (y). Skipping.")
        return {}
    cols = _exclude_target_column(
        resolve_columns(X, config, detect_categorical_columns),
        config,
        "WOEEncoder",
        y,
    )
    if not cols:
        return {}
    frame = _categorical_frame_for_fit(X, cols)
    return _woe_fit_common(frame, y, cols, config)


# -----------------------------------------------------------------------------
# fit_transform_train — leakage-safe cross-fitting of the training rows
# -----------------------------------------------------------------------------


def _resolve_woe_cv(y_bin: np.ndarray) -> int:
    """Return the fold count for out-of-fold encoding, shrunk for small data."""
    n_samples = len(y_bin)
    if n_samples < 2:
        raise ValueError(
            "WOEEncoder pipeline training requires at least 2 training rows for "
            f"leakage-safe cross-fitting; got {n_samples}."
        )
    min_class_count = int(min(float(y_bin.sum()), float(n_samples - y_bin.sum())))
    return min(5, max(2, min_class_count))


def _cross_fit_woe_values(
    frame: Any, y_bin: np.ndarray, cols: list[str], reg: float, n_folds: int
) -> dict[str, np.ndarray]:
    """Encode every training row with the WOE map fit on the other folds.

    A category absent from a fold's complement falls back to the apply-time
    default (0.0), matching unseen-category behaviour at serving time.
    """
    n = len(frame)
    encoded = {col: np.zeros(n, dtype=float) for col in cols}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    for train_idx, hold_idx in kf.split(np.arange(n)):
        mappings = _build_woe_artifact(frame.iloc[train_idx], y_bin[train_idx], cols, reg)[
            "mappings"
        ]
        for col in cols:
            held_values = _string_keys_with_nan(frame[col].iloc[hold_idx], True).to_numpy()
            mapping = mappings[col]
            encoded[col][hold_idx] = np.array([mapping.get(v, 0.0) for v in held_values])
    return encoded


def _woe_fit_transform_train_common(
    X: Any, y: Any, config: dict[str, Any]
) -> tuple[Mapping[str, Any], list[str], dict[str, np.ndarray] | None]:
    """Shared fit+cross-fit: returns ``(artifact, cols, encoded-or-None)``."""
    fit_y = _extract_target(X, y, config.get("target_column"))
    if fit_y is None:
        logger.warning("WOEEncoder requires a target variable (y). Skipping.")
        return {}, [], None
    cols = _exclude_target_column(
        resolve_columns(X, config, detect_categorical_columns),
        config,
        "WOEEncoder",
        fit_y,
    )
    if not cols:
        return {}, [], None
    frame = _categorical_frame_for_fit(X, cols)
    y_bin = _binary_target(fit_y)
    if y_bin is None:
        logger.warning("WOEEncoder requires a binary target (exactly 2 classes). Skipping.")
        return {}, [], None
    reg = float(config.get("regularization", 0.5))
    artifact = _build_woe_artifact(frame, y_bin, cols, reg)
    encoded = _cross_fit_woe_values(frame, y_bin, cols, reg, _resolve_woe_cv(y_bin))
    return artifact, cols, encoded


def _woe_fit_transform_train_pandas(
    X: Any, y: Any, config: dict[str, Any]
) -> tuple[Mapping[str, Any], Any, Any]:
    """Fit the full-data artifact and cross-fit the pandas training rows."""
    artifact, cols, encoded = _woe_fit_transform_train_common(X, y, config)
    if encoded is None:
        return artifact, X, y
    X_out = X.copy()
    for col in cols:
        X_out[col] = encoded[col]
    return artifact, X_out, y


def _woe_fit_transform_train_polars(
    X: Any, y: Any, config: dict[str, Any]
) -> tuple[Mapping[str, Any], Any, Any]:
    """Fit the full-data artifact and cross-fit the Polars training rows."""
    artifact, cols, encoded = _woe_fit_transform_train_common(X, y, config)
    if encoded is None:
        return artifact, X, y
    X_out = X.with_columns([pl.Series(col, encoded[col]) for col in cols])
    return artifact, X_out, y


@NodeRegistry.register("WOEEncoder", WOEEncoderApplier)
@node_meta(
    id="WOEEncoder",
    name="WOE / IV Encoder",
    category="Preprocessing",
    description=(
        "Weight-of-Evidence encoder for binary classification. Replaces each "
        "category with its log-odds and records Information Value per column."
    ),
    params={"regularization": 0.5, "columns": []},
    learns_from_data=True,
)
class WOEEncoderCalculator(BaseCalculator):
    """Fit the WOE mapping and per-column Information Value from a binary target.

    Supervised and binary-only: a target without exactly two classes logs a
    warning and yields an empty artifact rather than failing. The target column
    is excluded from the encoded set, since WOE computed against itself is
    near-perfect separation. The math is engine-agnostic, so fit funnels both
    engines through one narrow pandas boundary while ``apply`` stays in the
    caller's engine. ``fit_transform_train`` cross-fits the training rows so a
    row's own target never leaks into its encoding.
    """

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:  # pylint: disable=arguments-differ
        """Short-circuit an explicit empty column selection, else fit via the shared boundary.

        Both engine keys dispatch to the same ``_woe_fit``: it converts to
        pandas internally, so there is no engine-specific fit to choose between.
        """
        if user_picked_no_columns(config):
            return {}
        return cast(
            Mapping[str, Any],
            fit_dual_engine(
                (X, y) if y is not None else X, config, {"polars": _woe_fit, "pandas": _woe_fit}
            ),
        )

    def fit_transform_train(
        self, df: pd.DataFrame | SkyulfDataFrame | tuple, config: dict[str, Any]
    ) -> tuple[Mapping[str, Any], Any]:
        """Fit the full-data WOE artifact and cross-fit the training rows."""
        if user_picked_no_columns(config):
            return {}, df

        artifact, transformed = fit_transform_train_dual_engine(
            df,
            config,
            {"polars": _woe_fit_transform_train_polars, "pandas": _woe_fit_transform_train_pandas},
        )
        return artifact, transformed

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: dict[str, Any],
    ) -> SkyulfSchema | None:
        """Pass the input schema through: WOE values replace the source columns in place."""
        # WOE replaces values in source columns in place (now float-valued);
        # column names are unchanged.
        return input_schema


__all__ = ["WOEEncoderApplier", "WOEEncoderCalculator"]
