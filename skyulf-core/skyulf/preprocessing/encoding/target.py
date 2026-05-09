"""Target Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

from sklearn.preprocessing import TargetEncoder

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import TargetEncoderArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _exclude_target_column, detect_categorical_columns

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _resolve_apply_inputs(X: Any, params: Dict[str, Any]) -> Tuple[List[str], Any]:
    """Return ``(valid_cols, encoder)`` or ``([], None)`` if nothing to do."""
    cols = params.get("columns", [])
    encoder = params.get("encoder_object")
    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols or not encoder:
        return [], None
    return valid_cols, encoder


def _target_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    valid_cols, encoder = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    try:
        X_subset = X.select(valid_cols)
        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        encoded = encoder.transform(X_np)
        n_feats = len(valid_cols)
        if encoded.shape[1] == n_feats:
            # Standard binary / regression: replace in-place.
            new_cols = [pl.Series(col, encoded[:, i]) for i, col in enumerate(valid_cols)]
        else:
            # Multiclass: (n_samples, n_feats * n_classes) — create per-class cols.
            n_classes = encoded.shape[1] // n_feats
            new_cols = []
            for fi, col in enumerate(valid_cols):
                for ci in range(n_classes):
                    new_cols.append(pl.Series(f"{col}_cls{ci}", encoded[:, fi * n_classes + ci]))
            # Drop original columns to avoid schema conflict.
            X = X.drop(valid_cols)
        return X.with_columns(new_cols), y
    except Exception as e:
        logger.error("Target Encoding failed: %s", e)
        return X, y


def _target_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    valid_cols, encoder = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    X_out = X.copy()
    try:
        X_subset = X_out[valid_cols]
        X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
        encoded = encoder.transform(X_input)
        n_feats = len(valid_cols)
        if encoded.shape[1] == n_feats:
            # Standard binary / regression: replace in-place.
            X_out[valid_cols] = encoded
        else:
            # Multiclass: (n_samples, n_feats * n_classes) — create per-class cols.
            n_classes = encoded.shape[1] // n_feats
            X_out = X_out.drop(columns=valid_cols)
            for fi, col in enumerate(valid_cols):
                for ci in range(n_classes):
                    X_out[f"{col}_cls{ci}"] = encoded[:, fi * n_classes + ci]
    except Exception as e:
        logger.error("Target Encoding failed: %s", e)
    return X_out, y


class TargetEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=_target_apply_polars,
            pandas_func=_target_apply_pandas,
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _maybe_extract_y_polars(X: Any, y: Any, target_col: Optional[str]) -> Any:
    """Polars fit-time fallback: pull ``y`` out of ``X`` if missing."""
    if y is not None or not target_col:
        return y
    if target_col in X.columns:
        return X.get_column(target_col)
    return y


def _maybe_extract_y_pandas(X: Any, y: Any, target_col: Optional[str]) -> Any:
    """Pandas fit-time fallback: pull ``y`` out of ``X`` if missing."""
    if y is not None or not target_col:
        return y
    if target_col in X.columns:
        return X[target_col]
    return y


def _y_to_numpy(y: Any) -> Any:
    """Best-effort conversion of ``y`` into a 1-D numpy array."""
    if hasattr(y, "to_numpy"):
        return y.to_numpy()
    if hasattr(y, "to_pandas"):
        return y.to_pandas().to_numpy()
    return y


def _resolve_fit_cols(X: Any, y: Any, config: Dict[str, Any]) -> List[str]:
    """Pick the categorical columns to encode, excluding the target."""
    cols = resolve_columns(X, config, detect_categorical_columns)
    return _exclude_target_column(cols, config, "TargetEncoder", y)


def _fit_target_encoder(X_subset: Any, y: Any, config: Dict[str, Any]) -> TargetEncoder:
    """Run sklearn ``TargetEncoder.fit`` on a prepared subset.

    Handles both numeric and string (categorical) target columns:
    - For multiclass with string y, sklearn needs label-encoded integers.
    - Translates sklearn's ``ValueError: Unknown label type`` into a clear
      actionable message so the user knows exactly which config knob to turn.
    """
    from sklearn.preprocessing import LabelEncoder

    target_type = config.get("target_type", "auto")
    encoder = TargetEncoder(
        smooth=config.get("smooth", "auto"),
        target_type=target_type,
    )
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    y_np = _y_to_numpy(y)

    # If y is object/string and target_type is multiclass (or auto with many classes),
    # label-encode y to integers so sklearn can fit without complaints.
    if hasattr(y_np, "dtype") and y_np.dtype == object and target_type in ("multiclass", "auto"):
        le = LabelEncoder()
        y_np = le.fit_transform(y_np)

    try:
        encoder.fit(X_np, y_np)
    except ValueError as exc:
        msg = str(exc)
        if "unknown label type" in msg.lower() or "multiclass" in msg.lower():
            raise ValueError(
                "TargetEncoder failed: check your target column and your Target Type."
                f"(sklearn said: {msg})"
            ) from exc
        raise
    return encoder


def _target_fit_polars(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    y = _maybe_extract_y_polars(X, y, config.get("target_column"))
    if y is None:
        logger.warning("TargetEncoder requires a target variable (y). Skipping.")
        return {}

    cols = _resolve_fit_cols(X, y, config)
    if not cols:
        return {}

    encoder = _fit_target_encoder(X.select(cols), y, config)
    return {"type": "target_encoder", "columns": cols, "encoder_object": encoder}


def _target_fit_pandas(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    y = _maybe_extract_y_pandas(X, y, config.get("target_column"))
    if y is None:
        logger.warning("TargetEncoder requires a target variable (y). Skipping.")
        return {}

    cols = _resolve_fit_cols(X, y, config)
    if not cols:
        return {}

    encoder = _fit_target_encoder(X[cols], y, config)
    return {"type": "target_encoder", "columns": cols, "encoder_object": encoder}


@NodeRegistry.register("TargetEncoder", TargetEncoderApplier)
@node_meta(
    id="TargetEncoder",
    name="Target Encoder",
    category="Preprocessing",
    description="Encode categorical features using target statistics.",
    params={"smooth": "auto", "target_type": "auto", "columns": []},
)
class TargetEncoderCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> TargetEncoderArtifact:
        if user_picked_no_columns(config):
            return {}
        return cast(
            TargetEncoderArtifact,
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                polars_func=_target_fit_polars,
                pandas_func=_target_fit_pandas,
            ),
        )

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: Dict[str, Any],
    ) -> Optional[SkyulfSchema]:
        # Target encoder replaces values in source columns in place — same
        # column names, dtype becomes float (per-column dtype is best-effort
        # so we don't bother rewriting it).
        return input_schema


__all__ = ["TargetEncoderApplier", "TargetEncoderCalculator"]
