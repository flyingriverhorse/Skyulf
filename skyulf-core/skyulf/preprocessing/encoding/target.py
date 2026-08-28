"""Target Encoder node (Calculator + Applier)."""

import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.multiclass import type_of_target

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...types import DEFAULT_RANDOM_STATE
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import TargetEncoderArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine, fit_transform_train_dual_engine
from ._common import _exclude_target_column, _extract_target, detect_categorical_columns

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _resolve_apply_inputs(X: Any, params: dict[str, Any]) -> tuple[list[str], Any]:
    """Return ``(valid_cols, encoder)`` or ``([], None)`` if nothing to do."""
    cols = params.get("columns", [])
    encoder = params.get("encoder_object")
    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols or not encoder:
        return [], None
    return valid_cols, encoder


def _replace_target_encoded_polars(
    X: Any, y: Any, valid_cols: list[str], encoded: Any
) -> tuple[Any, Any]:
    """Replace encoded columns in a Polars frame."""
    import polars as pl

    n_feats = len(valid_cols)
    if encoded.shape[1] == n_feats:
        new_cols = [pl.Series(col, encoded[:, i]) for i, col in enumerate(valid_cols)]
        return X.with_columns(new_cols), y

    n_classes = encoded.shape[1] // n_feats
    new_cols = []
    for fi, col in enumerate(valid_cols):
        for ci in range(n_classes):
            new_cols.append(pl.Series(f"{col}_cls{ci}", encoded[:, fi * n_classes + ci]))
    return X.drop(valid_cols).with_columns(new_cols), y


def _replace_target_encoded_pandas(
    X: Any, y: Any, valid_cols: list[str], encoded: Any
) -> tuple[Any, Any]:
    """Replace encoded columns in a Pandas frame."""
    X_out = X.copy()
    n_feats = len(valid_cols)
    if encoded.shape[1] == n_feats:
        X_out[valid_cols] = encoded
        return X_out, y

    n_classes = encoded.shape[1] // n_feats
    X_out = X_out.drop(columns=valid_cols)
    for fi, col in enumerate(valid_cols):
        for ci in range(n_classes):
            X_out[f"{col}_cls{ci}"] = encoded[:, fi * n_classes + ci]
    return X_out, y


def _target_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols, encoder = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    X_subset = X.select(valid_cols)
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    encoded = encoder.transform(X_np)
    return _replace_target_encoded_polars(X, y, valid_cols, encoded)


def _target_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols, encoder = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    X_subset = X[valid_cols]
    X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
    encoded = encoder.transform(X_input)
    return _replace_target_encoded_pandas(X, y, valid_cols, encoded)


class TargetEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=_target_apply_polars,
            pandas_func=_target_apply_pandas,
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _y_to_numpy(y: Any) -> Any:
    """Best-effort conversion of ``y`` into a 1-D numpy array.

    Both pandas and Polars Series/DataFrames expose `.to_numpy()` natively,
    so a `.to_pandas().to_numpy()` fallback would never be reached in
    practice -- pandas has always had `.to_numpy()`, and Polars has too
    since well before this codebase's minimum supported version.
    """
    if hasattr(y, "to_numpy"):
        return y.to_numpy()
    return y


def _resolve_fit_cols(X: Any, y: Any, config: dict[str, Any]) -> list[str]:
    """Pick the categorical columns to encode, excluding the target."""
    cols = resolve_columns(X, config, detect_categorical_columns)
    return _exclude_target_column(cols, config, "TargetEncoder", y)


def _translate_target_encoder_error(exc: ValueError) -> None:
    """Raise a clearer error for sklearn target-type failures."""
    msg = str(exc)
    if "unknown label type" in msg.lower() or "multiclass" in msg.lower():
        raise ValueError(
            "TargetEncoder failed: check your target column and your Target Type."
            f"(sklearn said: {msg})"
        ) from exc
    raise exc


def _build_target_encoder(
    X_subset: Any, y: Any, config: dict[str, Any], *, cv: int = 5
) -> tuple[TargetEncoder, Any]:
    """Build the sklearn encoder and normalize the target array."""
    from sklearn.preprocessing import LabelEncoder

    _ = X_subset
    target_type = config.get("target_type", "auto")
    encoder = TargetEncoder(
        smooth=config.get("smooth", "auto"),
        target_type=target_type,
        cv=cv,
        shuffle=True,
        random_state=DEFAULT_RANDOM_STATE,
    )
    y_np = _y_to_numpy(y)

    # If y is object/string and target_type is multiclass (or auto with many classes),
    # label-encode y to integers so sklearn can fit without complaints.
    if hasattr(y_np, "dtype") and y_np.dtype == object and target_type in ("multiclass", "auto"):
        le = LabelEncoder()
        y_np = le.fit_transform(y_np)

    return encoder, y_np


def _resolve_target_encoder_training_cv(y_np: Any, target_type: str) -> int:
    """Return the highest deterministic non-leaky fold count for training rows."""
    y_array = np.asarray(y_np)
    n_samples = len(y_array)
    if n_samples < 2:
        raise ValueError(
            "TargetEncoder pipeline training requires at least 2 training rows for "
            f"leakage-safe cross-fitting; got {n_samples}."
        )

    safe_cv = min(5, n_samples)
    if y_array.ndim != 1:
        return safe_cv

    resolved_target_type = target_type
    if resolved_target_type == "auto":
        inferred_target_type = type_of_target(y_array)
        if inferred_target_type == "continuous":
            resolved_target_type = "regression"
        elif inferred_target_type in {"binary", "multiclass"}:
            resolved_target_type = inferred_target_type

    if resolved_target_type in {"binary", "multiclass"}:
        classes, counts = np.unique(y_array, return_counts=True)
        min_class_count = int(counts.min())
        if min_class_count < 2:
            class_counts = dict(zip(classes.tolist(), counts.tolist(), strict=False))
            raise ValueError(
                "TargetEncoder pipeline training requires at least 2 training rows in every "
                "target class for leakage-safe cross-fitting; "
                f"got class counts {class_counts}."
            )
        safe_cv = min(safe_cv, min_class_count)

    return safe_cv


def _fit_target_encoder(X_subset: Any, y: Any, config: dict[str, Any]) -> TargetEncoder:
    """Run sklearn ``TargetEncoder.fit`` on a prepared subset."""
    encoder, y_np = _build_target_encoder(X_subset, y, config)
    X_np, _ = SklearnBridge.to_sklearn(X_subset)

    try:
        encoder.fit(X_np, y_np)
    except ValueError as exc:
        _translate_target_encoder_error(exc)
    return encoder


def _fit_transform_target_encoder(X_subset: Any, y: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    """Fit sklearn ``TargetEncoder`` and cross-fit the training rows."""
    encoder, y_np = _build_target_encoder(X_subset, y, config)
    train_cv = _resolve_target_encoder_training_cv(y_np, str(config.get("target_type", "auto")))
    if train_cv != 5:
        encoder, y_np = _build_target_encoder(X_subset, y, config, cv=train_cv)
    X_np, _ = SklearnBridge.to_sklearn(X_subset)

    try:
        encoded = encoder.fit_transform(X_np, y_np)
    except ValueError as exc:
        _translate_target_encoder_error(exc)
    return encoder, encoded


def _target_fit_polars(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    y = _extract_target(X, y, config.get("target_column"))
    if y is None:
        logger.warning("TargetEncoder requires a target variable (y). Skipping.")
        return {}

    cols = _resolve_fit_cols(X, y, config)
    if not cols:
        return {}

    encoder = _fit_target_encoder(X.select(cols), y, config)
    return {"type": "target_encoder", "columns": cols, "encoder_object": encoder}


def _target_fit_pandas(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    y = _extract_target(X, y, config.get("target_column"))
    if y is None:
        logger.warning("TargetEncoder requires a target variable (y). Skipping.")
        return {}

    cols = _resolve_fit_cols(X, y, config)
    if not cols:
        return {}

    encoder = _fit_target_encoder(X[cols], y, config)
    return {"type": "target_encoder", "columns": cols, "encoder_object": encoder}


def _target_fit_transform_train_polars(
    X: Any, y: Any, config: dict[str, Any]
) -> tuple[Mapping[str, Any], Any, Any]:
    """Fit and cross-fit training rows for Polars-backed inputs."""
    fit_y = _extract_target(X, y, config.get("target_column"))
    if fit_y is None:
        logger.warning("TargetEncoder requires a target variable (y). Skipping.")
        return {}, X, y

    cols = _resolve_fit_cols(X, fit_y, config)
    if not cols:
        return {}, X, y

    encoder, encoded = _fit_transform_target_encoder(X.select(cols), fit_y, config)
    X_out, y_out = _replace_target_encoded_polars(X, y, cols, encoded)
    return {"type": "target_encoder", "columns": cols, "encoder_object": encoder}, X_out, y_out


def _target_fit_transform_train_pandas(
    X: Any, y: Any, config: dict[str, Any]
) -> tuple[Mapping[str, Any], Any, Any]:
    """Fit and cross-fit training rows for Pandas-backed inputs."""
    fit_y = _extract_target(X, y, config.get("target_column"))
    if fit_y is None:
        logger.warning("TargetEncoder requires a target variable (y). Skipping.")
        return {}, X, y

    cols = _resolve_fit_cols(X, fit_y, config)
    if not cols:
        return {}, X, y

    encoder, encoded = _fit_transform_target_encoder(X[cols], fit_y, config)
    X_out, y_out = _replace_target_encoded_pandas(X, y, cols, encoded)
    return {"type": "target_encoder", "columns": cols, "encoder_object": encoder}, X_out, y_out


@NodeRegistry.register("TargetEncoder", TargetEncoderApplier)
@node_meta(
    id="TargetEncoder",
    name="Target Encoder",
    category="Preprocessing",
    description="Encode categorical features using target statistics.",
    params={"smooth": "auto", "target_type": "auto", "columns": []},
    learns_from_data=True,
)
class TargetEncoderCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> TargetEncoderArtifact:  # pylint: disable=arguments-differ
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

    def fit_transform_train(
        self, df: pd.DataFrame | SkyulfDataFrame | tuple, config: dict[str, Any]
    ) -> tuple[TargetEncoderArtifact, Any]:
        """Fit sklearn TargetEncoder and cross-fit the pipeline training rows."""
        if user_picked_no_columns(config):
            return {}, df

        artifact, transformed = fit_transform_train_dual_engine(
            df,
            config,
            polars_func=_target_fit_transform_train_polars,
            pandas_func=_target_fit_transform_train_pandas,
        )
        return cast(
            TargetEncoderArtifact,
            artifact,
        ), transformed

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: dict[str, Any],
    ) -> SkyulfSchema | None:
        # For binary/regression targets, the encoder replaces values in
        # source columns in place — same column names, dtype becomes float
        # (per-column dtype is best-effort so we don't bother rewriting it).
        #
        # For multiclass targets, the apply logic (see
        # ``_target_apply_polars``/``_target_apply_pandas``) drops the
        # original columns and creates ``{col}_cls{i}`` columns instead — the
        # number of classes is data-dependent and unknown here, so we can't
        # confidently predict the output columns. The default/"auto"
        # target_type is resolved to multiclass at fit time whenever y has
        # more than two classes, so we must also treat "auto" as unknown
        # rather than assuming binary/regression. Only the explicit
        # "binary"/"regression" config values are confidently in-place.
        if config.get("target_type", "auto") not in ("binary", "regression"):
            return None
        return input_schema


__all__ = ["TargetEncoderApplier", "TargetEncoderCalculator"]
