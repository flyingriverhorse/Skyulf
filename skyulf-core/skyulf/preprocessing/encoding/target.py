"""Target Encoder node (Calculator + Applier)."""

import logging
from collections.abc import Mapping
from numbers import Integral
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.utils.multiclass import type_of_target

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...types import DEFAULT_RANDOM_STATE
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import TargetEncoderArtifact
from .._category_keys import category_key_expr, category_keys_pandas, uses_category_keys
from .._fitted_validation import local_state_fields
from .._output_names import validate_generated_column_names
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine, fit_transform_train_dual_engine
from ._common import (
    _exclude_target_column,
    _extract_target,
    detect_categorical_columns,
    encoded_columns_schema,
)

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


def _target_subset_pandas(X: Any, cols: list[str]) -> pd.DataFrame:
    """Normalize raw feature scalars before sklearn infers any category dtype."""
    return pd.DataFrame({col: category_keys_pandas(X[col]) for col in cols})


def _target_subset_polars(X: Any, cols: list[str]) -> Any:
    """Use the same versioned missing and numeric keys on the Polars fit/apply paths."""
    return X.select([category_key_expr(col).alias(col) for col in cols])


def _validate_multiclass_names(X: Any, cols: list[str], n_classes: int) -> None:
    """Reject class-feature collisions while allowing names of dropped originals."""
    if n_classes > 1:
        validate_generated_column_names(
            X.columns,
            (f"{col}_cls{ci}" for col in cols for ci in range(n_classes)),
            dropped_columns=cols,
            node_name="TargetEncoder",
        )


def _replace_target_encoded_polars(
    X: Any, y: Any, valid_cols: list[str], encoded: Any
) -> tuple[Any, Any]:
    """Replace encoded columns in a Polars frame."""
    n_feats = len(valid_cols)
    if encoded.shape[1] == n_feats:
        new_cols = [pl.Series(col, encoded[:, i]) for i, col in enumerate(valid_cols)]
        return X.with_columns(new_cols), y

    n_classes = encoded.shape[1] // n_feats
    _validate_multiclass_names(X, valid_cols, n_classes)
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
    _validate_multiclass_names(X, valid_cols, n_classes)
    X_out = X_out.drop(columns=valid_cols)
    for fi, col in enumerate(valid_cols):
        for ci in range(n_classes):
            X_out[f"{col}_cls{ci}"] = encoded[:, fi * n_classes + ci]
    return X_out, y


def _target_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols, encoder = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    X_subset = (
        _target_subset_polars(X, valid_cols) if uses_category_keys(params) else X.select(valid_cols)
    )
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    encoded = encoder.transform(X_np) if len(X) else np.empty((0, len(encoder.encodings_)))
    return _replace_target_encoded_polars(X, y, valid_cols, encoded)


def _target_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols, encoder = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    X_subset = _target_subset_pandas(X, valid_cols) if uses_category_keys(params) else X[valid_cols]
    X_input = X_subset.to_numpy() if hasattr(X_subset, "to_numpy") else X_subset
    encoded = encoder.transform(X_input) if len(X) else np.empty((0, len(encoder.encodings_)))
    return _replace_target_encoded_pandas(X, y, valid_cols, encoded)


def _validate_target_categories(encoder: Any, count: int) -> None:
    """Check feature/category axes without interpreting or coercing learned scalar labels."""
    if encoder.n_features_in_ != count:
        raise ValueError("Fitted target encoder feature count disagrees.")
    categories = encoder.categories_
    if not isinstance(categories, (list, tuple)) or len(categories) != count:
        raise ValueError("Fitted target categories must align with selected columns.")
    if any(
        not isinstance(values, np.ndarray) or values.ndim != 1 or not len(values)
        for values in categories
    ):
        raise ValueError("Fitted target categories must be nonempty vectors.")


def _target_output_width(encoder: Any) -> int:
    """Validate saved class labels and mean dimensions without deriving new statistics."""
    if encoder.target_type_ not in ("binary", "continuous", "multiclass"):
        raise ValueError("Unknown fitted target encoder target type.")
    classes = encoder.classes_
    if encoder.target_type_ != "continuous" and (
        not isinstance(classes, np.ndarray) or classes.ndim != 1 or not len(classes)
    ):
        raise ValueError("Fitted target classes must be a nonempty vector.")
    width = len(classes) if encoder.target_type_ == "multiclass" else 1
    mean = np.asarray(encoder.target_mean_)
    shape = (width,) if encoder.target_type_ == "multiclass" else ()
    if mean.shape != shape or mean.dtype.kind not in "fi":
        raise ValueError("Fitted target mean disagrees with the target output shape.")
    return width


def _validate_target_statistics(encoder: Any) -> None:
    """Bind full-data inference maps to their saved feature and target-class dimensions."""
    width = _target_output_width(encoder)
    encodings = encoder.encodings_
    if (
        not isinstance(encodings, (list, tuple))
        or len(encodings) != len(encoder.categories_) * width
    ):
        raise ValueError("Fitted target encoding count disagrees with learned categories.")
    for index, values in enumerate(encodings):
        if (
            not isinstance(values, np.ndarray)
            or values.dtype.kind not in "fi"
            or values.shape != (len(encoder.categories_[index // width]),)
        ):
            raise ValueError("Fitted target encoding vectors disagree with category counts.")


class TargetEncoderApplier(BaseApplier):
    """Replace categorical columns with the target statistics the fitted encoder learned.

    Both engines transform through the same sklearn ``TargetEncoder`` carried in
    the artifact, so parity comes from sharing it. The output shape follows the
    target type: binary and regression targets are encoded in place, while a
    multiclass target drops the originals and emits one ``{col}_cls{i}`` column
    per class. Multiclass output names must not collide with retained columns;
    conflicts raise ``ValueError`` before any input values are replaced.
    New artifacts use canonical missing and numeric feature keys on both
    engines; artifacts without a key version retain raw sklearn replay.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect the saved full-data encoder while leaving cross-fitted training values alone."""
        fields = {"type", "columns", "encoder_object"}
        if isinstance(raw, dict):
            fields.update({"category_key_version"}.intersection(raw))
        if not local_state_fields(raw, "target_encoder", fields, allow_empty=True):
            return raw
        uses_category_keys(raw)
        columns = raw["columns"]
        if not isinstance(columns, (list, tuple)) or any(not isinstance(c, str) for c in columns):
            raise ValueError("Fitted target encoder columns must be ordered strings.")
        encoder = raw["encoder_object"]
        required = {
            "n_features_in_",
            "categories_",
            "target_type_",
            "classes_",
            "target_mean_",
            "encodings_",
        }
        if type(encoder) is not TargetEncoder or not required.issubset(vars(encoder)):
            raise ValueError("Expected a fitted TargetEncoder.")
        if any(callable(value) for value in vars(encoder).values()):
            raise ValueError("Overridden target estimator methods are unsupported.")
        _validate_target_categories(encoder, len(columns))
        _validate_target_statistics(encoder)
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe saved target-statistic lookups without refitting or admitting workers."""
        if engine not in ("pandas", "polars"):
            return None
        TargetEncoderApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch to the engine-specific transform, forwarding ``(X, y)`` only when present."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"polars": _target_apply_polars, "pandas": _target_apply_pandas},
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


def _cross_fit_options(config: dict) -> tuple[int, bool, int | None]:
    """Validate user controls while retaining the historical deterministic defaults."""
    cv = config.get("cv", 5)
    shuffle = config.get("shuffle", True)
    seed = config.get("random_state", DEFAULT_RANDOM_STATE)
    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, Integral) or cv < 2:
        raise ValueError("TargetEncoder cv must be an integer of at least 2.")
    if type(shuffle) is not bool:
        raise ValueError("TargetEncoder shuffle must be a boolean.")
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise ValueError("TargetEncoder random_state must be a nonnegative integer or None.")
        if not 0 <= seed < 2**32:
            raise ValueError("TargetEncoder random_state must be in [0, 2**32 - 1].")
        seed = int(seed)
    return int(cv), shuffle, seed if shuffle else None


def _build_target_encoder(
    X_subset: Any, y: Any, config: dict[str, Any]
) -> tuple[TargetEncoder, Any]:
    """Build the sklearn encoder and normalize the target array."""
    _ = X_subset
    target_type = config.get("target_type", "auto")
    cv, shuffle, seed = _cross_fit_options(config)
    encoder = TargetEncoder(
        smooth=config.get("smooth", "auto"),
        target_type=target_type,
        cv=cv,
        shuffle=shuffle,
        random_state=seed,
    )
    y_np = _y_to_numpy(y)

    # If y is object/string and target_type is multiclass (or auto with many classes),
    # label-encode y to integers so sklearn can fit without complaints.
    if hasattr(y_np, "dtype") and y_np.dtype == object and target_type in ("multiclass", "auto"):
        le = LabelEncoder()
        y_np = le.fit_transform(y_np)

    return encoder, y_np


def _resolve_target_encoder_training_cv(y_np: Any, target_type: str, requested_cv: int = 5) -> int:
    """Return the highest deterministic non-leaky fold count for training rows."""
    y_array = np.asarray(y_np)
    n_samples = len(y_array)
    if n_samples < 2:
        raise ValueError(
            "TargetEncoder pipeline training requires at least 2 training rows for "
            f"leakage-safe cross-fitting; got {n_samples}."
        )

    safe_cv = min(requested_cv, n_samples)
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
    encoder.cv = _resolve_target_encoder_training_cv(
        y_np, str(config.get("target_type", "auto")), encoder.cv
    )
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

    encoder = _fit_target_encoder(_target_subset_polars(X, cols), y, config)
    if encoder.target_type_ == "multiclass" and encoder.classes_ is not None:
        _validate_multiclass_names(X, cols, len(encoder.classes_))
    return {
        "type": "target_encoder",
        "columns": cols,
        "encoder_object": encoder,
        "category_key_version": 1,
    }


def _target_fit_pandas(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    y = _extract_target(X, y, config.get("target_column"))
    if y is None:
        logger.warning("TargetEncoder requires a target variable (y). Skipping.")
        return {}

    cols = _resolve_fit_cols(X, y, config)
    if not cols:
        return {}

    encoder = _fit_target_encoder(_target_subset_pandas(X, cols), y, config)
    if encoder.target_type_ == "multiclass" and encoder.classes_ is not None:
        _validate_multiclass_names(X, cols, len(encoder.classes_))
    return {
        "type": "target_encoder",
        "columns": cols,
        "encoder_object": encoder,
        "category_key_version": 1,
    }


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

    encoder, encoded = _fit_transform_target_encoder(_target_subset_polars(X, cols), fit_y, config)
    X_out, y_out = _replace_target_encoded_polars(X, y, cols, encoded)
    return (
        {
            "type": "target_encoder",
            "columns": cols,
            "encoder_object": encoder,
            "category_key_version": 1,
        },
        X_out,
        y_out,
    )


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

    encoder, encoded = _fit_transform_target_encoder(_target_subset_pandas(X, cols), fit_y, config)
    X_out, y_out = _replace_target_encoded_pandas(X, y, cols, encoded)
    return (
        {
            "type": "target_encoder",
            "columns": cols,
            "encoder_object": encoder,
            "category_key_version": 1,
        },
        X_out,
        y_out,
    )


@NodeRegistry.register("TargetEncoder", TargetEncoderApplier)
@node_meta(
    id="TargetEncoder",
    name="Target Encoder",
    category="Preprocessing",
    description="Encode categorical features using target statistics.",
    params={
        "smooth": "auto",
        "target_type": "auto",
        "columns": [],
        "cv": 5,
        "shuffle": True,
        "random_state": DEFAULT_RANDOM_STATE,
    },
    learns_from_data=True,
)
class TargetEncoderCalculator(BaseCalculator):
    """Fit a sklearn ``TargetEncoder`` against ``y``, cross-fitting the training rows.

    A target is mandatory: with no ``y`` the node logs a warning and returns an
    empty artifact rather than failing. The target column is excluded from the
    encoded set, since encoding it against itself is degenerate. This calculator
    also satisfies ``TrainTransformCalculatorProtocol``, so the training split
    receives out-of-fold encodings — a row's own target never leaks into its
    feature — with the fold count capped by the smallest class count, while
    held-out splits go through the plain applier.
    """

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> TargetEncoderArtifact:  # pylint: disable=arguments-differ
        """Short-circuit an explicit empty column selection, else fit on the frame's own engine."""
        if user_picked_no_columns(config):
            return {}
        return cast(
            TargetEncoderArtifact,
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                {"polars": _target_fit_polars, "pandas": _target_fit_pandas},
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
            {
                "polars": _target_fit_transform_train_polars,
                "pandas": _target_fit_transform_train_pandas,
            },
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
        """Describe explicit floating features for binary or continuous targets.

        Automatic category selection and multiclass feature expansion depend on
        training values, so they return ``None`` for runtime introspection.
        An explicit empty selection is always a no-op.
        """
        columns = config.get("columns")
        if columns == []:
            return input_schema
        if config.get("target_type", "auto") not in ("binary", "continuous"):
            return None
        if columns is not None:
            columns = [col for col in columns if col != config.get("target_column")]
        return encoded_columns_schema(input_schema, columns, "float64")


__all__ = ["TargetEncoderApplier", "TargetEncoderCalculator"]
