"""Ordinal Encoder node (Calculator + Applier)."""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import OrdinalEncoder

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import OrdinalArtifact
from .._category_keys import (
    category_key_expr,
    category_keys_pandas,
    category_order_keys,
    uses_category_keys,
)
from .._fitted_validation import local_state_fields
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import _parse_categories_order, detect_categorical_columns
from ._target import apply_target_encoder, fit_target_encoder

# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _resolve_apply_inputs(X: Any, params: dict[str, Any]) -> tuple[list[str], Any, dict[str, Any]]:
    """Return ``(valid_cols, feature_encoder, target_encoders)``."""
    cols = params.get("columns", [])
    encoder = params.get("encoder_object")
    target_encoders: dict[str, Any] = params.get("encoders", {})
    valid_cols = [c for c in cols if c in X.columns]
    return valid_cols, encoder, target_encoders


def _apply_features_polars(
    X: Any, valid_cols: list[str], encoder: Any, canonical_keys: bool = False
) -> Any:
    # `fill_null("nan")` mirrors the pandas path's `.astype(str)` ("NaN" ->
    # "nan"), so polars nulls reuse the fitted "nan" class instead of drifting
    # into unknown_value (F-07).
    X_subset = X.select(valid_cols).select(
        [
            category_key_expr(c) if canonical_keys else pl.col(c).cast(pl.Utf8).fill_null("nan")
            for c in valid_cols
        ]
    )
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    encoded = (
        encoder.transform(X_np) if len(X) else np.empty((0, len(valid_cols)), dtype=encoder.dtype)
    )
    new_cols_pl = [pl.Series(col, encoded[:, i]) for i, col in enumerate(valid_cols)]
    return X.with_columns(new_cols_pl)


def _apply_features_pandas(
    X: Any, valid_cols: list[str], encoder: Any, canonical_keys: bool = False
) -> Any:
    X_out = X.copy()
    X_subset = _subset_to_str_pandas(X_out, valid_cols, canonical_keys)
    X_input = X_subset.to_numpy() if hasattr(X_subset, "to_numpy") else X_subset
    X_out[valid_cols] = (
        encoder.transform(X_input)
        if len(X)
        else np.empty((0, len(valid_cols)), dtype=encoder.dtype)
    )
    return X_out


def _target_to_str_array(y: Any) -> Any:
    """Best-effort conversion of ``y`` to a 2-D string numpy array (shape ``(n, 1)``).

    For Polars Series, casts to Utf8 and fills nulls with the literal "nan"
    string *before* calling ``.to_numpy()``. Polars' native ``.to_numpy()``
    casts integer columns containing nulls to float (NaN), which flips the
    string representation of every value (e.g. "1" -> "1.0") depending on
    whether nulls happen to be present in a given batch -- doing the string
    cast in Polars first keeps fit and apply representations identical
    regardless of null presence.

    Note: this returns a 2-D array (unlike ``label.py``'s ``_y_to_str_array``,
    which returns a flat 1-D array for sklearn's ``LabelEncoder``) because
    sklearn's ``OrdinalEncoder.fit``/``.transform`` expect a 2-D
    ``(n_samples, n_features)`` array, even for a single target column.
    """
    # Polars is a hard runtime dependency of this package (see setup.py) and is
    # already imported at process startup via `engines/polars_engine.py`'s
    # module-level import, so importing it here is not a lazy/optional-import
    # pattern -- it's just kept local to this function (matching the sibling
    # helpers in this module) to use `isinstance(y, pl.Series)` rather than
    # duck-typing on `hasattr(y, "fill_null")`, which could misroute any
    # unrelated object that happens to expose a `fill_null` method.
    if isinstance(y, pl.Series):
        return y.cast(pl.Utf8).fill_null("nan").to_numpy().reshape(-1, 1)
    raw = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    arr = raw.astype(str)
    # astype(str) renders object-None as "None" while the polars branch fills
    # with "nan" — collapse so both fit the same missing class (F-07).
    arr[arr == "None"] = "nan"
    return arr.reshape(-1, 1)


def _apply_target_polars(y: Any, enc: OrdinalEncoder) -> Any:
    if len(y) == 0:
        return pl.Series(getattr(y, "name", "target"), [], dtype=pl.Float32)
    y_arr = _target_to_str_array(y)
    encoded = enc.transform(y_arr).flatten()
    y_name = y.name if hasattr(y, "name") else "target"
    return pl.Series(y_name, encoded.astype(np.float32))


def _apply_target_pandas(y: Any, enc: OrdinalEncoder) -> Any:
    index = getattr(y, "index", None)
    if callable(index):
        index = None
    if len(y) == 0:
        return pd.Series(index=index, name=getattr(y, "name", None), dtype="float32")
    y_arr = _target_to_str_array(y)
    encoded = enc.transform(y_arr).flatten()
    return pd.Series(
        encoded,
        index=index,
        name=y.name if hasattr(y, "name") else None,
    )


def _ordinal_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols, encoder, target_encoders = _resolve_apply_inputs(X, params)
    if not valid_cols and "__target__" not in target_encoders:
        return X, y

    X_out: Any = X
    if valid_cols and encoder:
        X_out = _apply_features_polars(X, valid_cols, encoder, uses_category_keys(params))

    y_out = y
    if y is not None and "__target__" in target_encoders:
        y_out = _apply_target_polars(y, target_encoders["__target__"])
    return X_out, y_out


def _ordinal_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols, encoder, target_encoders = _resolve_apply_inputs(X, params)
    if not valid_cols and "__target__" not in target_encoders:
        return X, y

    X_out: Any = X
    if valid_cols and encoder:
        X_out = _apply_features_pandas(X, valid_cols, encoder, uses_category_keys(params))

    y_out = y
    if y is not None and "__target__" in target_encoders:
        y_out = _apply_target_pandas(y, target_encoders["__target__"])
    return X_out, y_out


def _validate_ordinal_encoder(encoder: Any, count: int) -> list[int]:
    """Inspect the actual string-key estimator without running sklearn transformations."""
    required = {
        "categories_",
        "n_features_in_",
        "dtype",
        "handle_unknown",
        "unknown_value",
        "_missing_indices",
        "_infrequent_enabled",
    }
    if type(encoder) is not OrdinalEncoder or not required.issubset(vars(encoder)):
        raise ValueError("Expected a fitted OrdinalEncoder.")
    if any(callable(value) for key, value in vars(encoder).items() if key != "dtype"):
        raise ValueError("Overridden ordinal estimator methods are unsupported.")
    if encoder.n_features_in_ != count:
        raise ValueError("Fitted ordinal feature count disagrees.")
    categories = encoder.categories_
    if not isinstance(categories, (list, tuple)) or len(categories) != count:
        raise ValueError("Fitted ordinal categories must match selected columns.")
    counts = [_validate_ordinal_categories(values) for values in categories]
    _validate_ordinal_options(encoder, counts)
    return counts


def _validate_ordinal_categories(values: Any) -> int:
    """Retain fitted category order and NumPy strings while checking vocabulary shape."""
    if not isinstance(values, np.ndarray) or values.ndim != 1 or not len(values):
        raise ValueError("Fitted ordinal categories must be nonempty vectors.")
    if any(not isinstance(value, str) for value in values) or len(set(values)) != len(values):
        raise ValueError("Fitted ordinal categories must contain unique string keys.")
    return len(values)


def _validate_ordinal_options(encoder: Any, counts: list[int]) -> None:
    """Check the saved fixed lookup policy rather than refitting its configured categories."""
    if encoder._infrequent_enabled or encoder._missing_indices != {}:
        raise ValueError("Unsupported ordinal fitted grouping or missing-category state.")
    if np.dtype(encoder.dtype) != np.dtype(np.float32):
        raise ValueError("Fitted ordinal dtype disagrees.")
    if encoder.handle_unknown not in ("error", "use_encoded_value"):
        raise ValueError("Unknown fitted ordinal lookup policy.")
    if encoder.handle_unknown == "error":
        return
    _validate_ordinal_unknown(encoder.unknown_value, counts)


def _validate_ordinal_unknown(value: Any, counts: list[int]) -> None:
    """Require the saved unknown code to remain distinct from fitted category positions."""
    if not isinstance(value, Real):
        raise ValueError("Fitted ordinal unknown value must be an integer or NaN.")
    if not isinstance(value, Integral) and not np.isnan(value):
        raise ValueError("Fitted ordinal unknown value must be an integer or NaN.")
    if any(0 <= value < count for count in counts):
        raise ValueError("Fitted ordinal unknown value overlaps known category codes.")


def _validate_ordinal_features(raw: dict) -> None:
    """Bind feature selection, fitted estimator and recorded vocabulary sizes."""
    columns = raw["columns"]
    if not isinstance(columns, (list, tuple)) or any(not isinstance(c, str) for c in columns):
        raise ValueError("Fitted ordinal columns must be ordered strings.")
    counts = _validate_ordinal_encoder(raw["encoder_object"], len(columns)) if columns else []
    if not columns and raw["encoder_object"] is not None:
        raise ValueError("An empty ordinal selection must not carry a feature encoder.")
    if (
        not isinstance(raw["categories_count"], (list, tuple))
        or list(raw["categories_count"]) != counts
    ):
        raise ValueError("Fitted ordinal counts disagree with learned categories.")


class OrdinalEncoderApplier(BaseApplier):
    """Replace categorical values in place with fitted ordinal indices, target included.

    Features use the shared ``encoder_object`` and versioned scalar keys; ``y``
    uses a separate ``__target__`` encoder with its existing string-label rules.
    Older feature artifacts retain their original string lookup. An unseen
    value becomes ``unknown_value`` unless ``handle_unknown`` is ``"error"``.
    Encoded output is ``float32``, not integer.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect feature and optional target encoders without changing lookup provenance."""
        fields = {"type", "columns", "encoder_object", "encoders", "categories_count"}
        if isinstance(raw, dict):
            fields.update({"category_key_version", "target_column"}.intersection(raw))
        if not local_state_fields(raw, "ordinal", fields, allow_empty=True):
            return raw
        uses_category_keys(raw)
        _validate_ordinal_features(raw)
        OrdinalEncoderApplier._validate_targets(raw)
        return raw

    @staticmethod
    def _validate_targets(raw: dict) -> None:
        """Keep embedded-target routing separate from selected feature encoders."""
        encoders = raw["encoders"]
        if type(encoders) is not dict or set(encoders) - {"__target__"}:
            raise ValueError("Unexpected fitted ordinal target encoders.")
        if "__target__" in encoders:
            _validate_ordinal_encoder(encoders["__target__"], 1)
        if "target_column" in raw and (
            not isinstance(raw["target_column"], str)
            or raw["target_column"] in raw["columns"]
            or "__target__" not in encoders
        ):
            raise ValueError("Fitted ordinal target routing disagrees with its encoders.")

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe saved ordinal lookups without enabling worker execution."""
        if engine not in ("pandas", "polars"):
            return None
        OrdinalEncoderApplier.validate_inference_state(state)
        # Legacy pandas datetime string formatting can depend on neighboring rows.
        legacy_pandas = (
            engine == "pandas" and state.get("columns") and not uses_category_keys(state)
        )
        context = "global" if legacy_pandas else "row"
        return ExecutionCapability(engine, "apply", "local", "preserve", context)

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch to the engine-specific encode, forwarding ``(X, y)`` only when ``y`` exists."""
        return apply_target_encoder(
            X,
            y,
            params,
            {"polars": _ordinal_apply_polars, "pandas": _ordinal_apply_pandas},
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _resolve_handle_unknown(config: dict[str, Any]) -> str:
    """OrdinalEncoder only accepts 'error' or 'use_encoded_value'."""
    raw = config.get("handle_unknown", "use_encoded_value")
    return raw if raw in ("error", "use_encoded_value") else "use_encoded_value"


def _make_ordinal_encoder(
    categories: str | list[list[str]], handle_unknown: str, unknown_value: Any
) -> OrdinalEncoder:
    return OrdinalEncoder(
        categories=categories,
        handle_unknown=handle_unknown,
        unknown_value=unknown_value if handle_unknown == "use_encoded_value" else None,
        dtype=np.float32,
    )


def _fit_target_encoder(
    y_series: Any,
    categories: str | list[list[str]],
    handle_unknown: str,
    unknown_value: Any,
) -> OrdinalEncoder:
    """Fit a one-column OrdinalEncoder on ``y``."""
    enc = _make_ordinal_encoder(categories, handle_unknown, unknown_value)
    y_arr = _target_to_str_array(y_series)
    enc.fit(y_arr)
    return enc


def _resolve_target_categories(raw_order: Any, n_features: int) -> str | list[list[str]]:
    """Slice the per-target row out of the categories_order table."""
    parsed = _parse_categories_order(raw_order, n_features + 1)
    if isinstance(parsed, list) and len(parsed) == n_features + 1:
        return [parsed[-1]]
    return "auto"


def _build_subset_polars(X: Any, feature_cols: list[str], canonical_keys: bool = False) -> Any:
    # Same fill_null("nan") normalisation as _apply_features_polars so fit
    # categories and apply-time strings agree for missing values (F-07).
    return X.select(feature_cols).select(
        [
            category_key_expr(c) if canonical_keys else pl.col(c).cast(pl.Utf8).fill_null("nan")
            for c in feature_cols
        ]
    )


def _fit_feature_encoder(
    X_subset: Any,
    feature_cols: list[str],
    config: dict[str, Any],
) -> tuple[OrdinalEncoder, list[int]]:
    cats = _parse_categories_order(config.get("categories_order"), len(feature_cols))
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    if isinstance(cats, list):
        cats = [category_order_keys(values, X_np[:, i]) for i, values in enumerate(cats)]
    enc = _make_ordinal_encoder(
        cats, _resolve_handle_unknown(config), config.get("unknown_value", -1)
    )
    enc.fit(X_np)
    counts = [len(c) for c in enc.categories_]
    return enc, counts


def _ordinal_fit_no_columns(y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    """Fit-time fallback when the user picked no feature columns."""
    target_encoders: dict[str, Any] = {}
    if y is not None:
        cats_y = _parse_categories_order(config.get("categories_order"), 1)
        target_encoders["__target__"] = _fit_target_encoder(
            y, cats_y, _resolve_handle_unknown(config), config.get("unknown_value", -1)
        )
    return {
        "type": "ordinal",
        "columns": [],
        "encoder_object": None,
        "encoders": target_encoders,
        "categories_count": [],
    }


def _should_encode_target(X: Any, y: Any, config: dict[str, Any]) -> bool:
    """True iff ``y`` exists and the configured target name is in `columns` but not in ``X``."""
    if y is None:
        return False
    target_col = config.get("target_column") or getattr(y, "name", None)
    if not target_col:
        return False
    cols_raw: list[str] = config.get("columns") or []
    return target_col in cols_raw and target_col not in X.columns


def _maybe_fit_features(
    X: Any, feature_cols: list[str], config: dict[str, Any], build_subset: Any
) -> tuple["OrdinalEncoder | None", list[int]]:
    if not feature_cols:
        return None, []
    return _fit_feature_encoder(build_subset(X, feature_cols), feature_cols, config)


def _maybe_fit_target_block(
    y: Any, n_features: int, config: dict[str, Any], encode_target: bool
) -> dict[str, Any]:
    if not (encode_target and y is not None):
        return {}
    cats_y = _resolve_target_categories(config.get("categories_order"), n_features)
    enc = _fit_target_encoder(
        y, cats_y, _resolve_handle_unknown(config), config.get("unknown_value", -1)
    )
    return {"__target__": enc}


def _ordinal_fit_dispatch(
    X: Any,
    y: Any,
    config: dict[str, Any],
    build_subset: Any,
) -> Mapping[str, Any]:
    """Engine-agnostic fit body. ``build_subset(X, feature_cols) -> X_subset``."""
    if user_picked_no_columns(config):
        return _ordinal_fit_no_columns(y, config)

    encode_target = _should_encode_target(X, y, config)
    cols = resolve_columns(X, config, detect_categorical_columns)
    feature_cols = [c for c in cols if c in X.columns]
    if not feature_cols and not encode_target:
        return {}

    feature_config, target_config = _encoding_orders(config, feature_cols, encode_target)
    feature_encoder, counts = _maybe_fit_features(X, feature_cols, feature_config, build_subset)
    target_encoders = _maybe_fit_target_block(y, len(feature_cols), target_config, encode_target)

    return {
        "type": "ordinal",
        "columns": feature_cols,
        "encoder_object": feature_encoder,
        "encoders": target_encoders,
        "categories_count": counts,
        "category_key_version": 1,
    }


def _encoding_orders(config: dict, features: list[str], encode_target: bool) -> tuple[dict, dict]:
    """Keep explicit category rows attached to column names when separating the target."""
    columns = config.get("columns") or []
    parsed = _parse_categories_order(config.get("categories_order"), len(columns))
    target = config.get("target_column")
    if not isinstance(parsed, list):
        return config, config
    orders = dict(zip(columns, parsed, strict=True))
    feature_rows = [",".join(orders[column]) for column in features]
    feature_config = {**config, "categories_order": feature_rows}
    if not encode_target or target not in columns:
        return feature_config, config
    return (
        feature_config,
        {**config, "categories_order": [*feature_rows, ",".join(orders[target])]},
    )


def _ordinal_fit_polars(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    return _ordinal_fit_dispatch(
        X, y, config, lambda frame, cols: _build_subset_polars(frame, cols, True)
    )


def _subset_to_str_pandas(X: Any, feature_cols: list[str], canonical_keys: bool = False) -> Any:
    # astype(str) renders float NaN as "nan" but object-None as "None";
    # collapse the latter so every missing representation shares one class
    # (F-07 parity with the polars fill_null("nan") path).
    if canonical_keys:
        return pd.DataFrame({col: category_keys_pandas(X[col]) for col in feature_cols})
    return X[feature_cols].astype(str).replace("None", "nan")


def _ordinal_fit_pandas(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    return _ordinal_fit_dispatch(
        X, y, config, lambda frame, cols: _subset_to_str_pandas(frame, cols, True)
    )


@NodeRegistry.register("OrdinalEncoder", OrdinalEncoderApplier)
@node_meta(
    id="OrdinalEncoder",
    name="Ordinal Encoder",
    category="Preprocessing",
    description="Encodes categorical features as an integer array.",
    params={
        "columns": [],
        "handle_unknown": "use_encoded_value",
        "unknown_value": -1,
        "categories_order": "",
    },
    learns_from_data=True,
)
class OrdinalEncoderCalculator(BaseCalculator):
    """Fit an ordinal encoder for the features and, when split out, for the target.

    Picking no feature columns is *not* a no-op here — unlike its sibling
    encoders, this node still fits a target encoder when ``y`` is available.
    The target is encoded only when ``y`` exists and the configured target name
    is listed in ``columns`` but absent from ``X``, i.e. it has already been
    split away. ``categories_order`` supplies an explicit ordering, and the
    target reads the last row of an ``n_features + 1`` row table.
    """

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Pass the input schema through: encoded columns keep their name and position."""
        # Ordinal encoding replaces categorical values with ints in place;
        # column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> OrdinalArtifact:  # pylint: disable=arguments-differ
        """Fit the encoders on the frame's own engine, forwarding ``y`` when present."""
        return cast(
            OrdinalArtifact,
            fit_target_encoder(
                X,
                y,
                config,
                {"polars": _ordinal_fit_polars, "pandas": _ordinal_fit_pandas},
            ),
        )


__all__ = ["OrdinalEncoderApplier", "OrdinalEncoderCalculator"]
