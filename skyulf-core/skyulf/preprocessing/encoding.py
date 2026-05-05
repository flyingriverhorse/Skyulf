import logging
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)

from ..utils import (
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
    user_picked_no_columns,
)
from .base import BaseApplier, BaseCalculator
from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..engines import EngineName, SkyulfDataFrame, get_engine
from ..engines.sklearn_bridge import SklearnBridge

logger = logging.getLogger(__name__)


def _parse_categories_order(
    raw: Any, n_cols: int
) -> Union[str, List[List[str]]]:
    """Convert the frontend categories_order string/list into sklearn-compatible categories.

    The frontend sends a newline-separated string where each line holds
    comma-separated category values for one column (in column-selection order).
    If the number of non-empty lines matches n_cols, a list-of-lists is returned.
    Any other case falls back to "auto".
    """
    if not raw:
        return "auto"
    if isinstance(raw, str):
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    elif isinstance(raw, list):
        lines = [str(ln).strip() for ln in raw if str(ln).strip()]
    else:
        return "auto"

    if not lines or len(lines) != n_cols:
        # Partial specification is ambiguous — fall back to auto-detection.
        return "auto"

    return [[v.strip() for v in line.split(",") if v.strip()] for line in lines]


# Encoders that destroy the original column structure (create N new columns).
# These must NEVER be applied to a target column — it would break downstream
# Feature/Target Split and model training nodes.
_COLUMN_DESTROYING_ENCODERS = frozenset(
    [
        "OneHotEncoder",
        "DummyEncoder",
        "HashEncoder",
        "TargetEncoder",
    ]
)

# Encoders that map values in-place and preserve the column name.
_TARGET_SAFE_ENCODERS = frozenset(["LabelEncoder", "OrdinalEncoder"])


def _exclude_target_column(
    columns: List[str],
    config: Dict[str, Any],
    encoder_name: str,
    y: Any = None,
) -> List[str]:
    """Remove the target column from the encoding list for column-destroying encoders.

    Detects the target column from:
      1. ``config['target_column']`` (explicit pipeline param)
      2. The ``name`` attribute of ``y`` (if it's a named Series)

    Returns the filtered column list and logs a warning when a column is removed.
    """
    if encoder_name not in _COLUMN_DESTROYING_ENCODERS:
        return columns

    target_col: str | None = config.get("target_column")
    if target_col is None and y is not None:
        target_col = getattr(y, "name", None)

    if target_col and target_col in columns:
        logger.warning(
            f"{encoder_name}: Excluding target column '{target_col}' from encoding. "
            f"{encoder_name} would replace the column with multiple derived columns, "
            "breaking downstream Feature/Target Split and model training. "
            "Use LabelEncoder or OrdinalEncoder for target columns instead."
        )
        columns = [c for c in columns if c != target_col]

    return columns


def detect_categorical_columns(df: Any) -> List[str]:
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        import polars as pl

        # Polars dtypes
        df_pl: Any = df
        return list(
            c
            for c, t in zip(df_pl.columns, df_pl.dtypes)
            if t in [pl.Utf8, pl.Categorical, pl.Object]
        )
    # Pandas
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# --- OneHot Encoder ---


class OneHotEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        if not params or not params.get("columns"):
            return pack_pipeline_output(X, y, is_tuple)

        cols = params["columns"]
        encoder = params.get("encoder_object")
        feature_names = params.get("feature_names")
        drop_original = params.get("drop_original", True)
        include_missing = params.get("include_missing", False)

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X

            try:
                X_subset = X_pl.select(valid_cols)
                if include_missing:
                    X_subset = X_subset.fill_null("__mlops_missing__")

                X_np, _ = SklearnBridge.to_sklearn(X_subset)
                encoded_array = encoder.transform(X_np)

                if hasattr(encoded_array, "toarray"):
                    encoded_array = encoded_array.toarray()

                encoded_df = pl.DataFrame(encoded_array, schema=feature_names)
                X_out = pl.concat([X_pl, encoded_df], how="horizontal")

                if drop_original:
                    X_out = X_out.drop(valid_cols)

                return pack_pipeline_output(X_out, y, is_tuple)
            except Exception as e:
                logger.error(f"OneHot Encoding failed: {e}")
                return pack_pipeline_output(X, y, is_tuple)

        # Pandas Path
        X_out = X.copy()

        # Ensure all expected columns are present for the encoder
        # If some columns are missing in input, we fill them with NaN
        # This allows encoder.transform to receive the correct number of features

        X_subset = X_out[valid_cols]

        if include_missing:
            X_subset = X_subset.fillna("__mlops_missing__")

        # Transform
        try:
            # Fix for "X has feature names, but ... was fitted without feature names"
            # We must pass the same format as fit. Fit used X_np (numpy).
            # So we convert X_subset (DataFrame) to numpy/values.
            if hasattr(X_subset, "values"):
                X_input = X_subset.values
            else:
                X_input = X_subset

            encoded_array = encoder.transform(X_input)

            if hasattr(encoded_array, "toarray"):
                encoded_array = encoded_array.toarray()
            elif hasattr(encoded_array, "values"):
                # If sklearn is configured to output pandas, we get a DataFrame.
                # We need the underlying numpy array to create a new DataFrame with our custom feature names.
                encoded_array = encoded_array.values

            # Create DataFrame from encoded array
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X_out.index)

            # Concatenate
            X_out = pd.concat(cast(Any, [X_out, encoded_df]), axis=1)

            # Drop original columns
            if drop_original:
                X_out = X_out.drop(columns=valid_cols)

        except Exception as e:
            logger.error(f"OneHot Encoding failed: {e}")
            # If encoding fails (e.g. new categories with handle_unknown='error'), we might just return original

        return pack_pipeline_output(X_out, y, is_tuple)


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
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "OneHotEncoder", y)

        if not cols:
            return {}

        # Config
        drop = "first" if config.get("drop_first", False) else None
        max_categories = config.get("max_categories", 20)  # Default limit to prevent explosion
        handle_unknown = "ignore" if config.get("handle_unknown", "ignore") == "ignore" else "error"
        prefix_separator = config.get("prefix_separator", "_")
        drop_original = config.get("drop_original", True)
        include_missing = config.get("include_missing", False)

        # Handle missing values for fit if requested
        if engine.name == EngineName.POLARS:
            X_pl: Any = X
            X_subset = X_pl.select(cols)
            if include_missing:
                X_subset = X_subset.fill_null("__mlops_missing__")
        else:
            X_subset = X[cols]
            if include_missing:
                X_subset = X_subset.fillna("__mlops_missing__")

        X_np, _ = SklearnBridge.to_sklearn(X_subset)

        # We use sklearn's OneHotEncoder
        # Note: sparse_output=False to return dense arrays for pandas
        encoder = OneHotEncoder(
            drop=drop,
            max_categories=max_categories,
            handle_unknown=handle_unknown,
            sparse_output=False,
            dtype=np.int8,  # Save memory
        )

        # Fix for "X has feature names, but OneHotEncoder was fitted without feature names"
        # Since we convert to numpy for fit, we lose feature names.
        # We can explicitly set feature names if available, but it's easier to just fit on numpy
        # and ensure transform also receives numpy (which we do via SklearnBridge).
        # However, SklearnBridge might return a DataFrame if configured? No, it returns (data, feature_names).
        # The issue is likely that SklearnBridge returns a Numpy array, so fit sees no names.
        # But in apply (Pandas path), we pass X_subset (DataFrame) directly to transform.

        encoder.fit(X_np)

        # Check for columns that produced no features
        if hasattr(encoder, "categories_"):
            for i, col in enumerate(cols):
                n_cats = len(encoder.categories_[i])
                # If drop='first' and n_cats == 1, we get 0 features (1-1=0)
                # If n_cats == 0, we get 0 features

                # We can check the actual output feature names to be sure, but checking categories is a good proxy.
                # Sklearn's get_feature_names_out handles the drop logic.

                if n_cats == 0:
                    logger.warning(
                        f"OneHotEncoder: Column '{col}' has 0 categories (empty or all missing). It will be dropped."
                    )
                elif drop == "first" and n_cats == 1:
                    logger.warning(
                        f"OneHotEncoder: Column '{col}' has only 1 category ('{encoder.categories_[i][0]}') "
                        "and 'Drop First' is enabled. This results in 0 encoded features. "
                        "The column will be effectively dropped."
                    )

        return {
            "type": "onehot",
            "columns": cols,
            "encoder_object": encoder,
            "feature_names": encoder.get_feature_names_out(cols).tolist(),
            "prefix_separator": prefix_separator,
            "drop_original": drop_original,
            "include_missing": include_missing,
        }


# --- Ordinal Encoder ---


class OrdinalEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        encoder = params.get("encoder_object")
        target_encoders: Dict[str, Any] = params.get("encoders", {})

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols and "__target__" not in target_encoders:
            return pack_pipeline_output(X, y, is_tuple)

        X_out: Any = X

        # --- Encode feature columns ---
        if valid_cols and encoder:
            if engine.name == EngineName.POLARS:
                import polars as pl

                X_pl: Any = X
                try:
                    X_subset = X_pl.select(valid_cols)
                    X_subset = X_subset.select([pl.col(c).cast(pl.Utf8) for c in valid_cols])
                    X_np, _ = SklearnBridge.to_sklearn(X_subset)
                    encoded_array = encoder.transform(X_np)
                    new_cols_pl = [pl.Series(col, encoded_array[:, i]) for i, col in enumerate(valid_cols)]
                    X_out = X_pl.with_columns(new_cols_pl)
                except Exception as e:
                    logger.error(f"Ordinal Encoding failed: {e}")
            else:
                X_out = X.copy()
                try:
                    X_subset = X_out[valid_cols].astype(str)
                    X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
                    encoded_array = encoder.transform(X_input)
                    X_out[valid_cols] = encoded_array
                except Exception as e:
                    logger.error(f"Ordinal Encoding failed: {e}")

        # --- Encode target y ---
        y_out = y
        if y is not None and "__target__" in target_encoders:
            enc = target_encoders["__target__"]
            try:
                if engine.name == EngineName.POLARS:
                    import polars as pl

                    y_arr = y.to_numpy().astype(str).reshape(-1, 1)
                    encoded = enc.transform(y_arr).flatten()
                    y_name = y.name if hasattr(y, "name") else "target"
                    y_out = pl.Series(y_name, encoded.astype(np.float32))
                else:
                    y_arr = (
                        y.to_numpy().astype(str).reshape(-1, 1)
                        if hasattr(y, "to_numpy")
                        else np.array(y).astype(str).reshape(-1, 1)
                    )
                    encoded = enc.transform(y_arr).flatten()
                    y_out = pd.Series(
                        encoded,
                        index=y.index if hasattr(y, "index") else None,
                        name=y.name if hasattr(y, "name") else None,
                    )
            except Exception as e:
                logger.error(f"Ordinal Encoding target failed: {e}")

        return pack_pipeline_output(X_out, y_out, is_tuple)


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
)
class OrdinalEncoderCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        # OrdinalEncoder only accepts 'error' or 'use_encoded_value' — not 'ignore'.
        # Sanitise in case the config carries over a value valid for OHE ('ignore').
        _raw_hu = config.get("handle_unknown", "use_encoded_value")
        handle_unknown = _raw_hu if _raw_hu in ("error", "use_encoded_value") else "use_encoded_value"
        unknown_value = config.get("unknown_value", -1)

        # Detect target column name from config or y (set by Feature/Target Split)
        target_col: str | None = config.get("target_column")
        if target_col is None and y is not None:
            target_col = getattr(y, "name", None)

        target_encoders: Dict[str, Any] = {}

        categories_order_raw = config.get("categories_order")

        def _fit_enc_on_y(
            y_series: Any,
            categories: Union[str, List[List[str]]] = "auto",
        ) -> OrdinalEncoder:
            enc = OrdinalEncoder(
                categories=categories,
                handle_unknown=handle_unknown,
                unknown_value=unknown_value,
                dtype=np.float32,
            )
            y_arr = (
                y_series.to_numpy().astype(str).reshape(-1, 1)
                if hasattr(y_series, "to_numpy")
                else np.array(y_series).astype(str).reshape(-1, 1)
            )
            enc.fit(y_arr)
            return enc

        # No columns selected → encode y by default, mirrors LabelEncoder behaviour
        if user_picked_no_columns(config):
            if y is not None:
                cats_y = _parse_categories_order(categories_order_raw, 1)
                target_encoders["__target__"] = _fit_enc_on_y(y, cats_y)
            return {
                "type": "ordinal",
                "columns": [],
                "encoder_object": None,
                "encoders": target_encoders,
                "categories_count": [],
            }

        # Check raw config for target before resolve_columns filters out y-only columns.
        # After Feature/Target Split, target_col is in y, not X, so resolve_columns would
        # silently discard it. We detect the intent here first.
        cols_raw: List[str] = config.get("columns") or []
        encode_target = bool(
            target_col
            and target_col in cols_raw
            and y is not None
            and target_col not in X.columns
        )

        cols = resolve_columns(X, config, detect_categorical_columns)
        feature_cols = [c for c in cols if c in X.columns]

        if not feature_cols and not encode_target:
            return {}

        feature_encoder: OrdinalEncoder | None = None
        categories_count: List[int] = []

        if feature_cols:
            cats_feat = _parse_categories_order(categories_order_raw, len(feature_cols))
            feature_encoder = OrdinalEncoder(
                categories=cats_feat,
                handle_unknown=handle_unknown,
                unknown_value=unknown_value,
                dtype=np.float32,
            )
            if engine.name == EngineName.POLARS:
                import polars as pl

                X_pl: Any = X
                X_subset = X_pl.select(feature_cols)
                X_subset = X_subset.select([pl.col(c).cast(pl.Utf8) for c in feature_cols])
            else:
                X_subset = X[feature_cols].astype(str)
            X_np, _ = SklearnBridge.to_sklearn(X_subset)
            feature_encoder.fit(X_np)
            categories_count = [len(cats) for cats in feature_encoder.categories_]

        if encode_target and y is not None:
            # For target encoding, check if the user provided a 1-line order for the target.
            # We skip feature_cols lines and take the last one, or derive 1 from the textarea.
            n_feat = len(feature_cols)
            cats_tgt = _parse_categories_order(categories_order_raw, n_feat + 1)
            if isinstance(cats_tgt, list) and len(cats_tgt) == n_feat + 1:
                cats_for_y: Union[str, List[List[str]]] = [cats_tgt[-1]]
            else:
                cats_for_y = "auto"
            target_encoders["__target__"] = _fit_enc_on_y(y, cats_for_y)

        return {
            "type": "ordinal",
            "columns": feature_cols,
            "encoder_object": feature_encoder,
            "encoders": target_encoders,
            "categories_count": categories_count,
        }


# --- Label Encoder (Target) ---


class LabelEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        encoders = params.get("encoders", {})
        cols = params.get("columns")

        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X
            X_out = X_pl.clone()
            y_out = y.clone() if y is not None else None

            if cols:
                missing_code = params.get("missing_code", -1)
                exprs = []
                for col in cols:
                    if col in X_out.columns and col in encoders:
                        le = encoders[col]
                        mapping = {
                            str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))
                        }
                        exprs.append(
                            pl.col(col)
                            .cast(pl.Utf8)
                            .replace(mapping, default=missing_code)
                            .cast(pl.Int64)
                            .alias(col)
                        )
                if exprs:
                    X_out = X_out.with_columns(exprs)

            if y_out is not None and "__target__" in encoders:
                le = encoders["__target__"]
                mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
                missing_code = params.get("missing_code", -1)
                y_out = y_out.cast(pl.Utf8).replace(mapping, default=missing_code).cast(pl.Int64)

        else:
            X_out = X.copy()
            y_out = y.copy() if y is not None else None

            if cols:
                # Transform features
                for col in cols:
                    if col in X_out.columns and col in encoders:
                        le = encoders[col]
                        # Handle unseen labels? LabelEncoder crashes on unseen.
                        # We need a safe transform helper.

                        # Fast safe transform:
                        # Map known classes to integers, unknown to -1 or NaN
                        # But LabelEncoder doesn't support unknown.
                        # We can use map.
                        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                        missing_code = params.get("missing_code", -1)
                        X_out[col] = X_out[col].astype(str).map(mapping).fillna(missing_code)

            # Transform target (always check if encoder exists)
            if y_out is not None and "__target__" in encoders:
                le = encoders["__target__"]
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                missing_code = params.get("missing_code", -1)
                y_out = y_out.astype(str).map(mapping).fillna(missing_code)

        return pack_pipeline_output(X_out, y_out, is_tuple)


@NodeRegistry.register("LabelEncoder", LabelEncoderApplier)
@node_meta(
    id="LabelEncoder",
    name="Label Encoder",
    category="Preprocessing",
    description="Encode target labels with value between 0 and n_classes-1.",
    params={"columns": [], "missing_code": -1},
)
class LabelEncoderCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        # Attempt to extract y from X if not provided
        target_col = config.get("target_column")
        if y is None and target_col:
            # Check if target_col is in X
            if engine.name == EngineName.POLARS:
                X_pl: Any = X
                cols = X_pl.columns
                if target_col in cols:
                    y = X_pl.get_column(target_col)
            else:
                cols = X.columns
                if target_col in cols:
                    y = X[target_col]

        # LabelEncoder is usually for the target variable y.
        # But sometimes used for features too.
        # Config: {'columns': [...]} or empty for target

        cols = config.get("columns")

        encoders = {}
        classes_count = {}

        if cols:
            # Encode features
            if engine.name == EngineName.POLARS:
                X_pl_data: Any = X
                valid_cols = [c for c in cols if c in X_pl_data.columns]
                import polars as pl

                for col in valid_cols:
                    le = LabelEncoder()
                    # Convert to numpy array of strings
                    col_data = X_pl_data.select(pl.col(col).cast(pl.Utf8)).to_series().to_numpy()
                    le.fit(col_data)
                    encoders[col] = le
                    classes_count[col] = len(le.classes_)
            else:
                valid_cols = [c for c in cols if c in X.columns]
                # Pandas Path
                for col in valid_cols:
                    le = LabelEncoder()
                    le.fit(X[col].astype(str))
                    encoders[col] = le
                    classes_count[col] = len(le.classes_)

            # Also check if target is in cols (if y has a name)
            if y is not None:
                y_name = getattr(y, "name", None)
                if y_name and y_name in cols:
                    le = LabelEncoder()
                    # Handle y conversion
                    if hasattr(y, "to_numpy"):  # Polars Series or Pandas Series
                        y_data = y.to_numpy().astype(str)
                    else:
                        y_data = np.array(y).astype(str)

                    le.fit(y_data)
                    encoders["__target__"] = le
                    classes_count["__target__"] = len(le.classes_)

        else:
            # Encode target y (default if no columns specified)
            if y is not None:
                le = LabelEncoder()
                # Handle y conversion
                if hasattr(y, "to_numpy"):
                    y_data = y.to_numpy().astype(str)
                else:
                    y_data = np.array(y).astype(str)

                le.fit(y_data)
                encoders["__target__"] = le
                classes_count["__target__"] = len(le.classes_)

        return {
            "type": "label_encoder",
            "encoders": encoders,
            "columns": cols,
            "classes_count": classes_count,
            "missing_code": config.get("missing_code", -1),
        }


# --- Target Encoder (Mean Encoding) ---


class TargetEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        encoder = params.get("encoder_object")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X

            try:
                X_subset = X_pl.select(valid_cols)
                X_np, _ = SklearnBridge.to_sklearn(X_subset)
                encoded_array = encoder.transform(X_np)

                new_cols = [pl.Series(col, encoded_array[:, i]) for i, col in enumerate(valid_cols)]
                X_out = X_pl.with_columns(new_cols)
                return pack_pipeline_output(X_out, y, is_tuple)
            except Exception as e:
                logger.error(f"Target Encoding failed: {e}")
                return pack_pipeline_output(X, y, is_tuple)

        # Pandas Path
        X_out = X.copy()

        try:
            X_subset = X_out[valid_cols]

            # Fix for "X has feature names..." warning
            if hasattr(X_subset, "values"):
                X_input = X_subset.values
            else:
                X_input = X_subset

            encoded_array = encoder.transform(X_input)
            X_out[valid_cols] = encoded_array
        except Exception as e:
            logger.error(f"Target Encoding failed: {e}")

        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("TargetEncoder", TargetEncoderApplier)
@node_meta(
    id="TargetEncoder",
    name="Target Encoder",
    category="Preprocessing",
    description="Encode categorical features using target statistics.",
    params={"smooth": "auto", "target_type": "auto", "columns": []},
)
class TargetEncoderCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        # Attempt to extract y from X if not provided
        target_col = config.get("target_column")
        if y is None and target_col:
            if engine.name == EngineName.POLARS:
                X_pl_y: Any = X
                if target_col in X_pl_y.columns:
                    y = X_pl_y.get_column(target_col)
            else:
                if target_col in X.columns:
                    y = X[target_col]

        if y is None:
            logger.warning("TargetEncoder requires a target variable (y). Skipping.")
            return {}

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "TargetEncoder", y)
        if not cols:
            return {}

        # Config: {'smooth': 'auto', 'target_type': 'auto'}
        smooth = config.get("smooth", "auto")
        target_type = config.get("target_type", "auto")

        encoder = TargetEncoder(smooth=smooth, target_type=target_type)

        # Use Bridge for fitting
        if engine.name == EngineName.POLARS:
            X_pl: Any = X
            X_subset = X_pl.select(cols)
        else:
            X_subset = X[cols]

        X_np, _ = SklearnBridge.to_sklearn(X_subset)

        # Handle y
        y_np = y
        if hasattr(y, "to_numpy"):
            y_np = y.to_numpy()
        elif hasattr(y, "to_pandas"):  # Polars Series
            y_np = y.to_pandas().to_numpy()

        encoder.fit(X_np, y_np)

        return {"type": "target_encoder", "columns": cols, "encoder_object": encoder}


# --- Hash Encoder ---


class HashEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        n_features = params.get("n_features", 10)

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X

            exprs = []
            for col in valid_cols:
                # Polars hash() returns u64. We take modulo n_features.
                # Note: Polars hash might differ from Python hash.
                # For consistency across engines, we might need a custom hash or accept divergence.
                # Here we use Polars native hash for speed.
                exprs.append((pl.col(col).cast(pl.Utf8).hash() % n_features).alias(col))

            X_out = X_pl.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_out = X.copy()

        # hasher = FeatureHasher(n_features=n_features, input_type='string')

        # Apply hashing to each column separately.
        # We use a simple deterministic hash() % n_features approach.

        for col in valid_cols:
            X_out[col] = X_out[col].astype(str).apply(lambda x: hash(x) % n_features)

        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("HashEncoder", HashEncoderApplier)
@node_meta(
    id="HashEncoder",
    name="Hash Encoder",
    category="Preprocessing",
    description="Encode categorical features using hashing.",
    params={"n_features": 8, "columns": []},
)
class HashEncoderCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, _ = unpack_pipeline_input(df)

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "HashEncoder", y)
        if not cols:
            return {}

        # Config: {'n_features': 10}
        n_features = config.get("n_features", 10)

        # FeatureHasher is stateless, no fit needed really, but we store config
        return {"type": "hash_encoder", "columns": cols, "n_features": n_features}


# --- Dummy Encoder (Pandas get_dummies) ---


class DummyEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        categories = params.get("categories", {})
        drop_first = params.get("drop_first", False)

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X

            X_out = X_pl

            for col in valid_cols:
                known_cats = categories.get(col, [])

                # Create dummies manually for Polars
                # For each category, create a boolean column (cast to int)

                cats_to_encode = known_cats
                if drop_first and len(cats_to_encode) > 1:
                    cats_to_encode = cats_to_encode[1:]

                dummy_exprs = []
                for cat in cats_to_encode:
                    # Column name: col_cat
                    dummy_name = f"{col}_{cat}"
                    dummy_exprs.append(
                        (pl.col(col).cast(pl.Utf8) == str(cat)).cast(pl.Int8).alias(dummy_name)
                    )

                X_out = X_out.with_columns(dummy_exprs)

            # Drop original columns
            X_out = X_out.drop(valid_cols)

            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_out = X.copy()

        for col in valid_cols:
            # Convert to categorical with known categories
            known_cats = categories.get(col, [])
            X_out[col] = pd.Categorical(X_out[col].astype(str), categories=known_cats)

        # Get dummies
        dummies = pd.get_dummies(X_out[valid_cols], drop_first=drop_first, dtype=int)

        # Drop original
        X_out = X_out.drop(columns=valid_cols)

        # Concatenate dummies
        X_out = pd.concat([X_out, dummies], axis=1)

        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("DummyEncoder", DummyEncoderApplier)
@node_meta(
    id="DummyEncoder",
    name="Dummy Encoder",
    category="Preprocessing",
    description="Convert categorical variables into dummy/indicator variables (pandas.get_dummies).",
    params={"columns": [], "drop_first": False},
)
class DummyEncoderCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "DummyEncoder", y)

        # We need to know all possible categories to align columns during transform
        categories = {}

        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X
            for col in cols:
                # Get unique values, sort them, convert to list
                cats = X_pl.select(pl.col(col).cast(pl.Utf8).unique().sort()).to_series().to_list()
                categories[col] = [str(c) for c in cats if c is not None]
        else:
            # Pandas
            for col in cols:
                categories[col] = sorted(X[col].dropna().unique().astype(str).tolist())

        return {
            "type": "dummy_encoder",
            "columns": cols,
            "categories": categories,
            "drop_first": config.get("drop_first", False),
        }
