"""Shared helpers for text-vectorization nodes.

Text/vectorization nodes always round-trip through pandas because sklearn
vectorizers consume ``Iterable[str]``, not Polars expressions.  The
``apply_text_pandas_only`` dispatcher handles the polars→pandas→polars
conversion transparently so individual Applier implementations never need to
deal with engine detection.

Boundary with ``dispatcher.py``:
    * ``dispatcher.py`` owns the generic dual-engine dispatch pattern.
    * This module owns the *text-specific* single-engine (pandas-only) pattern.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd

from ...utils import pack_pipeline_output, unpack_pipeline_input

# Signature: (X_pandas, y, params) -> (X_out_pandas, y_out)
TextApplyFn = Callable[[pd.DataFrame, Any, dict[str, Any]], tuple[pd.DataFrame, Any]]


def apply_text_pandas_only(
    df: Any,
    params: dict[str, Any],
    fn: TextApplyFn,
) -> Any:
    """Dispatcher for text/vectorization nodes that always use the pandas path.

    Converts Polars input to pandas before calling ``fn``, then converts the
    result back to Polars so the caller receives the same engine as the input.

    Args:
        df: Pipeline input (DataFrame or (X, y) tuple — any engine).
        params: Fitted parameters forwarded to ``fn``.
        fn: Engine-agnostic apply function with signature
            ``(X: pd.DataFrame, y, params) -> (X_out: pd.DataFrame, y_out)``.

    Returns:
        Repacked pipeline output in the original input format/engine.
    """
    X, y, is_tuple = unpack_pipeline_input(df)

    was_polars = hasattr(X, "to_pandas") and type(X).__module__.startswith("polars")

    X_pd: pd.DataFrame = X.to_pandas() if was_polars else X  # type: ignore[assignment]

    X_out_pd, y_out = fn(X_pd, y, params)

    X_out: Any = X_out_pd
    if was_polars and isinstance(X_out_pd, pd.DataFrame):
        import polars as pl

        X_out = pl.from_pandas(X_out_pd)

    return pack_pipeline_output(X_out, y_out, is_tuple)


def resolve_fit_text_columns(
    X: Any, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]] | None:
    """Resolve the pandas frame and valid text columns for a vectorizer ``fit``.

    Converts *X* to pandas if needed and filters ``config["columns"]`` down to
    columns actually present. Returns ``None`` when there are no configured or
    matching columns, signalling the caller should return an empty artifact.
    """
    cols: list[str] = config.get("columns", [])
    if not cols:
        return None

    if hasattr(X, "to_pandas"):
        X = X.to_pandas()

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols:
        return None

    return X, valid_cols


def _join_text_columns(X: pd.DataFrame, cols: list) -> pd.Series:
    """Concatenate one or more text columns into a single string series."""
    series = X[cols[0]].fillna("").astype(str)
    for col in cols[1:]:
        series = series + " " + X[col].fillna("").astype(str)
    return series


def _warn_large_output(output_cols: int, threshold: int = 10_000) -> str | None:
    """Return a warning message if the output column count exceeds *threshold*."""
    if output_cols > threshold:
        return (
            f"Vectorizer produces {output_cols:,} output columns. "
            "Consider reducing max_features to lower memory usage."
        )
    return None


def _vectorizer_transform_to_frame(
    X: pd.DataFrame, vectorizer: Any, valid_cols: list[str], output_columns: list[str]
) -> pd.DataFrame:
    """Transform text columns with a fitted vectorizer into a dense output frame."""
    text = _join_text_columns(X, valid_cols)
    encoded = vectorizer.transform(text)
    dense = encoded.toarray() if hasattr(encoded, "toarray") else encoded
    return pd.DataFrame(
        dense,
        columns=output_columns,  # ty: ignore[invalid-argument-type]
        index=X.index,
    )


def _drop_and_concat(
    X: pd.DataFrame, encoded_df: pd.DataFrame, valid_cols: list[str], drop_original: bool
) -> pd.DataFrame:
    """Optionally drop source columns from *X*, then concat the encoded frame."""
    X_out = X.copy()
    if drop_original:
        X_out = X_out.drop(columns=valid_cols)
    return pd.concat([X_out, encoded_df], axis=1)


def _sklearn_vectorizer_apply_pandas(
    X: pd.DataFrame, y: Any, params: dict[str, Any]
) -> tuple[pd.DataFrame, Any]:
    """Shared apply logic for fitted-vocabulary sklearn vectorizers.

    Used by the Count/TF-IDF/Hashing vectorizer appliers, which all transform
    text columns with a fitted ``vectorizer_object`` and concatenate the dense
    result onto the input frame identically.
    """
    cols: list[str] = params.get("columns", [])
    vectorizer: Any = params.get("vectorizer_object")
    output_columns: list[str] = params.get("output_columns", [])
    drop_original: bool = params.get("drop_original", False)

    if not cols or vectorizer is None or not output_columns:
        return X, y

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols:
        return X, y

    encoded_df = _vectorizer_transform_to_frame(X, vectorizer, valid_cols, output_columns)
    return _drop_and_concat(X, encoded_df, valid_cols, drop_original), y
