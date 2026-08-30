"""Shared helpers for text-vectorization nodes.

sklearn vectorizers consume ``Iterable[str]`` and emit sparse matrices, so the
text payload itself must cross into numpy/pandas. The frame around it does not:
``apply_text_dual_engine`` tries each node's native-Polars apply first (text
columns are pulled out as ``list[str]``, the encoded output is attached with
``hstack``) and only falls back to a full pandas round-trip when the native
path cannot preserve semantics (e.g. non-String text columns).

Boundary with ``dispatcher.py``:
    * ``dispatcher.py`` owns the generic dual-engine dispatch pattern.
    * This module owns the *text-specific* dispatch pattern.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ...engines.polars_engine import SkyulfPolarsWrapper
from ...utils import pack_pipeline_output, unpack_pipeline_input
from .._helpers import resolve_valid_columns

# Signature: (X_pandas, y, params) -> (X_out_pandas, y_out)
TextApplyFn = Callable[[pd.DataFrame, Any, dict[str, Any]], tuple[pd.DataFrame, Any]]
# Signature: (X_polars, params) -> X_out_polars, or None to fall back to pandas
PolarsTextApplyFn = Callable[[Any, dict[str, Any]], Any]


def apply_text_dual_engine(
    df: Any,
    params: dict[str, Any],
    pandas_fn: TextApplyFn,
    polars_fn: PolarsTextApplyFn | None = None,
) -> Any:
    """Dispatcher for text/vectorization nodes.

    When *polars_fn* is given and the input is Polars, the native path runs
    first; returning ``None`` from it falls back to the pandas path (used when
    the native path cannot reproduce pandas semantics exactly). Otherwise the
    input is converted to pandas, ``pandas_fn`` runs, and a Polars input gets
    its result converted back so the caller receives the input's engine.

    Args:
        df: Pipeline input (DataFrame or (X, y) tuple — any engine).
        params: Fitted parameters forwarded to the apply functions.
        pandas_fn: Apply function with signature
            ``(X: pd.DataFrame, y, params) -> (X_out: pd.DataFrame, y_out)``.
        polars_fn: Optional native apply with signature
            ``(X: pl.DataFrame, params) -> pl.DataFrame | None``.

    Returns:
        Repacked pipeline output in the original input format/engine.
    """
    X, y, is_tuple = unpack_pipeline_input(df)

    # Polars input arrives either as a raw pl.DataFrame or as a
    # SkyulfPolarsWrapper (documented public input type). The wrapper lives
    # under `skyulf.*` and exposes the real frame via `to_native()`, so a
    # plain module check misses it and would hand the wrapper itself to `fn`
    # (F-09).
    was_wrapped = (
        type(X).__module__.startswith("skyulf")
        and hasattr(X, "to_native")
        and type(X.to_native()).__module__.startswith("polars")
    )
    was_polars = (
        hasattr(X, "to_pandas") and type(X).__module__.startswith("polars")
    ) or was_wrapped

    if was_polars and polars_fn is not None:
        X_pl = X.to_native() if was_wrapped else X
        X_out_pl = polars_fn(X_pl, params)
        if X_out_pl is not None:
            if was_wrapped:
                X_out_pl = SkyulfPolarsWrapper(X_out_pl)
            return pack_pipeline_output(X_out_pl, y, is_tuple)

    X_pd: pd.DataFrame = X.to_pandas() if was_polars else X  # type: ignore[assignment]

    X_out_pd, y_out = pandas_fn(X_pd, y, params)

    X_out: Any = X_out_pd
    if was_polars and isinstance(X_out_pd, pd.DataFrame):
        import polars as pl

        X_out = pl.from_pandas(X_out_pd)
        if was_wrapped:
            X_out = SkyulfPolarsWrapper(X_out)

    return pack_pipeline_output(X_out, y_out, is_tuple)


def resolve_fit_text_valid_columns(X: Any, config: dict[str, Any]) -> list[str] | None:
    """Resolve valid text columns for a vectorizer ``fit`` without converting *X*.

    Filters ``config["columns"]`` down to columns actually present on *X*
    (any engine — Polars/Pandas both expose ``.columns``). Returns ``None``
    when there are no configured or matching columns, signalling the caller
    should return an empty artifact.

    Use this instead of :func:`resolve_fit_text_columns` when the caller only
    needs the resolved column names (no text data), e.g. hashing vectorizers
    or tokenizers that fit stateless/vocabulary-free artifacts.
    """
    cols: list[str] = config.get("columns", [])
    if not cols:
        return None

    valid_cols = resolve_valid_columns(X, cols)
    if not valid_cols:
        return None

    return valid_cols


def resolve_fit_text_columns(
    X: Any, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]] | None:
    """Resolve the narrowed pandas frame and valid text columns for a fit.

    Filters ``config["columns"]`` down to columns actually present, narrows the
    input to just those columns, then converts to pandas if needed. Returns
    ``None`` when there are no configured or matching columns, signalling the
    caller should return an empty artifact.

    Use this when the caller needs the actual text data (e.g. Count/TF-IDF
    vectorizers fitting a vocabulary). If only the resolved column names are
    needed, use :func:`resolve_fit_text_valid_columns` instead to avoid a
    wasted full-frame Pandas conversion.
    """
    cols: list[str] = config.get("columns", [])
    if not cols:
        return None

    valid_cols = resolve_valid_columns(X, cols)
    if not valid_cols:
        return None

    if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
        X = X.select(valid_cols).to_pandas()
    else:
        X = X[valid_cols]

    return X, valid_cols


def _join_text_columns(X: pd.DataFrame, cols: list) -> pd.Series:
    """Concatenate one or more text columns into a single string series."""
    series = X[cols[0]].fillna("").astype(str)
    for col in cols[1:]:
        series = series + " " + X[col].fillna("").astype(str)
    return series


def _join_text_columns_polars(X: Any, cols: list[str]) -> Any:
    """Native join of text columns; returns ``None`` if any column isn't String.

    The ``None`` return signals the caller to fall back to the pandas path,
    whose ``astype(str)`` stringifies non-text dtypes in ways Polars' ``cast``
    does not reproduce bit-for-bit (e.g. boolean casing).
    """
    import polars as pl

    series = None
    for col in cols:
        col_series = X.get_column(col)
        if col_series.dtype != pl.String:
            return None
        col_series = col_series.fill_null("")
        series = col_series if series is None else series + " " + col_series
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


def _drop_and_concat_polars(
    X: Any, encoded_frame: Any, valid_cols: list[str], drop_original: bool
) -> Any:
    """Native-Polars equivalent of :func:`_drop_and_concat`."""
    X_out = X.drop(valid_cols) if drop_original else X
    return X_out.hstack(encoded_frame)


def _sklearn_vectorizer_apply_polars(X: Any, params: dict[str, Any]) -> Any:
    """Native-Polars apply for Count/TF-IDF/Hashing vectorizers.

    Pulls the joined text out of Polars as ``list[str]``, transforms it with
    the fitted vectorizer, and attaches the dense matrix with ``hstack`` —
    the frame itself is never converted. Returns ``None`` to request a
    fallback to the pandas path when a text column is not String dtype.
    """
    import polars as pl

    cols: list[str] = params.get("columns", [])
    vectorizer: Any = params.get("vectorizer_object")
    output_columns: list[str] = params.get("output_columns", [])
    drop_original: bool = params.get("drop_original", False)

    if not cols or vectorizer is None or not output_columns:
        return X

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols:
        return X

    text = _join_text_columns_polars(X, valid_cols)
    if text is None:
        return None

    encoded = vectorizer.transform(text.to_list())
    dense = encoded.toarray() if hasattr(encoded, "toarray") else encoded
    encoded_frame = pl.from_numpy(np.asarray(dense), schema=output_columns)
    return _drop_and_concat_polars(X, encoded_frame, valid_cols, drop_original)
