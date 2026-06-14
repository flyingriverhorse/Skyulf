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

from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from ...utils import pack_pipeline_output, unpack_pipeline_input

# Signature: (X_pandas, y, params) -> (X_out_pandas, y_out)
TextApplyFn = Callable[[pd.DataFrame, Any, Dict[str, Any]], Tuple[pd.DataFrame, Any]]


def apply_text_pandas_only(
    df: Any,
    params: Dict[str, Any],
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


def _join_text_columns(X: pd.DataFrame, cols: list) -> pd.Series:
    """Concatenate one or more text columns into a single string series."""
    series = X[cols[0]].fillna("").astype(str)
    for col in cols[1:]:
        series = series + " " + X[col].fillna("").astype(str)
    return series


def _warn_large_output(output_cols: int, threshold: int = 10_000) -> Optional[str]:
    """Return a warning message if the output column count exceeds *threshold*."""
    if output_cols > threshold:
        return (
            f"Vectorizer produces {output_cols:,} output columns. "
            "Consider reducing max_features to lower memory usage."
        )
    return None
