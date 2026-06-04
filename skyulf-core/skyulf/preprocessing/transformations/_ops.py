"""Per-method expression builders shared by Simple + General transformations.

Each helper stays at low CCN; the public dispatch dicts (``_POLARS_OPS`` /
``_PANDAS_OPS``) map a method name to its builder.
"""

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd


def _polars_log(item: Dict[str, Any]) -> Any:
    import polars as pl

    col = item["column"]
    return pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).log1p()


def _polars_sqrt(item: Dict[str, Any]) -> Any:
    import polars as pl

    col = item["column"]
    return pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).sqrt()


def _polars_cbrt(item: Dict[str, Any]) -> Any:
    import polars as pl

    return pl.col(item["column"]).cbrt()


def _polars_reciprocal(item: Dict[str, Any]) -> Any:
    import polars as pl

    col = item["column"]
    return 1.0 / pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col))


def _polars_square(item: Dict[str, Any]) -> Any:
    import polars as pl

    return pl.col(item["column"]).pow(2)


def _polars_exp(item: Dict[str, Any]) -> Any:
    import polars as pl

    threshold = item.get("clip_threshold", 700)
    return pl.col(item["column"]).clip(upper_bound=threshold).exp()


_POLARS_OPS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "log": _polars_log,
    "sqrt": _polars_sqrt,
    "square_root": _polars_sqrt,
    "cube_root": _polars_cbrt,
    "reciprocal": _polars_reciprocal,
    "square": _polars_square,
    "exp": _polars_exp,
    "exponential": _polars_exp,
}


def _pandas_log(series: pd.Series, _item: Dict[str, Any]) -> Any:
    if (series < 0).any():
        series = series.where(series >= 0, np.nan)
    return np.log1p(series)


def _pandas_sqrt(series: pd.Series, _item: Dict[str, Any]) -> Any:
    if (series < 0).any():
        series = series.where(series >= 0, np.nan)
    return np.sqrt(series)


def _pandas_cbrt(series: pd.Series, _item: Dict[str, Any]) -> Any:
    return np.cbrt(series)


def _pandas_reciprocal(series: pd.Series, _item: Dict[str, Any]) -> Any:
    return 1.0 / series.replace(0, np.nan)


def _pandas_square(series: pd.Series, _item: Dict[str, Any]) -> Any:
    return np.square(series)


def _pandas_exp(series: pd.Series, item: Dict[str, Any]) -> Any:
    threshold = item.get("clip_threshold", 700)
    return np.exp(series.clip(upper=threshold))


_PANDAS_OPS: Dict[str, Callable[[pd.Series, Dict[str, Any]], Any]] = {
    "log": _pandas_log,
    "sqrt": _pandas_sqrt,
    "square_root": _pandas_sqrt,
    "cube_root": _pandas_cbrt,
    "reciprocal": _pandas_reciprocal,
    "square": _pandas_square,
    "exp": _pandas_exp,
    "exponential": _pandas_exp,
}
