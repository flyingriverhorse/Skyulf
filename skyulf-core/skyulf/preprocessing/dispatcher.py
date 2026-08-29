"""Dual-engine dispatch for preprocessing nodes.

This module owns the *control flow* that lets a single node run on either the
Polars or the Pandas engine: ``apply_dual_engine`` (and its fit counterpart)
unpacks the pipeline input, selects the engine-specific implementation, and
repacks the output. It is the single place that branches on the engine.

Boundary with ``_helpers.py``: leaf utilities used *inside* the engine branches
(column resolution, ``is_polars`` / ``to_pandas``, safe scaling) live in
``_helpers.py``. The dispatcher never implements column-level logic, and the
helpers never dispatch a whole node.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ..engines import EngineName, SkyulfDataFrame, SkyulfPolarsWrapper, get_engine
from ..utils import pack_pipeline_output, unpack_pipeline_input

logger = logging.getLogger(__name__)


def _unwrap_polars_wrapper(X: Any) -> tuple[Any, bool]:
    """Return ``(frame, was_wrapped)`` for the Polars dispatch branch.

    ``SkyulfPolarsWrapper`` is a documented public input type, but node
    implementations reach for native polars APIs (``pl.concat``,
    ``fill_null``, ...) that crash on the wrapper (F-09). Hand them the raw
    ``pl.DataFrame`` instead; callers re-wrap the output so the result keeps
    the caller's engine.
    """
    if isinstance(X, SkyulfPolarsWrapper):
        return X._df, True
    return X, False


def _rewrap_polars_output(X_out: Any, was_wrapped: bool) -> Any:
    if was_wrapped and type(X_out).__module__.startswith("polars"):
        return SkyulfPolarsWrapper(X_out)
    return X_out


def _check_xy_engine_parity(X: Any, y: Any) -> None:
    """Reject ``(X, y)`` pairs whose frames come from different engines (F-27).

    A pandas X cannot be indexed by a polars y (or vice versa); without this
    guard the mismatch surfaces deep inside an engine-specific implementation
    as a confusing ``AttributeError``. Engine-neutral y values (lists, numpy
    arrays) are always accepted.
    """
    if y is None:
        return
    x_is_polars = isinstance(X, SkyulfPolarsWrapper) or type(X).__module__.startswith("polars")
    y_is_polars = type(y).__module__.startswith("polars")
    y_is_pandas = isinstance(y, (pd.DataFrame, pd.Series))
    if (x_is_polars and y_is_pandas) or (y_is_polars and not x_is_polars):
        raise TypeError(
            "Mixed engines in (X, y): X is "
            f"{'polars' if x_is_polars else 'pandas'} but y is "
            f"{'polars' if y_is_polars else 'pandas'}. "
            "Both must use the same engine."
        )


def _callable_name(func: Callable[..., Any]) -> str:
    """Best-effort display name for a dispatch-target callable.

    Not every callable used here is a plain function (some are
    ``functools.partial``/lambdas/bound methods from tests and call sites),
    so ``__qualname__`` isn't guaranteed to exist -- fall back to
    ``__name__`` and finally ``repr`` for logging purposes only.
    """
    return getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))


def _log_dispatch_failure(
    exc: Exception, engine: str, operation: str, func: Callable[..., Any]
) -> None:
    """Log expected input errors quietly and unexpected dispatcher failures with a traceback."""
    message = "%s engine %s failed in %s"
    if isinstance(exc, ValueError):
        logger.debug(message + ": %s", engine, operation, _callable_name(func), exc)
    else:
        logger.exception(message, engine, operation, _callable_name(func), exc_info=exc)


# Type definitions for the processing functions
# They receive (X, y, params)
# Apply returns (X_transformed, y_transformed)
ApplyFunction = Callable[[Any, Any | None, dict[str, Any]], tuple[Any, Any | None]]
# Fit returns a mapping (TypedDicts are accepted via Mapping invariance).
FitFunction = Callable[[Any, Any | None, dict[str, Any]], Mapping[str, Any]]
TrainTransformFunction = Callable[
    [Any, Any | None, dict[str, Any]],
    tuple[Mapping[str, Any], Any, Any | None],
]


def apply_dual_engine(
    df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
    params: dict[str, Any],
    polars_func: ApplyFunction,
    pandas_func: ApplyFunction,
) -> Any:
    """
    Dispatcher to handle boilerplate for dual-engine Appliers.

    Args:
        df: Input data (DataFrame or Tuple).
        params: Configuration parameters.
        polars_func: Function to execute if engine is Polars.
                     Signature: (X, y, params) -> (X_out, y_out)
        pandas_func: Function to execute if engine is Pandas.
                     Signature: (X, y, params) -> (X_out, y_out)
                     Note: Input X is guaranteed to be a Pandas DataFrame/Series here.

    Returns:
        Packed output matching the input format.
    """
    X, y, is_tuple = unpack_pipeline_input(df)
    _check_xy_engine_parity(X, y)
    engine = get_engine(X)

    if engine.name == EngineName.POLARS:
        # Polars path
        # Unwrap SkyulfPolarsWrapper so polars_func sees a raw pl.DataFrame
        # (native pl APIs crash on the wrapper, F-09); re-wrap afterwards
        # so the output keeps the caller's engine.
        X_pl, was_wrapped = _unwrap_polars_wrapper(X)
        try:
            X_out, y_out = polars_func(X_pl, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Polars", "apply", polars_func)
            raise
        X_out = _rewrap_polars_output(X_out, was_wrapped)
    else:
        # Pandas path
        # Ensure X is pandas
        X_pd = X.to_pandas() if hasattr(X, "to_pandas") else X

        try:
            X_out, y_out = pandas_func(X_pd, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Pandas", "apply", pandas_func)
            raise

    return pack_pipeline_output(X_out, y_out, is_tuple)


def fit_dual_engine(
    df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
    params: dict[str, Any],
    polars_func: FitFunction,
    pandas_func: FitFunction,
) -> dict[str, Any]:
    """
    Dispatcher to handle boilerplate for dual-engine Calculators.

    Args:
        df: Inputs.
        params: Config.
        polars_func: (X, y, params) -> Dict[Result]
        pandas_func: (X, y, params) -> Dict[Result]

    Returns:
        Dictionary of fitted parameters.
    """
    X, y, _ = unpack_pipeline_input(df)
    _check_xy_engine_parity(X, y)
    engine = get_engine(X)

    if engine.name == EngineName.POLARS:
        X_pl, _ = _unwrap_polars_wrapper(X)
        try:
            return dict(polars_func(X_pl, y, params))
        except Exception as exc:
            _log_dispatch_failure(exc, "Polars", "fit", polars_func)
            raise
    else:
        X_pd = X.to_pandas() if hasattr(X, "to_pandas") else X
        try:
            return dict(pandas_func(X_pd, y, params))
        except Exception as exc:
            _log_dispatch_failure(exc, "Pandas", "fit", pandas_func)
            raise


def fit_transform_train_dual_engine(
    df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
    params: dict[str, Any],
    polars_func: TrainTransformFunction,
    pandas_func: TrainTransformFunction,
) -> tuple[dict[str, Any], Any]:
    """Dispatch an optional fit+train-transform hook across supported engines."""
    X, y, is_tuple = unpack_pipeline_input(df)
    _check_xy_engine_parity(X, y)
    engine = get_engine(X)

    if engine.name == EngineName.POLARS:
        X_pl, was_wrapped = _unwrap_polars_wrapper(X)
        try:
            artifact, X_out, y_out = polars_func(X_pl, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Polars", "fit_transform_train", polars_func)
            raise
        X_out = _rewrap_polars_output(X_out, was_wrapped)
    else:
        X_pd = X.to_pandas() if hasattr(X, "to_pandas") else X
        try:
            artifact, X_out, y_out = pandas_func(X_pd, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Pandas", "fit_transform_train", pandas_func)
            raise

    return dict(artifact), pack_pipeline_output(X_out, y_out, is_tuple)
