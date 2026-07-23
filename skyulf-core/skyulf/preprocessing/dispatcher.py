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

from ..engines import EngineName, SkyulfDataFrame, get_engine
from ..utils import pack_pipeline_output, unpack_pipeline_input

logger = logging.getLogger(__name__)


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
        logger.exception(message, engine, operation, _callable_name(func))


# Type definitions for the processing functions
# They receive (X, y, params)
# Apply returns (X_transformed, y_transformed)
ApplyFunction = Callable[[Any, Any | None, dict[str, Any]], tuple[Any, Any | None]]
# Fit returns a mapping (TypedDicts are accepted via Mapping invariance).
FitFunction = Callable[[Any, Any | None, dict[str, Any]], Mapping[str, Any]]


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
    engine = get_engine(X)

    if engine.name == EngineName.POLARS:
        # Polars path
        # We pass X directly. The func should handle typing (X_pl: Any = X)
        try:
            X_out, y_out = polars_func(X, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Polars", "apply", polars_func)
            raise
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
    engine = get_engine(X)

    if engine.name == EngineName.POLARS:
        try:
            return dict(polars_func(X, y, params))
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
