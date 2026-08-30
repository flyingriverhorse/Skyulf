"""Dual-engine dispatch for preprocessing nodes.

This module owns the *control flow* that lets a single node run on either the
Polars or the Pandas engine: ``apply_dual_engine`` (and its fit counterparts)
unpacks the pipeline input, selects the engine-specific implementation from a
mapping keyed by engine name, and repacks the output. It is the single place
that branches on the engine.

An engine with no registered implementation fails loudly with
``NotImplementedError`` instead of being silently collected to pandas (F-09):
pulling a distributed frame to the driver is a decision callers must make
explicitly, never a dispatch default.

Boundary with ``_helpers.py``: leaf utilities used *inside* the engine branches
(column resolution, ``is_polars`` / ``to_pandas``, safe scaling) live in
``_helpers.py``. The dispatcher never implements column-level logic, and the
helpers never dispatch a whole node.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

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
        return X.to_native(), True
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

_ImplFunc = TypeVar("_ImplFunc", bound=Callable[..., Any])


def _resolve_impl(
    implementations: Mapping[str, _ImplFunc], engine: EngineName, operation: str
) -> _ImplFunc:
    """Select the implementation for ``engine`` or fail loudly (F-09).

    Raised *before* any frame conversion: an unmapped engine must never be
    silently collected to pandas.
    """
    func = implementations.get(engine)
    if func is None:
        raise NotImplementedError(
            f"No '{engine}' implementation registered for '{operation}' "
            f"(available: {', '.join(sorted(implementations))})"
        )
    return func


def apply_dual_engine(
    df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
    params: dict[str, Any],
    implementations: Mapping[str, ApplyFunction],
) -> Any:
    """
    Dispatcher to handle boilerplate for dual-engine Appliers.

    Args:
        df: Input data (DataFrame or Tuple).
        params: Configuration parameters.
        implementations: Engine-specific implementations keyed by engine name,
                     e.g. ``{"polars": fn_pl, "pandas": fn_pd}``.
                     Signature: (X, y, params) -> (X_out, y_out)
                     Note: On the pandas path, input X is guaranteed to be a
                     Pandas DataFrame/Series here.

    Returns:
        Packed output matching the input format.

    Raises:
        NotImplementedError: If no implementation is registered for the input's
            engine, or the engine has no input-preparation path yet.
    """
    X, y, is_tuple = unpack_pipeline_input(df)
    _check_xy_engine_parity(X, y)
    engine = get_engine(X)
    func = _resolve_impl(implementations, engine.name, "apply")

    if engine.name == EngineName.POLARS:
        # Unwrap SkyulfPolarsWrapper so the implementation sees a raw
        # pl.DataFrame (native pl APIs crash on the wrapper, F-09); re-wrap
        # afterwards so the output keeps the caller's engine.
        X_prep, was_wrapped = _unwrap_polars_wrapper(X)
        try:
            X_out, y_out = func(X_prep, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Polars", "apply", func)
            raise
        X_out = _rewrap_polars_output(X_out, was_wrapped)
    elif engine.name == EngineName.PANDAS:
        # Ensure X is pandas
        X_prep = X.to_pandas() if hasattr(X, "to_pandas") else X
        try:
            X_out, y_out = func(X_prep, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Pandas", "apply", func)
            raise
    else:
        # Registered engine with an implementation but no input-preparation
        # path: fail loudly rather than silently collecting to pandas (F-09).
        raise NotImplementedError(
            f"No '{engine.name}' input-preparation path in apply_dual_engine yet"
        )

    return pack_pipeline_output(X_out, y_out, is_tuple)


def fit_dual_engine(
    df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
    params: dict[str, Any],
    implementations: Mapping[str, FitFunction],
) -> dict[str, Any]:
    """
    Dispatcher to handle boilerplate for dual-engine Calculators.

    Args:
        df: Inputs.
        params: Config.
        implementations: Engine-specific implementations keyed by engine name,
                     e.g. ``{"polars": fn_pl, "pandas": fn_pd}``.
                     Signature: (X, y, params) -> Dict[Result]

    Returns:
        Dictionary of fitted parameters.

    Raises:
        NotImplementedError: If no implementation is registered for the input's
            engine, or the engine has no input-preparation path yet.
    """
    X, y, _ = unpack_pipeline_input(df)
    _check_xy_engine_parity(X, y)
    engine = get_engine(X)
    func = _resolve_impl(implementations, engine.name, "fit")

    if engine.name == EngineName.POLARS:
        X_prep, _ = _unwrap_polars_wrapper(X)
        try:
            return dict(func(X_prep, y, params))
        except Exception as exc:
            _log_dispatch_failure(exc, "Polars", "fit", func)
            raise
    elif engine.name == EngineName.PANDAS:
        X_prep = X.to_pandas() if hasattr(X, "to_pandas") else X
        try:
            return dict(func(X_prep, y, params))
        except Exception as exc:
            _log_dispatch_failure(exc, "Pandas", "fit", func)
            raise
    else:
        raise NotImplementedError(
            f"No '{engine.name}' input-preparation path in fit_dual_engine yet"
        )


def fit_transform_train_dual_engine(
    df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
    params: dict[str, Any],
    implementations: Mapping[str, TrainTransformFunction],
) -> tuple[dict[str, Any], Any]:
    """Dispatch an optional fit+train-transform hook across supported engines.

    Args:
        df: Inputs.
        params: Config.
        implementations: Engine-specific implementations keyed by engine name,
                     e.g. ``{"polars": fn_pl, "pandas": fn_pd}``.
                     Signature: (X, y, params) -> (Dict[Result], X_out, y_out)

    Raises:
        NotImplementedError: If no implementation is registered for the input's
            engine, or the engine has no input-preparation path yet.
    """
    X, y, is_tuple = unpack_pipeline_input(df)
    _check_xy_engine_parity(X, y)
    engine = get_engine(X)
    func = _resolve_impl(implementations, engine.name, "fit_transform_train")

    if engine.name == EngineName.POLARS:
        X_prep, was_wrapped = _unwrap_polars_wrapper(X)
        try:
            artifact, X_out, y_out = func(X_prep, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Polars", "fit_transform_train", func)
            raise
        X_out = _rewrap_polars_output(X_out, was_wrapped)
    elif engine.name == EngineName.PANDAS:
        X_prep = X.to_pandas() if hasattr(X, "to_pandas") else X
        try:
            artifact, X_out, y_out = func(X_prep, y, params)
        except Exception as exc:
            _log_dispatch_failure(exc, "Pandas", "fit_transform_train", func)
            raise
    else:
        raise NotImplementedError(
            f"No '{engine.name}' input-preparation path in fit_transform_train_dual_engine yet"
        )

    return dict(artifact), pack_pipeline_output(X_out, y_out, is_tuple)
