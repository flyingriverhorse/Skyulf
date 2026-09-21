"""Dual-engine dispatch for preprocessing nodes.

This module owns the *control flow* that lets a single node run on either the
Polars, pandas or Spark engine: ``apply_dual_engine`` (and its fit counterparts)
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

from ..engines import (
    EngineName,
    SkyulfDataFrame,
    SkyulfPolarsWrapper,
    SkyulfSparkWrapper,
    get_engine,
)
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
    """Dispatcher to handle boilerplate for dual-engine Appliers.

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

    X_prep, was_wrapped = _prepare_input(X, engine.name, "apply_dual_engine")
    try:
        X_out, y_out = func(X_prep, y, params)
    except Exception as exc:
        _log_dispatch_failure(exc, engine.name.capitalize(), "apply", func)
        raise
    X_out = _restore_output(X_out, was_wrapped, engine.name)

    return pack_pipeline_output(X_out, y_out, is_tuple)


def fit_dual_engine(
    df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
    params: dict[str, Any],
    implementations: Mapping[str, FitFunction],
) -> dict[str, Any]:
    """Dispatcher to handle boilerplate for dual-engine Calculators.

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

    X_prep, _ = _prepare_input(X, engine.name, "fit_dual_engine")
    try:
        return dict(func(X_prep, y, params))
    except Exception as exc:
        _log_dispatch_failure(exc, engine.name.capitalize(), "fit", func)
        raise


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

    X_prep, was_wrapped = _prepare_input(X, engine.name, "fit_transform_train_dual_engine")
    try:
        artifact, X_out, y_out = func(X_prep, y, params)
    except Exception as exc:
        _log_dispatch_failure(exc, engine.name.capitalize(), "fit_transform_train", func)
        raise
    X_out = _restore_output(X_out, was_wrapped, engine.name)

    return dict(artifact), pack_pipeline_output(X_out, y_out, is_tuple)


def _prepare_pandas(X: Any) -> tuple[Any, bool]:
    """Preserve the local pandas conversion contract."""
    return (X.to_pandas() if hasattr(X, "to_pandas") else X), False


def _prepare_spark(X: Any) -> tuple[Any, bool]:
    """Keep Spark input distributed and remember wrapper shape."""
    wrapped = isinstance(X, SkyulfSparkWrapper)
    return (X.to_native() if wrapped else X), wrapped


def _prepare_input(X: Any, engine: str, caller: str) -> tuple[Any, bool]:
    """Resolve input preparation by engine without a conversion fallback."""
    preparations = {
        "pandas": _prepare_pandas,
        "polars": _unwrap_polars_wrapper,
        "spark": _prepare_spark,
    }
    prepare = preparations.get(engine)
    if prepare is None:
        raise NotImplementedError(f"No '{engine}' input-preparation path in {caller} yet")
    return prepare(X)


def _restore_output(X: Any, was_wrapped: bool, engine: str) -> Any:
    """Restore only the input wrapper convention, never collecting distributed output."""
    if engine == "spark" and was_wrapped:
        return SkyulfSparkWrapper(X)
    return _rewrap_polars_output(X, was_wrapped)
