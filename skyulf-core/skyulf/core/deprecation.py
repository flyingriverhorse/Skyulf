"""Deprecation policy and helpers for ``skyulf-core``.

Policy: a public symbol marked deprecated in minor ``X.Y`` keeps working through
``X.(Y+1)`` and may be removed in ``X.(Y+2)``. Deprecating a symbol emits a
:class:`DeprecationWarning` (so it is silent by default but visible under
``python -W default`` and in test suites) describing the replacement and the
version in which removal is planned.

Usage:
    >>> from skyulf.core.deprecation import deprecated
    >>> @deprecated(since="0.2", removed_in="0.4", replacement="new_func")
    ... def old_func():
    ...     ...
"""

import functools
import warnings
from collections.abc import Callable
from typing import Any, cast

__all__ = ["deprecated", "warn_deprecated"]


def _build_message(
    name: str,
    since: str | None,
    removed_in: str | None,
    replacement: str | None,
) -> str:
    parts = [f"'{name}' is deprecated"]
    if since:
        parts.append(f" since v{since}")
    if removed_in:
        parts.append(f" and is scheduled for removal in v{removed_in}")
    parts.append(".")
    if replacement:
        parts.append(f" Use '{replacement}' instead.")
    return "".join(parts)


def warn_deprecated(
    name: str,
    *,
    since: str | None = None,
    removed_in: str | None = None,
    replacement: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a standardised ``DeprecationWarning`` for ``name``.

    Use directly for deprecating things a decorator can't wrap (e.g. a config
    key or a positional argument).
    """
    warnings.warn(
        _build_message(name, since, removed_in, replacement),
        category=DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated[F: Callable[..., Any]](
    *,
    since: str | None = None,
    removed_in: str | None = None,
    replacement: str | None = None,
) -> Callable[[F], F]:
    """Decorator that marks a callable as deprecated.

    Wraps the callable so that calling it emits a :class:`DeprecationWarning`
    built from ``since`` / ``removed_in`` / ``replacement``. The original
    behaviour and return value are preserved.
    """

    def decorator(func: F) -> F:
        message = _build_message(
            getattr(func, "__qualname__", getattr(func, "__name__", "callable")),
            since,
            removed_in,
            replacement,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
