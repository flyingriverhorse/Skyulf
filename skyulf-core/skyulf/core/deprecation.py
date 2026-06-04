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
from typing import Any, Callable, Optional, TypeVar, cast

__all__ = ["deprecated", "warn_deprecated"]

_F = TypeVar("_F", bound=Callable[..., Any])


def _build_message(
    name: str,
    since: Optional[str],
    removed_in: Optional[str],
    replacement: Optional[str],
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
    since: Optional[str] = None,
    removed_in: Optional[str] = None,
    replacement: Optional[str] = None,
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


def deprecated(
    *,
    since: Optional[str] = None,
    removed_in: Optional[str] = None,
    replacement: Optional[str] = None,
) -> Callable[[_F], _F]:
    """Decorator that marks a callable as deprecated.

    Wraps the callable so that calling it emits a :class:`DeprecationWarning`
    built from ``since`` / ``removed_in`` / ``replacement``. The original
    behaviour and return value are preserved.
    """

    def decorator(func: _F) -> _F:
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

        return cast(_F, wrapper)

    return decorator
