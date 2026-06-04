"""Lightweight data-validation expectations (no Great Expectations dependency).

Each ``expect_*`` function checks a single condition on a DataFrame and raises
:class:`ExpectationError` with a precise message when the condition is violated.
Pure-Python and engine-agnostic: Pandas frames are used directly; Polars (or any
frame exposing ``to_pandas()``) is converted first.

Example:
    >>> import pandas as pd
    >>> from skyulf.profiling.expect import expect_no_nulls, expect_value_range
    >>> df = pd.DataFrame({"age": [21, 35, 40]})
    >>> expect_no_nulls(df)
    >>> expect_value_range(df, "age", minimum=0, maximum=120)
"""

from typing import Any, List, Optional, Sequence

import pandas as pd

__all__ = [
    "ExpectationError",
    "expect_columns_exist",
    "expect_no_nulls",
    "expect_value_range",
    "expect_unique",
]


class ExpectationError(ValueError):
    """Raised when a data-validation expectation is not met."""


def _as_pandas(df: Any) -> pd.DataFrame:
    """Return a Pandas view of ``df`` (converts Polars/wrapper frames)."""
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    raise TypeError(f"Unsupported frame type for expectations: {type(df)!r}")


def _resolve_columns(df: pd.DataFrame, columns: Optional[Sequence[str]]) -> List[str]:
    """Default to all columns; otherwise validate the requested subset exists."""
    if columns is None:
        return list(df.columns)
    expect_columns_exist(df, columns)
    return list(columns)


def expect_columns_exist(df: Any, columns: Sequence[str]) -> None:
    """Assert that every name in ``columns`` is present in ``df``."""
    frame = _as_pandas(df)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ExpectationError(f"Expected columns are missing: {missing}")


def expect_no_nulls(df: Any, columns: Optional[Sequence[str]] = None) -> None:
    """Assert that the given columns (default: all) contain no null values."""
    frame = _as_pandas(df)
    cols = _resolve_columns(frame, columns)
    null_counts = {c: int(frame[c].isnull().sum()) for c in cols}
    offenders = {c: n for c, n in null_counts.items() if n > 0}
    if offenders:
        raise ExpectationError(f"Null values found in columns: {offenders}")


def expect_value_range(
    df: Any,
    column: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    inclusive: bool = True,
) -> None:
    """Assert that all values in ``column`` fall within ``[minimum, maximum]``.

    ``minimum`` / ``maximum`` are optional (open-ended on the unset side).
    Null values are ignored. Set ``inclusive=False`` for a strict comparison.
    """
    frame = _as_pandas(df)
    expect_columns_exist(frame, [column])
    series = frame[column].dropna()
    _check_lower_bound(series, column, minimum, inclusive)
    _check_upper_bound(series, column, maximum, inclusive)


def _check_lower_bound(series: Any, column: str, minimum: Optional[float], inclusive: bool) -> None:
    if minimum is None:
        return
    violates = series < minimum if inclusive else series <= minimum
    if bool(violates.any()):
        bound = ">=" if inclusive else ">"
        raise ExpectationError(
            f"Column '{column}' has values that are not {bound} {minimum} "
            f"(min observed: {series.min()})"
        )


def _check_upper_bound(series: Any, column: str, maximum: Optional[float], inclusive: bool) -> None:
    if maximum is None:
        return
    violates = series > maximum if inclusive else series >= maximum
    if bool(violates.any()):
        bound = "<=" if inclusive else "<"
        raise ExpectationError(
            f"Column '{column}' has values that are not {bound} {maximum} "
            f"(max observed: {series.max()})"
        )


def expect_unique(df: Any, columns: Sequence[str]) -> None:
    """Assert that the combination of ``columns`` has no duplicate rows."""
    frame = _as_pandas(df)
    expect_columns_exist(frame, columns)
    duplicated = frame.duplicated(subset=list(columns), keep=False)
    dup_count = int(duplicated.sum())
    if dup_count:
        raise ExpectationError(
            f"Expected unique values for {list(columns)} but found {dup_count} duplicate rows"
        )
