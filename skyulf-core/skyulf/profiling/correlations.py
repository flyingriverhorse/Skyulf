import logging
from typing import cast

import numpy as np
import polars as pl

from .schemas import CorrelationMatrix

logger = logging.getLogger(__name__)

# A Pearson coefficient from two shared observations is always exactly +/-1.0,
# so such a cell is degenerate rather than merely noisy. Three is the smallest
# overlap that can produce a non-degenerate value.
MIN_PAIRWISE_OVERLAP = 3


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Narrow `LazyFrame.collect()` back to `DataFrame` (sync path only)."""
    return cast(pl.DataFrame, lf.collect())


def _cap_numeric_columns(numeric_cols: list[str]) -> list[str]:
    """Truncate to the first 20 numeric columns, logging a warning if any were dropped."""
    # HARD LIMIT: Top 20 numeric columns to prevent backend/frontend crash
    # Reduced from 50 to 20 as per user report of crashes
    if len(numeric_cols) > 20:
        dropped_cols = numeric_cols[20:]
        logger.warning(
            "calculate_correlations: %d numeric columns exceeds the 20-column "
            "cap; truncating to the first 20 by column order (not variance or "
            "relevance) and dropping %s from the correlation matrix.",
            len(numeric_cols),
            dropped_cols,
        )
        numeric_cols = numeric_cols[:20]
    return numeric_cols


def _nan_to_null(subset: pl.DataFrame) -> pl.DataFrame:
    """Rewrite NaN to null in every float column.

    polars keeps NaN distinct from null, and both consumers below are
    NaN-blind in a damaging way: ``std()`` of a NaN-bearing column is NaN,
    which :func:`_filter_constant_columns` rejects as "not > 1e-9" and so
    silently discards a perfectly usable column (and drops the *whole* matrix
    when fewer than two columns survive); and ``drop_nulls()`` keeps NaN rows,
    which then poison every coefficient. ``EDAAnalyzer`` normalizes at
    construction, but ``calculate_correlations`` is public API and can be
    handed any frame. Duplicated from the analyzer deliberately: importing
    ``_analyzer._utils`` would execute that package's ``__init__`` and pull the
    sklearn/scipy/statsmodels mixins into this leaf module.
    """
    float_cols = [name for name, dtype in subset.schema.items() if dtype.is_float()]
    if not float_cols:
        return subset
    return subset.with_columns([pl.col(c).fill_nan(None) for c in float_cols])


def _filter_constant_columns(subset: pl.DataFrame, numeric_cols: list[str]) -> list[str]:
    """Return the subset of columns with non-null, non-zero standard deviation."""
    # Handle constant columns to avoid RuntimeWarning: invalid value encountered in divide
    # Filter out columns with 0 std dev
    valid_cols = []
    for col in numeric_cols:
        std_val = subset[col].std()
        # Check for None (all nulls or single value) and 0 variance
        if std_val is not None and cast(float, std_val) > 1e-9:
            valid_cols.append(col)
    return valid_cols


def _clean_cell(value: object) -> float:
    """Coerce one correlation coefficient to a finite float (NaN/None/Inf -> 0.0)."""
    if value is None:
        return 0.0
    number = float(cast(float, value))
    return number if np.isfinite(number) else 0.0


def _pairwise_correlation_matrix(subset: pl.DataFrame, cols: list[str]) -> list[list[float]]:
    """Pearson matrix using pairwise deletion, i.e. pandas ``.corr()`` semantics.

    polars' ``DataFrame.corr()`` is listwise: it returns NaN for *every* cell
    when any column holds a null. The old workaround — ``drop_nulls()`` then
    ``corr()`` — discarded every row missing *any* column, so a sparse frame
    collapsed onto a handful of complete rows (or none at all) and the matrix
    was silently scored on a different sample per nothing: one shared sample
    for all cells, usually far smaller than the data. ``pl.corr(a, b)`` skips
    nulls per pair, so each cell is scored on its own overlap, matching pandas.
    """
    pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    n = len(cols)
    matrix = [[1.0] * n for _ in range(n)]
    if not pairs:
        return matrix

    exprs = []
    for i, j in pairs:
        exprs.append(pl.corr(cols[i], cols[j]).alias(f"r__{i}_{j}"))
        exprs.append(
            (pl.col(cols[i]).is_not_null() & pl.col(cols[j]).is_not_null())
            .sum()
            .alias(f"n__{i}_{j}")
        )
    row = subset.select(exprs).row(0, named=True)

    degenerate: list[str] = []
    for i, j in pairs:
        overlap = int(row[f"n__{i}_{j}"] or 0)
        if overlap < MIN_PAIRWISE_OVERLAP:
            degenerate.append(f"{cols[i]}~{cols[j]} (n={overlap})")
            value: object = 0.0
        else:
            value = row[f"r__{i}_{j}"]
        matrix[i][j] = matrix[j][i] = _clean_cell(value)

    if degenerate:
        logger.warning(
            "calculate_correlations: %d column pair(s) share fewer than %d "
            "observations and are reported as 0.0 instead of a coefficient, "
            "because at n=2 Pearson r is always exactly +/-1.0: %s. "
            "CorrelationMatrix has no 'unknown' cell, so 0.0 doubles for "
            "'not computable' (as it already does for NaN).",
            len(degenerate),
            MIN_PAIRWISE_OVERLAP,
            degenerate,
        )
    return matrix


def calculate_correlations(df: pl.LazyFrame, numeric_cols: list[str]) -> CorrelationMatrix | None:
    """
    Calculates Pearson correlation matrix for numeric columns.

    Missing values use **pairwise deletion** (pandas ``.corr()`` semantics):
    each coefficient is computed over the rows where *that pair* is observed,
    rather than over rows complete across every column. Pairs sharing fewer
    than :data:`MIN_PAIRWISE_OVERLAP` observations are reported as ``0.0`` with
    a warning, since a 2-point coefficient is degenerate. Columns with no
    variance (constant, or entirely missing) are excluded from the matrix.
    """
    if len(numeric_cols) < 2:
        return None

    try:
        # One collect for the whole matrix: N is capped at 20 columns, and the
        # pairwise pass below needs the values materialized anyway.

        numeric_cols = _cap_numeric_columns(numeric_cols)

        subset = _nan_to_null(_collect(df.select(numeric_cols)))

        if subset.height < 2:
            return None

        valid_cols = _filter_constant_columns(subset, numeric_cols)

        if len(valid_cols) < 2:
            return None

        # Re-select only valid columns
        subset = subset.select(valid_cols)

        matrix = _pairwise_correlation_matrix(subset, valid_cols)

        return CorrelationMatrix(columns=valid_cols, values=matrix)

    except Exception:
        logger.exception("Error calculating correlations")
        return None
