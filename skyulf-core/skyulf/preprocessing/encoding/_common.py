"""Shared helpers and constants for all encoder modules."""

import logging
from typing import Any

from ...engines import EngineName, get_engine

logger = logging.getLogger(__name__)

# Encoders that destroy the original column structure (create N new columns).
# These must NEVER be applied to a target column.
_COLUMN_DESTROYING_ENCODERS = frozenset(
    [
        "OneHotEncoder",
        "DummyEncoder",
        "HashEncoder",
        "TargetEncoder",
    ]
)

# Encoders that fit against `y` and replace values in-place (column name/count
# preserved, so they don't break Feature/Target Split by renaming) but would
# produce a degenerate, leaky encoding if the target column were included in
# its own `columns` list (e.g. WOE computed against itself is near-perfect
# separation, and silently overwrites the target's own values).
_SUPERVISED_INPLACE_ENCODERS = frozenset(["WOEEncoder"])


def _normalize_categories_lines(raw: Any) -> list[str]:
    """Split raw string/list input into non-empty stripped lines; empty list if unsupported type."""
    if isinstance(raw, str):
        return [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if isinstance(raw, list):
        return [str(ln).strip() for ln in raw if str(ln).strip()]
    return []


def _split_category_line(line: str) -> list[str]:
    """Split a single comma-separated category line into stripped non-empty values."""
    return [v.strip() for v in line.split(",") if v.strip()]


def _parse_categories_order(raw: Any, n_cols: int) -> str | list[list[str]]:
    """Convert the frontend categories_order string/list into sklearn-compatible categories.

    The frontend sends a newline-separated string where each line holds
    comma-separated category values for one column (in column-selection order).
    If the number of non-empty lines matches n_cols, a list-of-lists is returned.
    Any other case falls back to "auto".
    """
    if not raw:
        return "auto"
    if not isinstance(raw, str | list):
        return "auto"

    lines = _normalize_categories_lines(raw)
    if not lines or len(lines) != n_cols:
        return "auto"

    return [_split_category_line(line) for line in lines]


def _detect_target_column(config: dict[str, Any], y: Any) -> str | None:
    """Resolve the target column name from config['target_column'] or the name of y."""
    target_col: str | None = config.get("target_column")
    if target_col is None and y is not None:
        target_col = getattr(y, "name", None)
    return target_col


def _warn_excluding_target_column(encoder_name: str, target_col: str) -> None:
    """Log a warning explaining why the target column is excluded from encoding."""
    if encoder_name in _COLUMN_DESTROYING_ENCODERS:
        reason = (
            f"{encoder_name} would replace the column with multiple derived columns, "
            "breaking downstream Feature/Target Split and model training."
        )
    else:
        reason = (
            f"{encoder_name} fits against the target and replaces values in-place; "
            "encoding the target against itself produces a degenerate, leaky mapping "
            "and silently overwrites the target's own values."
        )
    logger.warning(
        f"{encoder_name}: Excluding target column '{target_col}' from encoding. "
        f"{reason} Use LabelEncoder or OrdinalEncoder for target columns instead."
    )


def _exclude_target_column(
    columns: list[str],
    config: dict[str, Any],
    encoder_name: str,
    y: Any = None,
) -> list[str]:
    """Remove the target column from the encoding list for encoders that must
    not encode it (column-destroying encoders, and supervised in-place
    encoders that would produce a leaky/degenerate encoding against
    themselves).

    Detects the target column from config['target_column'] or the name of y.
    Returns the filtered column list and logs a warning when a column is removed.
    """
    if (
        encoder_name not in _COLUMN_DESTROYING_ENCODERS
        and encoder_name not in _SUPERVISED_INPLACE_ENCODERS
    ):
        return columns

    target_col = _detect_target_column(config, y)

    if target_col and target_col in columns:
        _warn_excluding_target_column(encoder_name, target_col)
        columns = [c for c in columns if c != target_col]

    return columns


def detect_categorical_columns(df: Any) -> list[str]:
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        import polars as pl

        df_pl: Any = df
        return [
            c
            for c, t in zip(df_pl.columns, df_pl.dtypes, strict=True)
            if t in [pl.Utf8, pl.Categorical, pl.Object]
        ]
    return df.select_dtypes(include=["object", "category"]).columns.tolist()
