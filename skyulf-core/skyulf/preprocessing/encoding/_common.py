"""Shared helpers and constants for all encoder modules."""

import logging
import re
import unicodedata
from typing import Any

import polars as pl

from ...engines import EngineName, get_engine
from ...utils import is_decimal_series
from .._schema import SkyulfSchema

logger = logging.getLogger(__name__)


def encoded_columns_schema(
    input_schema: SkyulfSchema, columns: list[str] | None, dtype: str
) -> SkyulfSchema | None:
    """Describe explicit in-place encoding; abstain when values determine column selection."""
    if columns is None:
        return None
    output = input_schema
    for column in columns:
        if column in input_schema.columns:
            output = output.with_dtype(column, dtype)
    return output


def _category_slug(value: str) -> str:
    """Fold category text to an ASCII name without losing Danish or German letters."""
    folded = value.lower().translate(str.maketrans({"æ": "ae", "ø": "oe", "å": "aa", "ß": "ss"}))
    ascii_text = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    if not slug:
        raise ValueError(f"OneHotEncoder category {value!r} produces an empty slug.")
    return slug


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
    """Split string/list input into nonempty lines, rejecting unsupported explicit orders."""
    if isinstance(raw, str):
        return [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if isinstance(raw, list) and all(isinstance(line, str) for line in raw):
        return [line.strip() for line in raw if line.strip()]
    raise ValueError(
        "OrdinalEncoder categories_order must be a multiline string or a list "
        "of comma-separated strings, one line per selected column."
    )


def _split_category_line(line: str) -> list[str]:
    """Split a single comma-separated category line into stripped non-empty values."""
    return [v.strip() for v in line.split(",") if v.strip()]


def _parse_categories_order(raw: Any, n_cols: int) -> str | list[list[str]]:
    """Convert the frontend categories_order string/list into sklearn-compatible categories.

    The frontend sends a newline-separated string where each line holds
    comma-separated category values for one column (in column-selection order).
    Empty input selects automatic ordering. An explicit order must contain one
    nonempty comma-separated category line per selected column.
    """
    if raw is None:
        return "auto"

    lines = _normalize_categories_lines(raw)
    if not lines:
        return "auto"
    if len(lines) != n_cols:
        raise ValueError(
            f"OrdinalEncoder categories_order requires {n_cols} lines, one per selected "
            f"column; got {len(lines)}."
        )
    categories = [_split_category_line(line) for line in lines]
    if any(not values for values in categories):
        raise ValueError("OrdinalEncoder categories_order lines must contain category values.")
    return categories


def _detect_target_column(config: dict[str, Any], y: Any) -> str | None:
    """Resolve the target column name from config['target_column'] or the name of y."""
    target_col: str | None = config.get("target_column")
    if target_col is None and y is not None:
        target_col = getattr(y, "name", None)
    return target_col


def _extract_target(X: Any, y: Any, target_col: str | None) -> Any:
    """Return explicit y or extract target_col from a Pandas/Polars frame."""
    if y is not None or not target_col:
        return y
    if target_col not in X.columns:
        return y
    getter = getattr(X, "get_column", None)
    return getter(target_col) if getter else X[target_col]


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
    """Remove the target column from the encoding list for encoders that must not encode it.

    Those are the column-destroying encoders, and the supervised in-place
    encoders that would produce a leaky/degenerate encoding against themselves.

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
    """Auto-detect the string/categorical columns an encoder should target.

    The fallback every encoder passes to ``resolve_columns`` when the user
    configured no explicit column list: polars frames are matched on
    ``Utf8``/``Categorical``/``Enum``/``Object`` dtypes, pandas frames via
    ``select_dtypes`` on ``object``/``category``/``string`` — so "encode categoricals"
    means the same thing on either engine.
    """
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        df_pl: Any = df
        return [
            c
            for c, t in zip(df_pl.columns, df_pl.dtypes, strict=True)
            if t in [pl.Utf8, pl.Categorical, pl.Object] or isinstance(t, pl.Enum)
        ]
    return [
        col
        for col in df.select_dtypes(include=["object", "category", "string"]).columns
        if not is_decimal_series(df[col])
    ]
