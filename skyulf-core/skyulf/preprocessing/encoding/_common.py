"""Shared helpers and constants for all encoder modules."""

import logging
from typing import Any, Dict, List, Union

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

# Encoders that map values in-place and preserve the column name.
_TARGET_SAFE_ENCODERS = frozenset(["LabelEncoder", "OrdinalEncoder"])


def _parse_categories_order(raw: Any, n_cols: int) -> Union[str, List[List[str]]]:
    """Convert the frontend categories_order string/list into sklearn-compatible categories.

    The frontend sends a newline-separated string where each line holds
    comma-separated category values for one column (in column-selection order).
    If the number of non-empty lines matches n_cols, a list-of-lists is returned.
    Any other case falls back to "auto".
    """
    if not raw:
        return "auto"
    if isinstance(raw, str):
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    elif isinstance(raw, list):
        lines = [str(ln).strip() for ln in raw if str(ln).strip()]
    else:
        return "auto"

    if not lines or len(lines) != n_cols:
        return "auto"

    return [[v.strip() for v in line.split(",") if v.strip()] for line in lines]


def _exclude_target_column(
    columns: List[str],
    config: Dict[str, Any],
    encoder_name: str,
    y: Any = None,
) -> List[str]:
    """Remove the target column from the encoding list for column-destroying encoders.

    Detects the target column from config['target_column'] or the name of y.
    Returns the filtered column list and logs a warning when a column is removed.
    """
    if encoder_name not in _COLUMN_DESTROYING_ENCODERS:
        return columns

    target_col: str | None = config.get("target_column")
    if target_col is None and y is not None:
        target_col = getattr(y, "name", None)

    if target_col and target_col in columns:
        logger.warning(
            f"{encoder_name}: Excluding target column '{target_col}' from encoding. "
            f"{encoder_name} would replace the column with multiple derived columns, "
            "breaking downstream Feature/Target Split and model training. "
            "Use LabelEncoder or OrdinalEncoder for target columns instead."
        )
        columns = [c for c in columns if c != target_col]

    return columns


def detect_categorical_columns(df: Any) -> List[str]:
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        import polars as pl

        df_pl: Any = df
        return list(
            c
            for c, t in zip(df_pl.columns, df_pl.dtypes)
            if t in [pl.Utf8, pl.Categorical, pl.Object]
        )
    return df.select_dtypes(include=["object", "category"]).columns.tolist()
