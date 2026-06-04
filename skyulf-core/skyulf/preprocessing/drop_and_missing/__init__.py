"""Drop / missing-value nodes package.

Split from a single 409-LOC module into per-node files:
  _common.py            — shared helpers (y-index sync, subset normalization)
  deduplicate.py        — Deduplicate
  drop_columns.py       — DropMissingColumns
  drop_rows.py          — DropMissingRows
  missing_indicator.py  — MissingIndicator

All public names are re-exported here so existing imports such as
``from skyulf.preprocessing.drop_and_missing import DeduplicateCalculator``
continue to work unchanged.
"""

from .deduplicate import DeduplicateApplier, DeduplicateCalculator
from .drop_columns import DropMissingColumnsApplier, DropMissingColumnsCalculator
from .drop_rows import DropMissingRowsApplier, DropMissingRowsCalculator
from .missing_indicator import MissingIndicatorApplier, MissingIndicatorCalculator

__all__ = [
    "DeduplicateApplier",
    "DeduplicateCalculator",
    "DropMissingColumnsApplier",
    "DropMissingColumnsCalculator",
    "DropMissingRowsApplier",
    "DropMissingRowsCalculator",
    "MissingIndicatorApplier",
    "MissingIndicatorCalculator",
]
