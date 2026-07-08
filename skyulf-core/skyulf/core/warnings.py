"""Structured warning classification for Skyulf nodes.

Nodes historically each invented their own ``node_warnings`` shape (bare strings,
ad-hoc dicts). ``SkyulfWarning`` gives every node one consistent, machine-readable
envelope: a category (for filtering/aggregation), a stable ``code`` (for docs and
suppression), a human message, and optional structured context.

This is additive — nodes can adopt it incrementally; nothing forces the old
string shape to change at once.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["WarningCategory", "SkyulfWarning"]


class WarningCategory(str, Enum):
    """Coarse buckets for filtering and aggregating node warnings."""

    DATA_QUALITY = "data_quality"
    CONFIG = "config"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"
    DEGENERATE = "degenerate"


@dataclass
class SkyulfWarning:
    """A single structured warning emitted by a node.

    Attributes:
        category: Coarse bucket (see :class:`WarningCategory`).
        code: Stable, greppable identifier (e.g. ``"onehot.single_category"``).
        message: Human-readable explanation.
        context: Optional structured detail (column names, counts, thresholds).
    """

    category: WarningCategory
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "category": self.category.value,
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }
