"""Core types for the tuning strategy registry."""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TuningStrategyOption:
    """Represents a selectable hyperparameter search strategy."""

    value: str
    label: str
    description: str
    impl: str
    aliases: Tuple[str, ...]


def normalize_aliases(sequence: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    """Return a normalized tuple of alias strings (lowercase deduplication)."""

    if not sequence:
        return tuple()

    entries: List[str] = []
    seen: set[str] = set()
    for raw in sequence:
        if raw is None:
            continue
        candidate = str(raw).strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        entries.append(candidate)
    return tuple(entries)
