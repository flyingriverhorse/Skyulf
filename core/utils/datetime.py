"""Datetime helpers used across the application."""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


__all__ = ["utcnow"]
