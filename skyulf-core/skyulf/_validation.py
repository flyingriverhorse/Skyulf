"""Shared validation helpers for user-supplied node configuration."""

from collections.abc import Iterable
from typing import NoReturn


def raise_invalid_choice(
    value: object, valid_choices: Iterable[str | int], param_name: str
) -> NoReturn:
    """Raise a consistent error for an unsupported enum-like configuration value."""
    choices = sorted(valid_choices, key=str)
    raise ValueError(f"Unknown {param_name}: {value!r}. Valid choices: {choices}")
