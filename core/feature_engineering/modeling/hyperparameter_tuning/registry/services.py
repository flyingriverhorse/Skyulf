"""Helpers for working with hyperparameter tuning strategies."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

from config import get_settings

from .base import TuningStrategyOption, normalize_aliases
from .defaults import DEFAULT_TUNING_STRATEGIES, SUPPORTED_IMPLS


def _normalize_entry(raw: Dict[str, object], seen: set[str]) -> Optional[TuningStrategyOption]:
    value = str(raw.get("value", "")).strip()
    if not value:
        return None

    lowered_value = value.lower()
    if lowered_value in seen:
        return None

    label = str(raw.get("label") or value.replace("_", " ").title()).strip()
    description = str(raw.get("description") or "").strip()
    impl = str(raw.get("impl") or value).strip().lower()
    if impl not in SUPPORTED_IMPLS:
        # Fallback to random search when an unknown implementation is provided
        impl = "random"

    aliases_raw = raw.get("aliases")
    if isinstance(aliases_raw, (list, tuple, set)):
        aliases_iter: Optional[Iterable[Any]] = cast(Optional[Iterable[Any]], aliases_raw)
    elif aliases_raw is None:
        aliases_iter = None
    else:
        aliases_iter = [aliases_raw]

    aliases = normalize_aliases(aliases_iter)

    seen.add(lowered_value)
    return TuningStrategyOption(
        value=value,
        label=label,
        description=description,
        impl=impl,
        aliases=aliases,
    )


def _resolve_source(raw_entries: Any) -> Iterable[Dict[str, object]]:
    if isinstance(raw_entries, Sequence) and raw_entries:
        normalized_source: List[Dict[str, object]] = []
        for item in raw_entries:
            if isinstance(item, dict):
                normalized_source.append(item)
            elif isinstance(item, str):
                normalized_source.append({"value": item})
        return normalized_source
    return DEFAULT_TUNING_STRATEGIES


@lru_cache(maxsize=1)
def get_tuning_strategy_options() -> Tuple[TuningStrategyOption, ...]:
    settings = get_settings()
    raw_entries = getattr(settings, "HYPERPARAMETER_TUNING_STRATEGIES", None)

    entries: List[TuningStrategyOption] = []
    seen: set[str] = set()
    source = _resolve_source(raw_entries)

    for raw in source:
        entry = _normalize_entry(raw, seen)
        if entry:
            entries.append(entry)

    if not entries:
        for raw in DEFAULT_TUNING_STRATEGIES:
            entry = _normalize_entry(raw, seen)
            if entry:
                entries.append(entry)

    return tuple(entries)


@lru_cache(maxsize=1)
def get_strategy_alias_map() -> Dict[str, TuningStrategyOption]:
    alias_map: Dict[str, TuningStrategyOption] = {}
    for option in get_tuning_strategy_options():
        alias_map[option.value.lower()] = option
        for alias in option.aliases:
            alias_map[alias.lower()] = option
    return alias_map


def get_default_strategy_value() -> str:
    options = get_tuning_strategy_options()
    if not options:
        return "random"
    return options[0].value


def normalize_strategy_value(value: Any) -> str:
    if not value:
        return get_default_strategy_value()
    candidate = str(value).strip().lower()
    option = get_strategy_alias_map().get(candidate)
    if option:
        return option.value
    return get_default_strategy_value()


def get_strategy_impl(value: Any) -> str:
    lookup_key = str(value or "").strip().lower()
    option = get_strategy_alias_map().get(lookup_key)
    if option:
        return option.impl
    default_value = get_default_strategy_value()
    default_option = get_strategy_alias_map().get(default_value.lower())
    return default_option.impl if default_option else "random"


def get_strategy_option(value: Any) -> Optional[TuningStrategyOption]:
    return get_strategy_alias_map().get(str(value or "").strip().lower())


def get_strategy_choices_for_ui() -> List[Dict[str, str]]:
    choices: List[Dict[str, str]] = []
    for option in get_tuning_strategy_options():
        entry = {"value": option.value, "label": option.label}
        if option.description:
            entry["description"] = option.description
        choices.append(entry)
    return choices


def get_supported_strategy_values() -> List[str]:
    return [option.value for option in get_tuning_strategy_options()]


def resolve_strategy_selection(value: Any) -> Tuple[str, str]:
    """Return the canonical value and implementation for the requested strategy."""

    normalized_value = normalize_strategy_value(value)
    option = get_strategy_option(normalized_value)
    if option:
        return option.value, option.impl
    default_value = get_default_strategy_value()
    default_option = get_strategy_option(default_value)
    impl = default_option.impl if default_option else "random"
    return default_value, impl
