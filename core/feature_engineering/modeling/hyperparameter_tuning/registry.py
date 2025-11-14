"""Dynamic registry for hyperparameter tuning search strategies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

from config import get_settings


@dataclass(frozen=True)
class TuningStrategyOption:
    """Represents a selectable hyperparameter search strategy."""

    value: str
    label: str
    description: str
    impl: str
    aliases: Tuple[str, ...]


_DEFAULT_TUNING_STRATEGIES: Sequence[Dict[str, object]] = (
    {
        "value": "random",
        "label": "Random search",
        "description": "Sample candidate hyperparameters uniformly at random.",
        "impl": "random",
        "aliases": ("random_search",),
    },
    {
        "value": "grid",
        "label": "Grid search",
        "description": "Evaluate every combination in the search space.",
        "impl": "grid",
        "aliases": ("grid_search",),
    },
    {
        "value": "halving",
        "label": "Successive halving (grid)",
        "description": "Successively allocate resources to the best grid candidates.",
        "impl": "halving",
        "aliases": ("successive_halving", "halving_grid"),
    },
    {
        "value": "halving_random",
        "label": "Successive halving (random)",
        "description": "Random sampling with successive halving to prune weak candidates.",
        "impl": "halving_random",
        "aliases": ("successive_halving_random", "halving_search"),
    },
    {
        "value": "optuna",
        "label": "Optuna (TPE)",
        "description": "Bayesian optimisation with pruning via Optuna.",
        "impl": "optuna",
        "aliases": ("bayesian", "optuna_tpe"),
    },
)


_SUPPORTED_IMPLS = {"random", "grid", "halving", "halving_random", "optuna"}


def _as_tuple(sequence: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if not sequence:
        return tuple()
    result: List[str] = []
    seen: set[str] = set()
    for entry in sequence:
        if entry is None:
            continue
        candidate = str(entry).strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(candidate)
    return tuple(result)


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
    if impl not in _SUPPORTED_IMPLS:
        # Fallback to random search when an unknown implementation is provided
        impl = "random"

    aliases_raw = raw.get("aliases")
    if isinstance(aliases_raw, (list, tuple, set)):
        aliases_iter: Optional[Iterable[Any]] = cast(Optional[Iterable[Any]], aliases_raw)
    elif aliases_raw is None:
        aliases_iter = None
    else:
        aliases_iter = [aliases_raw]

    aliases = _as_tuple(aliases_iter)

    seen.add(lowered_value)
    return TuningStrategyOption(
        value=value,
        label=label,
        description=description,
        impl=impl,
        aliases=aliases,
    )


@lru_cache(maxsize=1)
def get_tuning_strategy_options() -> Tuple[TuningStrategyOption, ...]:
    settings = get_settings()
    raw_entries = getattr(settings, "HYPERPARAMETER_TUNING_STRATEGIES", None)

    entries: List[TuningStrategyOption] = []
    seen: set[str] = set()
    source: Iterable[Dict[str, object]]
    if isinstance(raw_entries, Sequence) and raw_entries:
        normalized_source: List[Dict[str, object]] = []
        for item in raw_entries:
            if isinstance(item, dict):
                normalized_source.append(item)
            elif isinstance(item, str):
                normalized_source.append({"value": item})
        source = normalized_source
    else:
        source = _DEFAULT_TUNING_STRATEGIES

    for raw in source:
        entry = _normalize_entry(raw, seen)
        if entry:
            entries.append(entry)

    if not entries:
        for raw in _DEFAULT_TUNING_STRATEGIES:
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
    # Fall back to default when the value is unexpected
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
