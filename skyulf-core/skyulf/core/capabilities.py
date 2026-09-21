"""Explicit node execution support; declarations never perform fit or apply."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .execution import _ENGINES, _require_choice, _require_positive_integer

type ConfigValue = str | int | float | bool | None


class UnsupportedExecutionError(ValueError):
    """Describe the requested node operation and why execution is unavailable."""

    def __init__(self, node_type: str, operation: str, engine: str, reason: str) -> None:
        """Expose structured fields as well as a human-readable error."""
        self.node_type = node_type
        self.operation = operation
        self.engine = engine
        self.reason = reason
        super().__init__(f"{node_type}.{operation} on {engine}: {reason}")


def _validate_config_match(match: tuple[tuple[str, ConfigValue], ...]) -> None:
    """Keep capability selectors immutable and limited to exact scalar matches."""
    if not isinstance(match, tuple):
        raise ValueError("config_match must be a tuple of key/value tuples.")
    keys = []
    for pair in match:
        _validate_config_pair(pair)
        keys.append(pair[0])
    if len(set(keys)) != len(keys):
        raise ValueError("config_match must not repeat a key.")


def _validate_config_pair(pair: tuple[str, ConfigValue]) -> None:
    """Reject mutable, non-scalar and non-finite selector values."""
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError("config_match entries must be key/value tuples.")
    key, value = pair
    if not isinstance(key, str) or not key.strip():
        raise ValueError("config_match keys must be nonempty strings.")
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("config_match values must be scalar values.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("config_match values must be finite.")


@dataclass(frozen=True)
class ExecutionCapability:
    """Declare one operation's engine, row/context effects and state codec.

    Config selectors require explicit, type-exact values. Callers must normalize
    defaults before querying; missing keys never match an explicit selector.
    The declaration is an implementation promise, not runtime verification.
    """

    engine: str
    operation: str
    execution_kind: str
    row_effect: str
    context: str
    codec_version: int | None = None
    config_match: tuple[tuple[str, ConfigValue], ...] = ()

    def __post_init__(self) -> None:
        """Reject malformed metadata before it can advertise support."""
        _require_choice("engine", self.engine, _ENGINES)
        _require_choice("operation", self.operation, ("fit", "apply"))
        _require_choice("execution_kind", self.execution_kind, ("native", "python_batch", "local"))
        _require_choice("row_effect", self.row_effect, ("preserve", "filter", "expand"))
        _require_choice("context", self.context, ("row", "group", "window", "global"))
        if self.codec_version is not None:
            _require_positive_integer("codec_version", self.codec_version)
        _validate_config_match(self.config_match)

    def matches_config(self, config: Mapping[str, Any]) -> bool:
        """Require every selector, distinguishing booleans from numeric values."""
        return all(
            key in config and type(config[key]) is type(value) and config[key] == value
            for key, value in self.config_match
        )


def validate_capabilities(capabilities: tuple[ExecutionCapability, ...]) -> None:
    """Require immutable, validated declarations on a registered calculator."""
    if not isinstance(capabilities, tuple) or any(
        not isinstance(item, ExecutionCapability) for item in capabilities
    ):
        raise ValueError("execution_capabilities must be a tuple of ExecutionCapability objects.")


def require_capability(
    node_type: str, operation: str, engine: str, *, config: Mapping[str, Any]
) -> None:
    """Reject undeclared execution without initializing an engine or estimator.

    Existing local pipelines are unaffected until they explicitly use this
    preflight. An empty declaration means unsupported, including for local
    engines; it does not remove the existing local execution path.

    Raises:
        UnsupportedExecutionError: The node, operation, engine or configuration
            has no matching explicit execution declaration.
    """
    from ..registry import NodeRegistry  # noqa: PLC0415 - registry validates these declarations

    try:
        _require_choice("operation", operation, ("fit", "apply"))
        _require_choice("engine", engine, _ENGINES)
        calculator = NodeRegistry.get_calculator(node_type)
    except ValueError as exc:
        raise UnsupportedExecutionError(node_type, operation, engine, str(exc)) from exc
    capabilities = vars(calculator).get("__execution_capabilities__", ())
    validate_capabilities(capabilities)
    for capability in capabilities:
        if (
            capability.engine == engine
            and capability.operation == operation
            and capability.matches_config(config)
        ):
            return
    raise UnsupportedExecutionError(
        node_type, operation, engine, "No declared support for this operation and configuration."
    )
