"""Inspect fitted apply context without executing transformations or admitting workers."""

import re
from typing import Any

from ..core.capabilities import ExecutionCapability, validate_capabilities
from ..registry import NodeRegistry


def _declared_capability(
    node_type: str, config: dict, state: dict, applier: type, *, engine: str
) -> ExecutionCapability | None:
    """Reuse existing declarations and their node-owned fitted normalization."""
    calculator = NodeRegistry.get_calculator(node_type)
    capabilities = vars(calculator).get("__execution_capabilities__", ())
    validate_capabilities(capabilities)
    if not capabilities:
        return None
    validate = getattr(applier, "validate_fitted_state", None)
    resolve = getattr(applier, "resolve_fitted_config", None)
    try:
        if validate is not None:
            state = validate(state)
        normalized = resolve(config, state) if resolve is not None else config
    except (TypeError, ValueError):
        # Existing validators describe a reviewed worker subset, not every
        # valid local artifact. Unsupported state supplies no context promise.
        return None
    return next(
        (
            capability
            for capability in capabilities
            if capability.operation == "apply"
            and capability.engine == engine
            and capability.matches_config(normalized)
        ),
        None,
    )


def _hook_capability(applier: type, state: dict, *, engine: str) -> ExecutionCapability | None:
    """Require a hook to describe the requested apply operation or explicitly abstain."""
    hook = getattr(applier, "inference_capability", None)
    if not callable(hook):
        raise ValueError("inference_capability must be a callable hook.")
    capability = hook(state, engine=engine)
    if capability is None:
        return None
    if not isinstance(capability, ExecutionCapability):
        raise ValueError("inference_capability must return ExecutionCapability or None.")
    if capability.operation != "apply" or capability.engine != engine:
        raise ValueError("inference_capability must describe apply on the requested engine.")
    return capability


def _matches_applier(applier: Any, registered: type) -> bool:
    """Bind metadata to the saved class without borrowing it for instance overrides."""
    if isinstance(applier, type):
        return applier is registered
    if type(applier) is not registered:
        return False
    return not {"apply", "inference_capability"}.intersection(getattr(applier, "__dict__", {}))


def _captured_capability(
    node_type: str, state: dict, applier: Any, *, engine: str
) -> ExecutionCapability | None:
    """Inspect a source-named saved class without importing or registering its recipe."""
    if applier is None:
        return None
    saved = applier if isinstance(applier, type) else type(applier)
    if not _matches_applier(applier, saved) or not _captured_identity(node_type, saved):
        return None
    if "inference_capability" not in vars(saved):
        return None
    return _hook_capability(saved, state, engine=engine)


def _captured_identity(node_type: str, applier: type) -> bool:
    """Require the source digest namespace and the exact saved applier identity."""
    module = applier.__module__
    if re.fullmatch(r"_skyulf_project_[0-9a-f]{64}(?:\.[A-Za-z_]\w*)*", module) is None:
        return False
    identity = (
        rf"{re.escape(module)}\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\.{re.escape(applier.__qualname__)}"
    )
    return "<locals>" not in applier.__qualname__ and re.fullmatch(identity, node_type) is not None


def get_inference_capability(
    node_type: str,
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    engine: str,
    applier: Any = None,
) -> ExecutionCapability | None:
    """Describe saved apply context, returning None for undeclared implementations.

    An optional saved applier must match the registered class, or an unregistered
    captured project class with a matching source-specific node identity. The
    caller must verify that source digest against its trusted artifact manifest.
    A class-owned ``inference_capability(state, *, engine)`` hook takes precedence
    over static
    execution declarations; subclasses must declare their own hook. Existing
    declarations use the applier's fitted-state validation and configuration
    normalization when available. State outside those reviewed subsets remains
    unknown; malformed explicit hook declarations raise.

    This diagnostic query does not invoke fit, apply, recipe builders or custom
    transformation callbacks. Hooks are trusted code; their metadata never
    grants partition execution permission.
    """
    try:
        registered = NodeRegistry.get_applier(node_type)
    except ValueError:
        return _captured_capability(node_type, state, applier, engine=engine)
    if applier is not None and not _matches_applier(applier, registered):
        return None
    if "inference_capability" in vars(registered):
        return _hook_capability(registered, state, engine=engine)
    return _declared_capability(node_type, config, state, registered, engine=engine)
