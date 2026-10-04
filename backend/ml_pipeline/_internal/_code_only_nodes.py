"""Keep code-only Core nodes out of server-side graph execution.

``ColumnFunction``, ``FittedFunction`` and ``RowFilterFunction`` call a Python
function named in their parameters. They are meant for SDK and Bundle project
code; accepting them from an HTTP graph would let a request choose which
server-side function runs.
"""

from collections.abc import Iterable
from typing import Any

from skyulf.registry import NodeRegistry

_CODE_ONLY_TAG = "code_only"


def code_only_step_types() -> frozenset[str]:
    """Return registry names whose metadata marks them as code-only."""
    return frozenset(
        name
        for name, meta in NodeRegistry.get_all_metadata().items()
        if _CODE_ONLY_TAG in (meta.get("tags") or [])
    )


def reject_code_only_steps(step_type: str, params: dict[str, Any]) -> None:
    """Raise when a node, or a step nested in its ``steps`` list, is code-only."""
    blocked = code_only_step_types()
    found = [name for name in _step_types(step_type, params) if name in blocked]
    if found:
        raise ValueError(
            f"Step type {found[0]} runs project Python functions and is available only "
            "in the skyulf SDK or Databricks Bundle project code, not in canvas graphs."
        )


def _step_types(step_type: str, params: dict[str, Any]) -> Iterable[str]:
    """Yield the node type and every nested feature-engineering transformer."""
    yield step_type
    steps = params.get("steps") if isinstance(params, dict) else None
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and isinstance(step.get("transformer"), str):
                yield step["transformer"]
