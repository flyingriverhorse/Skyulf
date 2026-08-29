"""Mermaid diagram rendering for pipeline topology.

Extracted from ``pipeline.py`` (F-19): diagram generation is a read-only
presentation concern and has no business living next to the fit path.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def mermaid_escape(text: str) -> str:
    """Escape characters that would break a Mermaid node label."""
    return text.replace('"', "'").replace("[", "(").replace("]", ")")


def build_mermaid_diagram(
    preprocessing_steps: Sequence[Mapping[str, Any]],
    modeling_config: Mapping[str, Any],
) -> str:
    """Render a pipeline topology as a Mermaid ``flowchart`` string.

    Produces a top-down graph ``data -> [preprocessing steps] -> model``.
    Useful in docs and PR descriptions. Pure function over the config parts.
    """
    lines = ["flowchart TD", "    data[Input Data]"]
    prev = "data"

    for i, step in enumerate(preprocessing_steps):
        node = f"pp{i}"
        name = step.get("name", f"step_{i}")
        transformer = step.get("transformer", "?")
        label = mermaid_escape(f"{name} ({transformer})")
        lines.append(f"    {node}[{label}]")
        lines.append(f"    {prev} --> {node}")
        prev = node

    if modeling_config:
        label = mermaid_escape(str(modeling_config.get("type", "model")))
        lines.append(f"    model([{label}])")
        lines.append(f"    {prev} --> model")

    return "\n".join(lines)
