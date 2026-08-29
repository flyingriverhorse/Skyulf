"""Job-level Mermaid topology diagram.

Maps the executed node chain onto the core library's mermaid builder
(``skyulf.pipeline.diagram.build_mermaid_diagram``) so the backend and
``SkyulfPipeline.to_mermaid()`` share one diagram format. The result is
persisted on ``training_jobs.metrics["pipeline_diagram"]`` and rendered on
the Experiments page.
"""

import logging
from collections.abc import Mapping
from typing import Any

from backend.ml_pipeline._execution.schemas import NodeExecutionResult
from backend.ml_pipeline._execution.summary import _family_of
from skyulf.pipeline.diagram import build_mermaid_diagram, params_summary

logger = logging.getLogger(__name__)

_MODELING_FAMILIES = {"train", "tune"}
# The core diagram already starts at an "Input Data" node, so loaders would
# only duplicate it.
_HIDDEN_FAMILIES = {"loader"}


def build_pipeline_diagram(
    node_results: dict[str, NodeExecutionResult],
    model_type: str | None = None,
    node_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> str | None:
    """Render the executed node chain as a Mermaid flowchart.

    Nodes are walked in execution (topological) order: data loaders are
    dropped, the first training/tuning node becomes the model stage (labeled
    with ``model_type`` when known), and everything else becomes a
    preprocessing step. Labels use the human display name from node metadata
    (never the raw node id); the second label line is the node's runtime
    summary, falling back to a digest of its config params (``node_params``)
    when no summary exists. Returns ``None`` when nothing renderable ran or
    the build fails — the diagram is advisory and must never block a run.
    """
    try:
        steps: list[dict[str, Any]] = []
        modeling: dict[str, Any] = {}
        for result in node_results.values():
            step_type = result.step_type or "unknown"
            family = _family_of(step_type)
            if family in _HIDDEN_FAMILIES:
                continue
            metadata = result.metadata or {}
            summary = metadata.get("summary") or params_summary(
                (node_params or {}).get(result.node_id)
            )
            if family in _MODELING_FAMILIES and not modeling:
                modeling = {"type": model_type or step_type, "details": summary}
                continue
            display = metadata.get("display_name")
            entry: dict[str, Any]
            if display and display != step_type:
                entry = {"name": display, "transformer": step_type}
            else:
                entry = {"name": step_type}
            entry["details"] = summary
            steps.append(entry)
        if not steps and not modeling:
            return None
        return build_mermaid_diagram(steps, modeling)
    except Exception as exc:  # noqa: BLE001 - advisory artifact; never block a run
        logger.warning("pipeline diagram build failed: %s", exc)
        return None
