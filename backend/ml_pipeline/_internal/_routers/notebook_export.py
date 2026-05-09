"""Notebook export endpoint.

Generates a downloadable Jupyter notebook (`.ipynb`) from a saved pipeline
graph so data scientists can leave the canvas and continue iterating in
their preferred environment.

Two modes are supported:

* ``compact`` — minimal notebook that wires up `SkyulfPipeline.load(...)`
  and re-runs the same graph end-to-end. Best for "I want to predict on a
  new file" workflows.
* ``full`` — walks the saved graph node-by-node and renders one cell per
  preprocessing step (using `NodeRegistry.get_calculator/get_applier`) plus
  a final modeling cell. Best for teaching, debugging, or hand-tweaking a
  single step.

The endpoint hand-builds the notebook JSON. Avoiding the ``nbformat`` dep
keeps cold-start small and the schema (nbformat 4.5) is stable enough to
not warrant the extra runtime weight.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.engine import get_async_session
from backend.database.models import FeatureEngineeringPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])

ExportMode = Literal["full", "compact"]


# ---------------------------------------------------------------------------
# Notebook primitives. We keep them as plain dicts for full nbformat 4.5
# compatibility without pulling in `nbformat`.
# ---------------------------------------------------------------------------


def _md_cell(text: str) -> Dict[str, Any]:
    """Build a markdown cell. Splits into a list of lines per nbformat spec."""
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _wrap_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap a list of cells in a minimal nbformat 4.5 envelope."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Mode builders. Both consume the saved canvas graph
# (``{"nodes": [...], "edges": [...]}``) and emit a list of cells.
# ---------------------------------------------------------------------------


def _build_compact_notebook(dataset_id: str, graph: Dict[str, Any]) -> Dict[str, Any]:
    """Compact mode: load + run via `SkyulfPipeline`."""
    config_json = json.dumps(graph, indent=2)
    cells = [
        _md_cell(
            f"# Skyulf pipeline — `{dataset_id}` (compact)\n\n"
            "Auto-exported from the Skyulf canvas. This notebook runs the saved\n"
            "pipeline end-to-end. Edit the data path and target column below.\n"
        ),
        _code_cell("import pandas as pd\n" "from skyulf.pipeline import SkyulfPipeline\n"),
        _md_cell("## 1. Pipeline configuration\n"),
        _code_cell(f"pipeline_config = {config_json}\n"),
        _md_cell("## 2. Load your data\n"),
        _code_cell(
            "# Replace with your dataset path.\n"
            'df = pd.read_csv("data.csv")\n'
            'target_column = "<your_target>"\n'
        ),
        _md_cell("## 3. Fit + evaluate\n"),
        _code_cell(
            "pipeline = SkyulfPipeline(pipeline_config)\n"
            "metrics = pipeline.fit(df, target_column=target_column)\n"
            "metrics\n"
        ),
        _md_cell("## 4. Predict on new data\n"),
        _code_cell('# new_df = pd.read_csv("new_data.csv")\n' "# pipeline.predict(new_df)\n"),
    ]
    return _wrap_notebook(cells)


def _node_to_cell(node: Dict[str, Any], step_idx: int) -> Optional[Dict[str, Any]]:
    """Render a single canvas node as a code cell (full mode).

    Returns ``None`` for nodes that don't fit the Calculator/Applier shape
    (e.g. dataset source nodes that only declare a file path).
    """
    node_type = node.get("type") or node.get("data", {}).get("nodeType")
    if not node_type:
        return None
    config = node.get("data", {}).get("config", {}) or {}
    config_json = json.dumps(config, indent=2)
    var_suffix = f"_{step_idx}"
    src = (
        f"# Step {step_idx}: {node_type}\n"
        f"calc{var_suffix} = NodeRegistry.get_calculator({node_type!r})()\n"
        f"applier{var_suffix} = NodeRegistry.get_applier({node_type!r})()\n"
        f"config{var_suffix} = {config_json}\n"
        f"params{var_suffix} = calc{var_suffix}.fit(df, config{var_suffix})\n"
        f"df = applier{var_suffix}.apply(df, params{var_suffix})\n"
    )
    return _code_cell(src)


def _build_full_notebook(dataset_id: str, graph: Dict[str, Any]) -> Dict[str, Any]:
    """Full mode: explicit per-node Calculator/Applier cells."""
    nodes = graph.get("nodes") or []
    cells: List[Dict[str, Any]] = [
        _md_cell(
            f"# Skyulf pipeline — `{dataset_id}` (full)\n\n"
            "Each preprocessing node is rendered as its own cell so you can\n"
            "tweak parameters, swap nodes, or inspect intermediate state.\n"
        ),
        _code_cell(
            "import pandas as pd\n"
            "from skyulf.registry import NodeRegistry\n"
            "import skyulf  # noqa: F401  # populates the registry\n"
        ),
        _md_cell("## Load your data\n"),
        _code_cell('df = pd.read_csv("data.csv")\n'),
        _md_cell("## Apply each step\n"),
    ]
    rendered = 0
    for idx, node in enumerate(nodes, start=1):
        cell = _node_to_cell(node, idx)
        if cell is not None:
            cells.append(cell)
            rendered += 1
    if rendered == 0:
        cells.append(
            _md_cell(
                "_No preprocessing nodes were found in this graph. The pipeline "
                "may consist solely of dataset/sink nodes._\n"
            )
        )
    cells.append(_md_cell("## Result\n"))
    cells.append(_code_cell("df.head()\n"))
    return _wrap_notebook(cells)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


async def _load_graph(dataset_id: str, session: AsyncSession) -> Dict[str, Any]:
    """Fetch the active pipeline graph for ``dataset_id``.

    Mirrors the DB lookup in ``pipelines_io.load_pipeline``. We deliberately
    don't fall back to JSON-on-disk here — the export endpoint is a
    convenience and the JSON storage backend is dev-only.
    """
    stmt = select(FeatureEngineeringPipeline).where(
        FeatureEngineeringPipeline.dataset_source_id == dataset_id,
        FeatureEngineeringPipeline.is_active,
    )
    result = await session.execute(stmt)
    pipeline = result.scalar_one_or_none()
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    graph = getattr(pipeline, "graph", None)
    if not isinstance(graph, dict):
        raise HTTPException(status_code=400, detail="Pipeline has no graph")
    return graph


@router.get("/pipeline/{dataset_id}/export-notebook")
async def export_pipeline_notebook(
    dataset_id: str,
    mode: ExportMode = Query("compact", description="`compact` or `full`."),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Return a `.ipynb` file representing the saved pipeline."""
    graph = await _load_graph(dataset_id, session)
    builder = _build_full_notebook if mode == "full" else _build_compact_notebook
    notebook = builder(dataset_id, graph)
    body = json.dumps(notebook, indent=1).encode("utf-8")
    filename = f"skyulf_pipeline_{dataset_id}_{mode}.ipynb"
    return Response(
        content=body,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
