"""Notebook export endpoint.

Generates a downloadable Jupyter notebook (`.ipynb`) from a saved pipeline
graph so data scientists can leave the canvas and continue iterating in
their preferred environment, and so MLOps users have a reproducible
artifact for production handoff.

The notebook is built from the **canonical** `PipelineConfig` shape that
the execution engine consumes (`{nodes: [{node_id, step_type, params,
inputs}], metadata}`). The frontend already produces this shape via
`pipelineConverter.ts`; we accept it as a POST body so we don't duplicate
that converter on the backend.

Two modes are supported:

* ``compact`` — minimal notebook that wires up `SkyulfPipeline(config)`
  with a properly split `{preprocessing, modeling}` block, fits, evaluates,
  predicts on new data, and shows a save/load + FastAPI handoff snippet.
* ``full`` — explains each preprocessing configuration, then trains through
  `SkyulfPipeline` with partitioning before learned transforms. Fitted artifacts
  remain inspectable; save/load includes the model and raw-row inference.

Cell builders live in `_notebook_builders` to keep this module focused on
graph classification and the HTTP endpoint.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.engine import get_async_session
from backend.database.models import DataSource, FeatureEngineeringPipeline
from backend.utils.file_utils import extract_file_path_from_source

from . import _notebook_builders as nb
from ._notebook_branched import (
    _CompactBranchCtx,
    _FullBranchCtx,
    build_compact_branched,
    build_full_branched,
)
from ._notebook_builders import _NodeIn, _PipelineIn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])

ExportMode = Literal["full", "compact"]

# Step types that are NOT preprocessing transformations. They get rendered
# as dedicated sections (data load, splits, modeling).
_DATA_LOADER_STEPS = {"data_loader"}
_MODELING_STEPS = {"training"}
_RESAMPLER_STEPS = {"Oversampling", "Undersampling"}
# Visual / preview-only nodes that have no runtime effect.
_PREVIEW_STEPS = {"data_preview", "Unknown"}


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _build_adjacency(
    nodes: list[_NodeIn],
) -> tuple[dict[str, _NodeIn], dict[str, int], dict[str, list[str]]]:
    by_id = {n.node_id: n for n in nodes}
    indeg: dict[str, int] = dict.fromkeys(by_id, 0)
    children: dict[str, list[str]] = {nid: [] for nid in by_id}
    for n in nodes:
        for src in n.inputs:
            if src in by_id:
                indeg[n.node_id] += 1
                children[src].append(n.node_id)
    return by_id, indeg, children


def _kahn_walk(
    by_id: dict[str, _NodeIn], indeg: dict[str, int], children: dict[str, list[str]]
) -> list[str]:
    queue = [nid for nid, deg in indeg.items() if deg == 0]
    ordered: list[str] = []
    while queue:
        cur = queue.pop(0)
        ordered.append(cur)
        for child in children[cur]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    seen = set(ordered)
    ordered.extend(nid for nid in by_id if nid not in seen)  # cycle fallback
    return ordered


def _topo_sort(nodes: list[_NodeIn]) -> list[_NodeIn]:
    """Kahn's algorithm. Stable: ties resolved by original list order."""
    by_id, indeg, children = _build_adjacency(nodes)
    return [by_id[i] for i in _kahn_walk(by_id, indeg, children)]


def _classify(
    nodes: list[_NodeIn],
) -> tuple[
    _NodeIn | None,
    list[_NodeIn],
    _NodeIn | None,
    _NodeIn | None,
    _NodeIn | None,
]:
    """Bucket the topologically-sorted nodes into the sections of a notebook.

    Returns ``(loader, preprocess, feature_target_split, train_test_split, model)``.
    """
    loader: _NodeIn | None = None
    feat_target: _NodeIn | None = None
    train_test: _NodeIn | None = None
    model: _NodeIn | None = None
    preprocess: list[_NodeIn] = []
    for n in nodes:
        st = n.step_type
        if st in _DATA_LOADER_STEPS:
            loader = loader or n
        elif st == "feature_target_split":
            feat_target = feat_target or n
        elif st == "TrainTestSplitter":
            train_test = train_test or n
        elif st in _MODELING_STEPS:
            model = model or n
        elif st in _RESAMPLER_STEPS or st not in _PREVIEW_STEPS:
            preprocess.append(n)
    return loader, preprocess, feat_target, train_test, model


# ---------------------------------------------------------------------------
# Branch detection (per-model-terminal slicing)
# ---------------------------------------------------------------------------


def _terminal_models(nodes: list[_NodeIn]) -> list[_NodeIn]:
    """Models with no *runtime* descendants — each defines an independent training branch.

    A canvas with two trainers fed by separate splits = two branches; the
    notebook renders one section per branch so the user can fit, evaluate,
    and persist each model independently.

    Visual-only nodes (`data_preview`, `Unknown`) hanging off a model do NOT
    disqualify it as a terminal — they have no runtime effect on the model's
    output. Without this filter, dropping a Preview node onto a trainer would
    silently strip that whole branch from the exported notebook.
    """
    by_id, _, children = _build_adjacency(nodes)
    out: list[_NodeIn] = []
    for n in _topo_sort(nodes):
        if n.step_type not in _MODELING_STEPS:
            continue
        runtime_children = [
            cid
            for cid in children.get(n.node_id, [])
            if cid in by_id and by_id[cid].step_type not in _PREVIEW_STEPS
        ]
        if not runtime_children:
            out.append(by_id[n.node_id])
    return out


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _expand_parallel_terminals(nodes: list[_NodeIn]) -> list[_NodeIn]:
    """Split multi-input parallel trainers into one virtual terminal per input.

    Mirrors the runtime engine rule in
    :func:`backend.ml_pipeline._execution.graph_utils.partition_parallel_pipeline`:
    a trainer with ``execution_mode == "parallel"`` and several incoming
    edges trains a separate model per input. The notebook must therefore
    render one branch section per produced model — not one per node.

    The original multi-input node is replaced by N virtual ``_NodeIn`` copies,
    each with ``inputs=[single_root]`` and a unique ``node_id`` suffix, so the
    existing per-branch ancestor walk works unchanged.
    """
    expanded: list[_NodeIn] = []
    for n in nodes:
        unique_inputs = _dedupe_preserve_order(list(n.inputs))
        is_parallel = (
            n.step_type in _MODELING_STEPS
            and n.params.get("execution_mode") == "parallel"
            and len(unique_inputs) > 1
        )
        if not is_parallel:
            expanded.append(n)
            continue
        # Strip routing key (the engine consumes it; estimator must not see it).
        clean_params = {k: v for k, v in n.params.items() if k != "execution_mode"}
        for i, src in enumerate(unique_inputs):
            expanded.append(
                _NodeIn(
                    node_id=f"{n.node_id}__path{i}",
                    step_type=n.step_type,
                    params=clean_params,
                    inputs=[src],
                )
            )
    return expanded


def _ancestors_in_topo(target_id: str, all_nodes: list[_NodeIn]) -> list[_NodeIn]:
    """All ancestors of ``target_id`` (excluding self) in topological order."""
    by_id, _, _ = _build_adjacency(all_nodes)
    seen = {target_id}
    stack = [target_id]
    while stack:
        cur = stack.pop()
        node = by_id.get(cur)
        if node is None:
            continue
        for src in node.inputs:
            if src not in seen and src in by_id:
                seen.add(src)
                stack.append(src)
    seen.discard(target_id)
    return [n for n in _topo_sort(all_nodes) if n.node_id in seen]


def _resolve_dataset_path(
    loader: _NodeIn | None,
    dataset_name: str | None,
    db_file_path: str | None = None,
) -> str:
    """Pick the most useful CSV path for the notebook's `pd.read_csv(...)`.

    Priority: explicit loader path > on-disk path looked up from `data_sources` >
    loader's `file_path` > dataset display name > `<dataset_id>.csv` fallback.
    """
    if loader is None:
        return db_file_path or "data.csv"
    p = loader.params
    return str(
        p.get("path")
        or db_file_path
        or p.get("file_path")
        or dataset_name
        or f"{p.get('dataset_id', 'data')}.csv"
    )


def _resolve_target_column(
    feat_target: _NodeIn | None, train_test: _NodeIn | None, model: _NodeIn | None = None
) -> str | None:
    for src in (model, feat_target, train_test):
        if src and isinstance(src.params.get("target_column"), str):
            return src.params["target_column"]
    return None


def _validate_export_ids(nodes: list[_NodeIn]) -> None:
    """Reject duplicate IDs and references to missing input nodes."""
    ids = {node.node_id for node in nodes}
    if len(ids) != len(nodes):
        raise HTTPException(400, "Notebook export requires unique node IDs")
    if any(source not in ids for node in nodes for source in node.inputs):
        raise HTTPException(400, "Notebook export has a missing input node")


def _validate_export_order(ordered: list[_NodeIn]) -> None:
    """Reject edges that contradict the topological order."""
    positions = {node.node_id: index for index, node in enumerate(ordered)}
    if any(
        positions[source] >= positions[node.node_id] for node in ordered for source in node.inputs
    ):
        raise HTTPException(400, "Notebook export does not support cycles")


def _export_ancestor_ids(nodes: list[_NodeIn], terminals: list[_NodeIn]) -> set[str]:
    """Collect terminal models and their ancestors for notebook selection."""
    selected = {
        node.node_id
        for terminal in terminals
        for node in _ancestors_in_topo(terminal.node_id, nodes)
    }
    selected.update(terminal.node_id for terminal in terminals)
    return selected


def _select_export_ancestry(nodes: list[_NodeIn], terminals: list[_NodeIn]) -> list[_NodeIn]:
    """Keep every terminal model's ancestry without omitting training branches."""
    if terminals:
        selected = _export_ancestor_ids(nodes, terminals)
        if any(
            node.step_type in _MODELING_STEPS and node.node_id not in selected for node in nodes
        ):
            raise HTTPException(
                400, "Notebook export cannot omit a training branch with post-model nodes"
            )
        nodes = [node for node in nodes if node.node_id in selected]
    elif any(node.step_type in _MODELING_STEPS for node in nodes):
        raise HTTPException(400, "Notebook export requires training nodes to end their branches")
    return nodes


def _validate_export_runtime(runtime: list[_NodeIn]) -> None:
    """Require one dataset and reject merged runtime inputs."""
    if sum(node.step_type in _DATA_LOADER_STEPS for node in runtime) > 1:
        raise HTTPException(
            400, "Notebook export supports one data loader; export each dataset separately"
        )
    if any(len(set(node.inputs)) > 1 for node in runtime):
        raise HTTPException(
            400,
            "Notebook export does not support merged inputs; use independent parallel training branches",
        )


def _validate_export_branches(
    nodes: list[_NodeIn], terminals: list[_NodeIn], runtime: list[_NodeIn]
) -> None:
    """Reject repeated structural steps within each exported model branch."""
    branches = [
        _ancestors_in_topo(terminal.node_id, nodes) + [terminal] for terminal in terminals
    ] or [runtime]
    for branch in branches:
        for step_type in ("training", "feature_target_split", "TrainTestSplitter"):
            if sum(node.step_type == step_type for node in branch) > 1:
                raise HTTPException(
                    400, f"Notebook export supports one {step_type} per model branch"
                )


def _validate_preprocessing_chain(runtime: list[_NodeIn]) -> None:
    """Require preprocessing-only exports to form one connected output chain."""
    if sum(not node.inputs for node in runtime) > 1:
        raise HTTPException(400, "Notebook export requires a connected preprocessing chain")
    for node in runtime:
        if sum(node.node_id in child.inputs for child in runtime) > 1:
            raise HTTPException(400, "Notebook export requires one preprocessing output chain")


def _export_nodes(cfg: _PipelineIn) -> list[_NodeIn]:
    """Select model ancestry and reject graphs the sequential notebook cannot represent."""
    _validate_export_ids(cfg.nodes)
    ordered = _topo_sort(cfg.nodes)
    _validate_export_order(ordered)
    nodes = _expand_parallel_terminals(ordered)
    terminals = _terminal_models(nodes)
    nodes = _select_export_ancestry(nodes, terminals)
    runtime = [node for node in nodes if node.step_type not in _PREVIEW_STEPS]
    _validate_export_runtime(runtime)
    _validate_export_branches(nodes, terminals, runtime)
    if not terminals:
        _validate_preprocessing_chain(runtime)
    return nodes


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_compact_notebook(
    cfg: _PipelineIn,
    dataset_id: str,
    dataset_name: str | None,
    db_file_path: str | None = None,
) -> dict[str, Any]:
    nodes = _export_nodes(cfg)
    terminals = _terminal_models(nodes)
    if len(terminals) > 1:
        return build_compact_branched(
            _CompactBranchCtx(
                cfg=cfg,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                all_nodes=nodes,
                terminals=terminals,
                ancestors_in_topo=_ancestors_in_topo,
                classify=_classify,
                data_path_resolver=lambda loader: _resolve_dataset_path(
                    loader, dataset_name, db_file_path
                ),
                target_resolver=_resolve_target_column,
                resolved_from_db=db_file_path is not None,
            )
        )
    loader, preprocess, feat_target, train_test, model = _classify(nodes)
    if model is None:
        return _build_full_notebook(cfg, dataset_id, dataset_name, db_file_path)
    # Compact mode hands the whole feature-engineering chain (including
    # splitters) to SkyulfPipeline; FeatureEngineer skips splitters during
    # transform, so this is safe.
    full_chain = (
        nb.training_chain(preprocess, feat_target, train_test)
        if model
        else [n for n in (feat_target, train_test) if n is not None] + preprocess
    )
    skyulf_cfg = nb.build_skyulf_config(full_chain, model)
    data_path = _resolve_dataset_path(loader, dataset_name, db_file_path)
    target_col = _resolve_target_column(feat_target, train_test, model) or "<target_column>"
    config_json = nb._to_py_literal(skyulf_cfg)
    cells: list[dict[str, Any]] = [
        nb.md_cell(
            nb.compact_summary_md(
                cfg, dataset_id, dataset_name, preprocess, feat_target, train_test, model
            )
        ),
        nb.md_cell(nb.pipeline_diagram_md(full_chain, model)),
    ]
    cells.extend(
        nb.compact_load_cells(data_path, target_col, resolved_from_db=db_file_path is not None)
    )
    cells.extend(nb.compact_run_cells(config_json))
    cells.extend(nb.compact_persist_cells())
    return nb.wrap_notebook(cells)


def _build_full_notebook(
    cfg: _PipelineIn,
    dataset_id: str,
    dataset_name: str | None,
    db_file_path: str | None = None,
) -> dict[str, Any]:
    nodes = _export_nodes(cfg)
    terminals = _terminal_models(nodes)
    if len(terminals) > 1:
        return build_full_branched(
            _FullBranchCtx(
                cfg=cfg,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                all_nodes=nodes,
                terminals=terminals,
                ancestors_in_topo=_ancestors_in_topo,
                classify=_classify,
                data_path_resolver=lambda loader: _resolve_dataset_path(
                    loader, dataset_name, db_file_path
                ),
                resolved_from_db=db_file_path is not None,
            )
        )
    loader, preprocess, feat_target, train_test, model = _classify(nodes)
    data_path = _resolve_dataset_path(loader, dataset_name, db_file_path)
    diagram_chain = [n for n in (feat_target, train_test) if n is not None] + preprocess
    cells: list[dict[str, Any]] = [
        nb.md_cell(
            nb.full_summary_md(
                cfg, dataset_id, dataset_name, nodes, preprocess, feat_target, train_test, model
            )
        ),
        nb.md_cell(nb.pipeline_diagram_md(diagram_chain, model)),
    ]
    cells.extend(nb.full_intro_cells(data_path, resolved_from_db=db_file_path is not None))
    if model is not None:
        cells.extend(nb.full_training_cells(preprocess, feat_target, train_test, model))
        return nb.wrap_notebook(cells)
    if not preprocess:
        cells.append(nb.md_cell("_No preprocessing transformations in this pipeline._\n"))
    cells.append(
        nb.md_cell(
            "### Preprocessing-only export\n\n"
            "There is no model in this graph. Feature/target and train/test split nodes "
            "are omitted; transformations fit all supplied rows. Use a training node "
            "for held-out evaluation and model predictions.\n"
        )
    )
    for i, n in enumerate(preprocess, start=1):
        cells.append(nb.node_to_cell(n, i))
    cells.extend(nb.persist_cells(preprocess))
    return nb.wrap_notebook(cells)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


async def _lookup_dataset_name(dataset_id: str, session: AsyncSession) -> str | None:
    """Best-effort dataset display name; failure is non-fatal."""
    stmt = select(FeatureEngineeringPipeline.name).where(
        FeatureEngineeringPipeline.dataset_source_id == dataset_id,
        FeatureEngineeringPipeline.is_active,
    )
    try:
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        return str(row) if row else None
    except Exception:  # noqa: BLE001 - non-fatal lookup; None fallback  # pragma: no cover — non-critical lookup
        return None


async def _lookup_dataset_file_path(dataset_id: str, session: AsyncSession) -> str | None:
    """Resolve a Canvas numeric ID or source UUID to its on-disk file path.

    Returned as a forward-slash POSIX path so notebooks open cleanly on any OS.
    Failure is non-fatal — the caller falls back to the loader node's params.
    """
    try:
        # Match DataIngestionService.get_source: numeric strings identify the PK.
        source_filter = (
            DataSource.id == int(dataset_id)
            if dataset_id.isdigit()
            else DataSource.source_id == dataset_id
        )
        stmt = select(DataSource).where(source_filter)
        result = await session.execute(stmt)
        ds = result.scalar_one_or_none()
        if ds is None:
            return None
        source_dict: dict[str, Any] = {
            "file_path": None,
            "connection_info": ds.config if isinstance(ds.config, dict) else {},
            "config": ds.config if isinstance(ds.config, dict) else {},
        }
        path = extract_file_path_from_source(source_dict)
        if path is None:
            return None
        return path.as_posix() if isinstance(path, Path) else str(path)
    except Exception:  # noqa: BLE001 - non-fatal lookup; None fallback  # pragma: no cover — non-critical lookup
        return None


@router.post("/pipeline/{dataset_id}/export-notebook")
async def export_pipeline_notebook(
    dataset_id: str,
    config: _PipelineIn = Body(..., description="Converted pipeline config (engine shape)."),
    mode: ExportMode = Query("full", description="`compact` or `full`."),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Return a `.ipynb` file representing the pipeline.

    The body must be the converted pipeline config (the same shape the
    execution engine consumes). The frontend produces this via
    `pipelineConverter.ts`; sending it directly avoids duplicating that
    converter on the backend.
    """
    if not config.nodes:
        raise HTTPException(status_code=400, detail="Pipeline has no nodes")
    try:
        dataset_name = await _lookup_dataset_name(dataset_id, session)
        db_file_path = await _lookup_dataset_file_path(dataset_id, session)
        builder = _build_full_notebook if mode == "full" else _build_compact_notebook
        notebook = builder(config, dataset_id, dataset_name, db_file_path)
        body = json.dumps(notebook, indent=1).encode("utf-8")
        # Sanitise dataset_id before embedding it in the Content-Disposition
        # header — raw user input could inject header fields via `"` or CRLF.
        safe_id = re.sub(r"[^\w.\-]", "_", dataset_id)
        filename = f"skyulf_pipeline_{safe_id}_{mode}.ipynb"
        return Response(
            content=body,
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        logger.exception("Failed to export notebook for dataset %s", dataset_id)
        raise HTTPException(status_code=500, detail="Failed to generate notebook") from None
