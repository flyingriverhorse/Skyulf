"""Multi-branch notebook builders.

Emits one section per terminal training node so canvases with N independent
training paths produce clearly separated, self-contained notebook sections
rather than a single ambiguous block. Imported by ``notebook_export`` when
``_terminal_models()`` detects more than one model at the graph leaves.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import _notebook_builders as nb
from ._notebook_builders import (
    _NodeIn,
    _PipelineIn,
    _to_py_literal,
    build_skyulf_config,
    code_cell,
    compact_load_cells,
    config_fingerprint,
    full_intro_cells,
    md_cell,
    wrap_notebook,
)

# (loader, preprocess, feat_target, train_test, model)
_Classified = tuple[
    _NodeIn | None,
    list[_NodeIn],
    _NodeIn | None,
    _NodeIn | None,
    _NodeIn | None,
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _branch_letter(idx: int) -> str:
    """0→A, 1→B, … 25→Z, 26→AA (recursive)."""
    if idx < 26:
        return chr(ord("A") + idx)
    return _branch_letter(idx // 26 - 1) + _branch_letter(idx % 26)


# ---------------------------------------------------------------------------
# Full-mode branched builders
# ---------------------------------------------------------------------------

_SPLIT_OR_MODEL = {
    "data_loader",
    "feature_target_split",
    "TrainTestSplitter",
    "training",
}


def _branch_sections_md(
    cfg: _PipelineIn,
    dataset_id: str,
    dataset_name: str | None,
    terminals: list[_NodeIn],
) -> str:
    items = "\n".join(
        f"- **Branch {_branch_letter(i)}** — `{t.step_type}` (`{nb._model_algorithm(t)}`)"
        for i, t in enumerate(terminals)
    )
    return (
        f"# Skyulf pipeline — `{dataset_name or dataset_id}` (multi-branch)\n\n"
        f"_Config fingerprint: `{config_fingerprint(cfg)}`._\n\n"
        f"This pipeline has **{len(terminals)} training branches**. "
        f"Each branch is trained, evaluated, and persisted independently.\n\n"
        f"**Branches:**\n{items}\n\n"
        f"Each branch fits its complete preprocessing chain on its own training "
        f"partition, including nodes shared on the canvas.\n"
    )


def _branch_topology_md(letter: str, branch_nodes: list[_NodeIn]) -> dict[str, Any]:
    lines = [f"<details><summary>Branch {letter} nodes (topological order)</summary>\n", "```"]
    for i, n in enumerate(branch_nodes, start=1):
        lines.append(f"  {i:>2}. {n.step_type}")
    lines.extend(["```", "</details>\n"])
    return md_cell("\n".join(lines))


@dataclass
class _FullBranchCtx:
    """Groups the parameters for `build_full_branched` to stay under param-count limit."""

    cfg: _PipelineIn
    dataset_id: str
    dataset_name: str | None
    all_nodes: list[_NodeIn]
    terminals: list[_NodeIn]
    ancestors_in_topo: Callable[[str, list[_NodeIn]], list[_NodeIn]]
    classify: Callable[[list[_NodeIn]], _Classified]
    data_path_resolver: Callable[[_NodeIn | None], str]
    resolved_from_db: bool


def build_full_branched(ctx: _FullBranchCtx) -> dict[str, Any]:
    """Full notebook: an independent preprocessing and training pipeline per branch.

    Accepts a :class:`_FullBranchCtx` dataclass instead of many positional
    arguments to stay under the Codacy parameter-count limit.
    """
    branches, classifications = _collect_branches(
        ctx.all_nodes, ctx.terminals, ctx.ancestors_in_topo, ctx.classify
    )
    loader = next((n for n in ctx.all_nodes if n.step_type == "data_loader"), None)
    data_path = ctx.data_path_resolver(loader)
    cells: list[dict[str, Any]] = [
        md_cell(_branch_sections_md(ctx.cfg, ctx.dataset_id, ctx.dataset_name, ctx.terminals)),
    ]
    cells.extend(full_intro_cells(data_path, resolved_from_db=ctx.resolved_from_db))
    for index, (branch_nodes, classified) in enumerate(zip(branches, classifications, strict=True)):
        letter = _branch_letter(index)
        _loader, preprocess, feat_target, train_test, model = classified
        if model is None:
            continue
        cells.append(md_cell(f"## Branch {letter}\n"))
        cells.extend(nb.full_training_cells(preprocess, feat_target, train_test, model, letter))
        cells.append(code_cell(f"metrics_{letter} = pipeline_{letter}_metrics\n"))
        cells.append(_branch_topology_md(letter, branch_nodes))
    letters = [_branch_letter(i) for i in range(len(ctx.terminals))]
    n_branches = len(ctx.terminals)
    cells.extend(_metrics_comparison_cell(letters, 5 + n_branches))
    return wrap_notebook(cells)


def _metrics_helper_cell() -> dict[str, Any]:
    """Deprecated alias kept for backwards compatibility; delegates to builders."""
    return nb.metrics_helper_cell()


def _branch_comparison_code(metrics_dict_literal: str) -> str:
    return (
        "import pandas as pd\n"
        "from IPython.display import display\n\n"
        "_branch_metrics = {" + metrics_dict_literal + "}\n"
        "try:\n"
        "    _frames = []\n"
        "    for _label, _m in _branch_metrics.items():\n"
        "        _sub = _summarize_metrics(_m)\n"
        "        if _sub.empty:\n"
        "            continue\n"
        "        _sub.index = pd.MultiIndex.from_product(\n"
        "            [[_label], _sub.index], names=['branch', 'split']\n"
        "        )\n"
        "        _frames.append(_sub)\n"
        "    if _frames:\n"
        "        _df = pd.concat(_frames, axis=0)\n"
        "        styled = (\n"
        "            _df.style\n"
        "            .format('{:.4f}', na_rep='-')\n"
        "            .background_gradient(cmap='RdYlGn', axis=0)\n"
        "            .set_caption('Branch metrics comparison (rows = branch / split)')\n"
        "        )\n"
        "        display(styled)\n"
        "    else:\n"
        "        print('No numeric metrics found in any branch.')\n"
        "        for _label, _m in _branch_metrics.items():\n"
        "            print(f'  {_label}: {_m!r}')\n"
        "except Exception as _e:\n"
        "    print(f'Failed to render comparison table: {_e}')\n"
        "    for _label, _m in _branch_metrics.items():\n"
        "        print(f'  {_label}: {_m!r}')\n"
    )


def _metrics_comparison_cell(letters: list[str], section_no: int) -> list[dict[str, Any]]:
    """Side-by-side metrics table for all trained branches (full mode)."""
    metrics_vars = ", ".join(f'"Branch {l}": metrics_{l}' for l in letters)  # noqa: E741
    return [
        md_cell(
            f"## {section_no}. Metrics comparison\n\n"
            "Runs **after all branches have trained**. Each branch's evaluation\n"
            "report is flattened to its scalar metrics (per train/test split) and\n"
            "displayed side-by-side so you can pick the best model.\n"
        ),
        code_cell(_branch_comparison_code(metrics_vars)),
    ]


# ---------------------------------------------------------------------------
# Compact-mode branched builders
# ---------------------------------------------------------------------------


def _compact_branch_summary_md(
    cfg: _PipelineIn,
    dataset_id: str,
    dataset_name: str | None,
    terminals: list[_NodeIn],
) -> str:
    items = "\n".join(
        f"- **Branch {_branch_letter(i)}** — `{nb._model_algorithm(t)}`"
        for i, t in enumerate(terminals)
    )
    return (
        f"# Skyulf pipeline — `{dataset_name or dataset_id}` (compact, multi-branch)\n\n"
        f"_Config fingerprint: `{config_fingerprint(cfg)}`._\n\n"
        f"This pipeline has **{len(terminals)} training branches**. Each branch builds "
        f"its own `SkyulfPipeline`, fits on the same training frame, and saves "
        f"its artifact independently — deploy / score each model separately.\n\n"
        f"**Branches:**\n{items}\n"
    )


def _compact_branch_cells(
    letter: str, section_no: int, classified: _Classified
) -> list[dict[str, Any]]:
    """Fit + persist cells for one compact-mode branch."""
    _loader, preprocess, feat_target, train_test, model = classified
    target_col = (
        (model.params.get("target_column") if model else None)
        or (feat_target.params.get("target_column") if feat_target else None)
        or (train_test.params.get("target_column") if train_test else None)
        or "<target_column>"
    )
    full_chain = nb.training_chain(preprocess, feat_target, train_test)
    cfg_dict = build_skyulf_config(full_chain, model)
    config_json = _to_py_literal(cfg_dict)
    var = f"pipeline_{letter}"
    algo = nb._model_algorithm(model) if model is not None else "none"
    step_names = ", ".join(n.step_type for n in preprocess) or "none"
    target_note = f" · target: `{target_col}`" if target_col else ""
    return [
        md_cell(
            f"## {section_no}. Branch {letter} — `{algo}`{target_note}\n\n"
            f"Preprocessing: {step_names}.\n"
            f"Builds a `SkyulfPipeline`, fits it on the full frame, and saves to "
            f"`skyulf_pipeline_{letter}.pkl`.\n"
        ),
        code_cell(f"{var}_config = {config_json}\n"),
        code_cell(
            f"BRANCH_{letter}_TARGET = {target_col!r}  # target column for this branch\n"
            f"{var} = SkyulfPipeline({var}_config)\n"
            f"{var}_metrics = {var}.fit(df, target_column=BRANCH_{letter}_TARGET)\n"
            f"_summarize_metrics({var}_metrics).style.format('{{:.4f}}', na_rep='-')"
            f".background_gradient(cmap='RdYlGn', axis=0)"
            f".set_caption('Branch {letter} \u2014 train vs test')\n"
        ),
        code_cell(
            f'{var}.save("skyulf_pipeline_{letter}.pkl")\n'
            f'# loaded_{letter} = SkyulfPipeline.load("skyulf_pipeline_{letter}.pkl")\n'
            f"# loaded_{letter}.predict(new_df)\n"
        ),
    ]


def _inference_snippet(letters: list[str]) -> dict[str, Any]:
    return code_cell(
        "# Choose the branch with the best metrics and load its artifact:\n"
        "# new_df = pd.read_csv('new_data.csv')\n"
        + "".join(
            f'# pred_{l} = SkyulfPipeline.load("skyulf_pipeline_{l}.pkl").predict(new_df)\n'
            for l in letters  # noqa: E741
        )
    )


@dataclass
class _CompactBranchCtx:
    """Groups the parameters for `build_compact_branched` to stay under param-count limit."""

    cfg: _PipelineIn
    dataset_id: str
    dataset_name: str | None
    all_nodes: list[_NodeIn]
    terminals: list[_NodeIn]
    ancestors_in_topo: Callable[[str, list[_NodeIn]], list[_NodeIn]]
    classify: Callable[[list[_NodeIn]], _Classified]
    data_path_resolver: Callable[[_NodeIn | None], str]
    target_resolver: Callable[[_NodeIn | None, _NodeIn | None], str | None]
    resolved_from_db: bool


def _collect_branches(
    all_nodes: list[_NodeIn],
    terminals: list[_NodeIn],
    ancestors_in_topo: Callable[[str, list[_NodeIn]], list[_NodeIn]],
    classify: Callable[[list[_NodeIn]], _Classified],
) -> tuple[list[list[_NodeIn]], list[_Classified]]:
    """Build per-branch node lists and classifications in one pass."""
    branches: list[list[_NodeIn]] = []
    classifications: list[_Classified] = []
    for t in terminals:
        anc = ancestors_in_topo(t.node_id, all_nodes)
        nodes = anc + [t]
        branches.append(nodes)
        classifications.append(classify(nodes))
    return branches, classifications


def _compact_compare_predict_cells(letters: list[str], base_section: int) -> list[dict[str, Any]]:
    """Metrics comparison + predict-on-new-data tail for compact multi-branch notebooks."""
    compare_vars = ", ".join(f'"Branch {l}": pipeline_{l}_metrics' for l in letters)  # noqa: E741
    cells: list[dict[str, Any]] = [
        md_cell(
            f"## {base_section}. Metrics comparison\n\n"
            "All branches have now been trained. Each branch's metrics are\n"
            "flattened to scalar values per split (train/test) and displayed\n"
            "side-by-side before choosing which artifact to deploy.\n"
        ),
        code_cell(_branch_comparison_code(compare_vars)),
        md_cell(
            f"## {base_section + 1}. Predict on new data (per branch)\n\n"
            "Load the artifact for the branch with the best metrics and score new rows.\n"
        ),
        _inference_snippet(letters),
    ]
    return cells


def build_compact_branched(ctx: _CompactBranchCtx) -> dict[str, Any]:
    """Compact notebook: one `SkyulfPipeline` per terminal model.

    Accepts a :class:`_CompactBranchCtx` dataclass instead of many positional
    arguments to stay under the Codacy parameter-count limit.
    """
    _branches, classifications = _collect_branches(
        ctx.all_nodes, ctx.terminals, ctx.ancestors_in_topo, ctx.classify
    )
    loader = next((n for n in ctx.all_nodes if n.step_type == "data_loader"), None)
    data_path = ctx.data_path_resolver(loader)
    target_col: str | None = None
    for c in classifications:
        target_col = (c[4].params.get("target_column") if c[4] else None) or ctx.target_resolver(
            c[2], c[3]
        )
        if target_col:
            break
    target_col = target_col or "<target_column>"
    letters = [_branch_letter(i) for i in range(len(ctx.terminals))]
    cells: list[dict[str, Any]] = [
        md_cell(
            _compact_branch_summary_md(ctx.cfg, ctx.dataset_id, ctx.dataset_name, ctx.terminals)
        ),
    ]
    cells.extend(compact_load_cells(data_path, target_col, resolved_from_db=ctx.resolved_from_db))
    for i, classified in enumerate(classifications):
        cells.extend(_compact_branch_cells(letters[i], 3 + i, classified))
    cells.extend(_compact_compare_predict_cells(letters, 3 + len(ctx.terminals)))
    return wrap_notebook(cells)


__all__ = ["build_full_branched", "build_compact_branched", "_CompactBranchCtx", "_FullBranchCtx"]
