"""Multi-branch notebook builders.

Emits one section per terminal training node so canvases with N independent
training paths produce clearly separated, self-contained notebook sections
rather than a single ambiguous block. Imported by ``notebook_export`` when
``_terminal_models()`` detects more than one model at the graph leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    modeling_cells,
    node_to_cell,
    split_cells,
    wrap_notebook,
)

# (loader, preprocess, feat_target, train_test, model)
_Classified = Tuple[
    Optional[_NodeIn],
    List[_NodeIn],
    Optional[_NodeIn],
    Optional[_NodeIn],
    Optional[_NodeIn],
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _branch_letter(idx: int) -> str:
    """0→A, 1→B, … 25→Z, 26→AA (recursive)."""
    if idx < 26:
        return chr(ord("A") + idx)
    return _branch_letter(idx // 26 - 1) + _branch_letter(idx % 26)


def _shared_preprocess_ids(branches: List[List[_NodeIn]]) -> set:
    """Node IDs that appear in every branch — fitted once before per-branch sections."""
    if not branches:
        return set()
    sets = [set(n.node_id for n in b) for b in branches]
    return set.intersection(*sets)


# ---------------------------------------------------------------------------
# Full-mode branched builders
# ---------------------------------------------------------------------------

_SPLIT_OR_MODEL = {
    "data_loader",
    "feature_target_split",
    "TrainTestSplitter",
    "basic_training",
    "advanced_tuning",
}


def _branch_sections_md(
    cfg: _PipelineIn,
    dataset_id: str,
    dataset_name: Optional[str],
    terminals: List[_NodeIn],
) -> str:
    items = "\n".join(
        f"- **Branch {_branch_letter(i)}** — `{t.step_type}` "
        f"(`{nb._model_algorithm(t)}`, node `{t.node_id}`)"
        for i, t in enumerate(terminals)
    )
    return (
        f"# Skyulf pipeline — `{dataset_name or dataset_id}` (multi-branch)\n\n"
        f"_Config fingerprint: `{config_fingerprint(cfg)}`._\n\n"
        f"This pipeline has **{len(terminals)} training branches**. "
        f"Each branch is trained, evaluated, and persisted independently.\n\n"
        f"**Branches:**\n{items}\n\n"
        f"Shared preprocessing steps (common to every branch) are fitted once at the top; "
        f"branch-specific steps live inside each branch's section.\n"
    )


def _shared_preprocess_cells(shared: List[_NodeIn]) -> List[Dict[str, Any]]:
    if not shared:
        return [
            md_cell(
                "## 4. Shared preprocessing\n\n"
                "_None._ Each branch has its own preprocessing chain.\n"
            ),
            code_cell(
                "# No shared nodes — snapshot the raw loaded frame so each branch\n"
                "# can reset to the same starting point.\n"
                "df_shared = df.copy()\n"
            ),
        ]
    cells: List[Dict[str, Any]] = [
        md_cell(
            "## 4. Shared preprocessing\n\n"
            "These steps are common to every branch. They are fitted once on the "
            "full training frame before branching.\n"
        )
    ]
    for i, n in enumerate(shared, start=1):
        cells.append(node_to_cell(n, i))
    cells.append(
        code_cell(
            "# Snapshot the post-shared frame so each branch starts from the same point.\n"
            "df_shared = df.copy()\n"
        )
    )
    return cells


def _branch_persist_cell(letter: str, branch_only: List[_NodeIn]) -> Dict[str, Any]:
    artifact_dict = "".join(
        f"    {i:>2}: ({n.step_type!r}, step{i:02d}_artifact),\n"
        for i, n in enumerate(branch_only, start=1)
    )
    return code_cell(
        f"# Persist branch {letter}'s artifacts + fitted estimator.\n"
        "import pickle\n\n"
        "branch_artifacts = {\n" + artifact_dict + "}\n"
        f'with open(f"skyulf_branch_{letter}.pkl", "wb") as fh:\n'
        '    pickle.dump({"artifacts": branch_artifacts, "estimator": estimator}, fh)\n'
        f'print("Saved branch {letter} → skyulf_branch_{letter}.pkl")\n'
    )


def _branch_topology_md(letter: str, branch_nodes: List[_NodeIn]) -> Dict[str, Any]:
    lines = [f"<details><summary>Branch {letter} nodes (topological order)</summary>\n", "```"]
    for i, n in enumerate(branch_nodes, start=1):
        inputs = ", ".join(n.inputs) if n.inputs else "—"
        lines.append(f"  {i:>2}. {n.step_type:<28}  node_id={n.node_id}  inputs=[{inputs}]")
    lines.extend(["```", "</details>\n"])
    return md_cell("\n".join(lines))


def _full_branch_section(
    letter: str,
    section_no: int,
    branch_nodes: List[_NodeIn],
    classified: _Classified,
    shared_ids: set,
) -> List[Dict[str, Any]]:
    """Cells for one training branch in the full notebook."""
    _loader, preprocess, feat_target, train_test, model = classified
    branch_only = [n for n in preprocess if n.node_id not in shared_ids]
    algo = nb._model_algorithm(model) if model is not None else "none"
    title = (
        f"## {section_no}. Branch {letter} — "
        f"{model.step_type if model else 'no model'} (`{algo}`)"
    )
    cells: List[Dict[str, Any]] = [
        md_cell(
            f"{title}\n\n"
            f"Replays branch **{letter}** from the shared snapshot. "
            f"Branch-specific preprocessing: {len(branch_only)} step(s); "
            f"feat/target split: {'yes' if feat_target else 'no'}; "
            f"train/test split: {'yes' if train_test else 'no'}.\n"
        ),
        code_cell(
            f"# Reset to shared snapshot before running branch {letter}.\ndf = df_shared.copy()\n"
        ),
    ]
    for i, n in enumerate(branch_only, start=1):
        cells.append(node_to_cell(n, i))
    cells.extend(split_cells(feat_target, train_test, in_branch=True))
    cells.extend(modeling_cells(model, in_branch=True, branch_letter=letter))
    if model is not None:
        cells.append(_branch_persist_cell(letter, branch_only))
    cells.append(_branch_topology_md(letter, branch_nodes))
    return cells


@dataclass
class _FullBranchCtx:
    """Groups the parameters for `build_full_branched` to stay under param-count limit."""

    cfg: _PipelineIn
    dataset_id: str
    dataset_name: Optional[str]
    all_nodes: List[_NodeIn]
    terminals: List[_NodeIn]
    ancestors_in_topo: Callable[[str, List[_NodeIn]], List[_NodeIn]]
    classify: Callable[[List[_NodeIn]], _Classified]
    data_path_resolver: Callable[[Optional[_NodeIn]], str]
    resolved_from_db: bool


def _assemble_full_branch_cells(
    terminals: List[_NodeIn],
    branches: List[List[_NodeIn]],
    classifications: List[_Classified],
    shared_ids: set,
) -> List[Dict[str, Any]]:
    """Build per-branch section cells (extracted to reduce CCN of caller)."""
    cells: List[Dict[str, Any]] = []
    for i, (_t, branch_nodes, classified) in enumerate(zip(terminals, branches, classifications)):
        cells.extend(
            _full_branch_section(_branch_letter(i), 5 + i, branch_nodes, classified, shared_ids)
        )
    return cells


def _compute_shared_nodes(
    all_nodes: List[_NodeIn], branches: List[List[_NodeIn]]
) -> Tuple[set, List[_NodeIn]]:
    """Return (shared_ids, shared_node_list) for the given branches."""
    shared_ids = _shared_preprocess_ids(
        [[n for n in b if n.step_type not in _SPLIT_OR_MODEL] for b in branches]
    )
    shared_nodes = [
        n for n in all_nodes if n.node_id in shared_ids and n.step_type not in _SPLIT_OR_MODEL
    ]
    return shared_ids, shared_nodes


def build_full_branched(ctx: _FullBranchCtx) -> Dict[str, Any]:
    """Full notebook: shared preprocess section + one training section per branch.

    Accepts a :class:`_FullBranchCtx` dataclass instead of many positional
    arguments to stay under the Codacy parameter-count limit.
    """
    branches, classifications = _collect_branches(
        ctx.all_nodes, ctx.terminals, ctx.ancestors_in_topo, ctx.classify
    )
    shared_ids, shared_nodes = _compute_shared_nodes(ctx.all_nodes, branches)
    loader = next((n for n in ctx.all_nodes if n.step_type == "data_loader"), None)
    data_path = ctx.data_path_resolver(loader)
    cells: List[Dict[str, Any]] = [
        md_cell(_branch_sections_md(ctx.cfg, ctx.dataset_id, ctx.dataset_name, ctx.terminals)),
    ]
    cells.extend(full_intro_cells(data_path, resolved_from_db=ctx.resolved_from_db))
    cells.extend(_shared_preprocess_cells(shared_nodes))
    cells.extend(_assemble_full_branch_cells(ctx.terminals, branches, classifications, shared_ids))
    letters = [_branch_letter(i) for i in range(len(ctx.terminals))]
    n_branches = len(ctx.terminals)
    cells.extend(_metrics_comparison_cell(letters, 5 + n_branches))
    cells.extend(_full_inference_cells(letters, 6 + n_branches))
    return wrap_notebook(cells)


def _full_inference_cells(letters: List[str], section_no: int) -> List[Dict[str, Any]]:
    """Predict-on-new-data section appended after all branch sections in full mode."""
    load_lines = "".join(
        f'# branch_{l} = pickle.load(open("skyulf_branch_{l}.pkl", "rb"))\n'
        f'# estimator_{l} = branch_{l}["estimator"]\n'
        f"# pred_{l} = estimator_{l}.predict(new_df)\n\n"
        for l in letters  # noqa: E741
    )
    return [
        md_cell(
            f"## {section_no}. Predict on new data (per branch)\n\n"
            "Each branch's estimator was pickled at the end of its section.\n"
            "Load any artifact below and call `predict` on new rows.\n"
        ),
        code_cell(
            "import pickle\n\n"
            "# new_df = pd.read_csv('new_data.csv')\n"
            "# (edit the path and uncomment one block per branch you want to score)\n\n"
            + load_lines
        ),
    ]


def _metrics_helper_cell() -> Dict[str, Any]:
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


def _metrics_comparison_cell(letters: List[str], section_no: int) -> List[Dict[str, Any]]:
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
    dataset_name: Optional[str],
    terminals: List[_NodeIn],
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
) -> List[Dict[str, Any]]:
    """Fit + persist cells for one compact-mode branch."""
    _loader, preprocess, feat_target, train_test, model = classified
    target_col = feat_target.params.get("target_column", "") if feat_target else ""
    full_chain = [n for n in (feat_target, train_test) if n is not None] + preprocess
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
            f'BRANCH_{letter}_TARGET = "{target_col}"  # target column for this branch\n'
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


def _inference_snippet(letters: List[str]) -> Dict[str, Any]:
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
    dataset_name: Optional[str]
    all_nodes: List[_NodeIn]
    terminals: List[_NodeIn]
    ancestors_in_topo: Callable[[str, List[_NodeIn]], List[_NodeIn]]
    classify: Callable[[List[_NodeIn]], _Classified]
    data_path_resolver: Callable[[Optional[_NodeIn]], str]
    target_resolver: Callable[[Optional[_NodeIn], Optional[_NodeIn]], Optional[str]]
    resolved_from_db: bool


def _collect_branches(
    all_nodes: List[_NodeIn],
    terminals: List[_NodeIn],
    ancestors_in_topo: Callable[[str, List[_NodeIn]], List[_NodeIn]],
    classify: Callable[[List[_NodeIn]], _Classified],
) -> Tuple[List[List[_NodeIn]], List[_Classified]]:
    """Build per-branch node lists and classifications in one pass."""
    branches: List[List[_NodeIn]] = []
    classifications: List[_Classified] = []
    for t in terminals:
        anc = ancestors_in_topo(t.node_id, all_nodes)
        nodes = anc + [t]
        branches.append(nodes)
        classifications.append(classify(nodes))
    return branches, classifications


def _compact_compare_predict_cells(letters: List[str], base_section: int) -> List[Dict[str, Any]]:
    """Metrics comparison + predict-on-new-data tail for compact multi-branch notebooks."""
    compare_vars = ", ".join(f'"Branch {l}": pipeline_{l}_metrics' for l in letters)  # noqa: E741
    cells: List[Dict[str, Any]] = [
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


def build_compact_branched(ctx: _CompactBranchCtx) -> Dict[str, Any]:
    """Compact notebook: one `SkyulfPipeline` per terminal model.

    Accepts a :class:`_CompactBranchCtx` dataclass instead of many positional
    arguments to stay under the Codacy parameter-count limit.
    """
    _branches, classifications = _collect_branches(
        ctx.all_nodes, ctx.terminals, ctx.ancestors_in_topo, ctx.classify
    )
    loader = next((n for n in ctx.all_nodes if n.step_type == "data_loader"), None)
    data_path = ctx.data_path_resolver(loader)
    target_col: Optional[str] = None
    for c in classifications:
        target_col = ctx.target_resolver(c[2], c[3])
        if target_col:
            break
    target_col = target_col or "<target_column>"
    letters = [_branch_letter(i) for i in range(len(ctx.terminals))]
    cells: List[Dict[str, Any]] = [
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
