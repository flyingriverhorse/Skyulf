"""Notebook cell builders (compact + full mode).

Split out of `notebook_export.py` to keep the endpoint module under the
file-NLOC complexity budget. Pure functions only — they consume already-
classified `_NodeIn` lists and emit raw nbformat 4.5 cell dicts.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class _NodeIn(BaseModel):
    node_id: str
    step_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list)


class _PipelineIn(BaseModel):
    pipeline_id: Optional[str] = None
    nodes: List[_NodeIn]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Notebook primitives
# ---------------------------------------------------------------------------


def md_cell(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def wrap_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_DROP_PARAMS = {
    "_display_name",
    "_merge_strategy",
    "datasetId",
    "datasetName",
    "definitionType",
    "catalogType",
    "label",
    "title",
    "node_id",
    # Skyulf routing keys — consumed by the engine, not by the underlying estimator.
    # Passing them through causes XGBoost/sklearn warnings ("Parameters not used").
    "algorithm",
    "execution_mode",
}


def strip_internal_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k not in _DROP_PARAMS and not k.startswith("_")}


def metrics_helper_cell() -> Dict[str, Any]:
    """Notebook cell defining `_summarize_metrics(m)`.

    Skyulf's `evaluate()` and `SkyulfPipeline.fit()` return deeply-nested
    dicts (per-split classification reports, residuals, ...). This helper
    flattens them into a tidy DataFrame (rows=split, cols=metric, numeric
    only) so styling never trips on nested values.
    """
    return code_cell(
        "def _summarize_metrics(m):\n"
        '    """Flatten a Skyulf metrics dict into rows=split, cols=metric (numeric only)."""\n'
        "    import pandas as pd\n"
        "    if not isinstance(m, dict):\n"
        "        return pd.DataFrame()\n"
        "    # SkyulfPipeline.fit wraps eval in {'preprocessing': ..., 'modeling': <eval>}\n"
        "    if isinstance(m.get('modeling'), dict):\n"
        "        m = m['modeling']\n"
        "    splits = m.get('splits') if isinstance(m.get('splits'), dict) else None\n"
        "    rows = {}\n"
        "    if splits:\n"
        "        for name, report in splits.items():\n"
        "            scalars = getattr(report, 'metrics', None)\n"
        "            if scalars is None and isinstance(report, dict):\n"
        "                scalars = report.get('metrics', {})\n"
        "            if isinstance(scalars, dict):\n"
        "                rows[name] = {\n"
        "                    k: v for k, v in scalars.items() if isinstance(v, (int, float))\n"
        "                }\n"
        "    else:\n"
        "        flat = {k: v for k, v in m.items() if isinstance(v, (int, float))}\n"
        "        if flat:\n"
        "            rows['metrics'] = flat\n"
        "    return pd.DataFrame(rows).T\n"
    )


def _to_py_literal(d: Any) -> str:
    """Serialize *d* as a Python literal string.

    ``json.dumps`` produces JSON-syntax booleans/null (``true``, ``false``,
    ``null``) which are not valid Python identifiers.  This helper replaces
    them with their Python equivalents so the cell can be executed as-is.
    """
    j = json.dumps(d, indent=2, default=str)
    j = re.sub(r"\btrue\b", "True", j)
    j = re.sub(r"\bfalse\b", "False", j)
    j = re.sub(r"\bnull\b", "None", j)
    return j


def config_fingerprint(cfg: _PipelineIn) -> str:
    payload = json.dumps(
        [{"step": n.step_type, "params": n.params, "inputs": n.inputs} for n in cfg.nodes],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _model_algorithm(model: _NodeIn) -> str:
    """Resolve the registry key for a modeling node.

    `basic_training` / `advanced_tuning` are job-router step types — the real
    NodeRegistry key lives in `params.algorithm` (e.g. `xgboost_classifier`).
    Falls back to `step_type` for nodes that ARE direct registry entries.
    """
    return str(model.params.get("algorithm") or model.params.get("type") or model.step_type)


def _model_label(model: Optional[_NodeIn]) -> str:
    if model is None:
        return "none"
    return _model_algorithm(model)


def _build_modeling_block(model: _NodeIn) -> Dict[str, Any]:
    """Map a model `_NodeIn` to a SkyulfPipeline-compatible modeling dict.

    Re-injects ``type`` from ``params.algorithm`` (the engine's routing key,
    stripped from runtime params) so ``_init_model_estimator`` can resolve
    the estimator. For ``advanced_tuning`` we map to ``hyperparameter_tuner``
    and flatten the nested ``tuning_config`` fields so ``TuningCalculator.fit``
    picks them up via its keyword filter.
    """
    algorithm = str(model.params.get("algorithm") or model.params.get("type") or "")
    clean_params = strip_internal_params(model.params)
    if model.step_type == "advanced_tuning" and algorithm:
        tuning_cfg = clean_params.pop("tuning_config", {}) or {}
        return {
            "type": "hyperparameter_tuner",
            "node_id": model.node_id,
            "base_model": {"type": algorithm},
            **clean_params,
            **tuning_cfg,
        }
    return {"type": algorithm or model.step_type, "node_id": model.node_id, **clean_params}


def build_skyulf_config(preprocess: List[_NodeIn], model: Optional[_NodeIn]) -> Dict[str, Any]:
    """Convert engine `NodeConfig` shape -> SkyulfPipeline shape."""
    steps = [
        {"name": n.node_id, "transformer": n.step_type, "params": strip_internal_params(n.params)}
        for n in preprocess
    ]
    modeling = _build_modeling_block(model) if model is not None else {}
    return {"preprocessing": steps, "modeling": modeling}


# ---------------------------------------------------------------------------
# Compact mode cells
# ---------------------------------------------------------------------------


def compact_summary_md(
    cfg: _PipelineIn,
    dataset_id: str,
    dataset_name: Optional[str],
    preprocess: List[_NodeIn],
    feat_target: Optional[_NodeIn],
    train_test: Optional[_NodeIn],
    model: Optional[_NodeIn],
) -> str:
    return (
        f"# Skyulf pipeline — `{dataset_name or dataset_id}` (compact)\n\n"
        f"_Auto-exported from the Skyulf canvas. Config fingerprint: "
        f"`{config_fingerprint(cfg)}`._\n\n"
        f"This notebook runs the saved pipeline end-to-end via `SkyulfPipeline`. "
        f"It is meant for **production / inference handoff**: fit once on the training set, "
        f"persist the artifact, then `predict()` on new data.\n\n"
        f"**Graph summary:** {len(preprocess)} preprocessing step(s) · "
        f"feature/target split: {'yes' if feat_target else 'no'} · "
        f"train/test split: {'yes' if train_test else 'no'} · "
        f"model: `{_model_label(model)}`.\n"
    )


_FASTAPI_SNIPPET_CELL = code_cell(
    "FASTAPI_SNIPPET = '''\n"
    "from fastapi import FastAPI\n"
    "import pandas as pd\n"
    "from skyulf.pipeline import SkyulfPipeline\n"
    "\n"
    "app = FastAPI()\n"
    'pipeline = SkyulfPipeline.load("skyulf_pipeline.pkl")\n'
    "\n"
    '@app.post("/predict")\n'
    "def predict(rows: list[dict]):\n"
    "    df = pd.DataFrame(rows)\n"
    '    return {"predictions": pipeline.predict(df).tolist()}\n'
    "'''\n"
    "print(FASTAPI_SNIPPET)\n"
)


def compact_load_cells(
    data_path: str, target_col: str, resolved_from_db: bool = False
) -> List[Dict[str, Any]]:
    return [
        md_cell("## 1. Imports\n"),
        code_cell(
            "import pandas as pd\n"
            "import skyulf  # noqa: F401  # populates the NodeRegistry\n"
            "from skyulf.pipeline import SkyulfPipeline\n"
        ),
        md_cell(_data_path_guidance_md(data_path, resolved_from_db)),
        code_cell(
            "# EDIT THIS to the absolute or relative path of your training CSV.\n"
            '# Use a raw string (r"...") on Windows to avoid escaping backslashes.\n'
            f'TRAIN_PATH = r"{data_path}"\n'
            f'TARGET_COLUMN = "{target_col}"\n'
            "\n"
            "df = pd.read_csv(TRAIN_PATH)\n"
            "df.head()\n"
        ),
        metrics_helper_cell(),
    ]


def compact_run_cells(config_json: str) -> List[Dict[str, Any]]:
    return [
        md_cell(
            "## 3. Pipeline configuration\n\n"
            "Generated from the canvas. Edit any `params` block to tweak a step "
            "without re-opening the UI.\n"
        ),
        code_cell(f"pipeline_config = {config_json}\n"),
        md_cell("## 4. Fit + evaluate\n"),
        code_cell(
            "pipeline = SkyulfPipeline(pipeline_config)\n"
            "metrics = pipeline.fit(df, target_column=TARGET_COLUMN)\n"
            "_summarize_metrics(metrics).style.format('{:.4f}', na_rep='-')"
            ".background_gradient(cmap='RdYlGn', axis=0)"
            ".set_caption('Pipeline metrics \u2014 train vs test')\n"
        ),
        md_cell("## 5. Predict on new data\n"),
        code_cell(
            "# new_df = pd.read_csv('new_data.csv')\n"
            "# predictions = pipeline.predict(new_df)\n"
            "# predictions[:10]\n"
        ),
    ]


def compact_persist_cells() -> List[Dict[str, Any]]:
    return [
        md_cell(
            "## 6. Persist for production\n\n"
            "`SkyulfPipeline.save()` pickles the fitted pipeline (preprocessing artifacts "
            "+ model) into a single file. Reload with `SkyulfPipeline.load(path)`.\n"
        ),
        code_cell(
            'ARTIFACT_PATH = "skyulf_pipeline.pkl"\n'
            "pipeline.save(ARTIFACT_PATH)\n"
            "loaded = SkyulfPipeline.load(ARTIFACT_PATH)\n"
            "# loaded.predict(new_df)\n"
        ),
        md_cell(
            "## 7. (Optional) FastAPI inference snippet\n\n"
            "Drop this into a `service.py` to expose the pipeline behind HTTP.\n"
        ),
        _FASTAPI_SNIPPET_CELL,
    ]


# ---------------------------------------------------------------------------
# Full mode cells
# ---------------------------------------------------------------------------


def _data_path_guidance_md(data_path: str, resolved_from_db: bool) -> str:
    """Markdown block above every load cell.

    The exporter tries to auto-fill ``DATA_PATH`` from the registered
    ``DataSource``, but the resolved path is server-side and may not exist on
    the user's machine (different OS, container vs host, file moved). So
    we always print clear instructions on how to set it correctly.
    """
    if resolved_from_db:
        provenance = (
            f"The exporter pre-filled `DATA_PATH` with the path Skyulf knows: "
            f"`{data_path}`. **If you're running this notebook on a different machine "
            f"or the file has moved, edit the path below.**"
        )
    else:
        provenance = (
            "The exporter could not resolve a real on-disk path for this dataset "
            f"(falling back to `{data_path}`). **You must edit `DATA_PATH` below "
            "to point at the CSV on your machine before running the next cell.**"
        )
    return (
        "## 2. Load data\n\n"
        f"{provenance}\n\n"
        "**How to set `DATA_PATH`:**\n"
        "- Use a Python raw string so backslashes don't need escaping: "
        '`DATA_PATH = r"C:\\path\\to\\data.csv"`\n'
        '- On macOS / Linux: `DATA_PATH = "/Users/me/data/train.csv"`\n'
        '- Relative to the notebook: `DATA_PATH = "data/train.csv"`\n'
        "- For other formats swap `pd.read_csv` for `pd.read_parquet`, "
        "`pd.read_excel`, etc.\n"
    )


def node_to_cell(n: _NodeIn, idx: int) -> Dict[str, Any]:
    config_json = _to_py_literal(strip_internal_params(n.params))
    var = f"step{idx:02d}"
    return code_cell(
        f"# Step {idx}: {n.step_type}  ·  node_id = {n.node_id}\n"
        f"{var}_calc = NodeRegistry.get_calculator({n.step_type!r})()\n"
        f"{var}_apply = NodeRegistry.get_applier({n.step_type!r})()\n"
        f"{var}_config = {config_json}\n"
        f"{var}_artifact = {var}_calc.fit(df, {var}_config)\n"
        f"df = {var}_apply.apply(df, {var}_artifact)\n"
        f"df.head()\n"
    )


def topology_summary(nodes: List[_NodeIn]) -> str:
    lines = ["```", "Topology (topological order):"]
    for i, n in enumerate(nodes, start=1):
        inputs = ", ".join(n.inputs) if n.inputs else "—"
        label = n.params.get("_display_name") or n.step_type
        lines.append(f"  {i:>2}. {label:<28}  step_type={n.step_type:<28}  inputs=[{inputs}]")
    lines.append("```\n")
    return "\n".join(lines)


def full_summary_md(
    cfg: _PipelineIn,
    dataset_id: str,
    dataset_name: Optional[str],
    nodes: List[_NodeIn],
    preprocess: List[_NodeIn],
    feat_target: Optional[_NodeIn],
    train_test: Optional[_NodeIn],
    model: Optional[_NodeIn],
) -> str:
    return (
        f"# Skyulf pipeline — `{dataset_name or dataset_id}` (full)\n\n"
        f"_Auto-exported from the Skyulf canvas. Config fingerprint: "
        f"`{config_fingerprint(cfg)}`._\n\n"
        f"Each preprocessing node is rendered as its own cell so you can\n"
        f"tweak parameters, swap nodes, or inspect intermediate state.\n\n"
        f"**Graph:** {len(preprocess)} preprocessing step(s) · "
        f"feature/target split: {'yes' if feat_target else 'no'} · "
        f"train/test split: {'yes' if train_test else 'no'} · "
        f"model: `{_model_label(model)}`.\n\n" + topology_summary(nodes)
    )


def _no_model_cells(in_branch: bool = False) -> List[Dict[str, Any]]:
    h = "###" if in_branch else "## 6."
    return [
        md_cell(
            f"{h} Modeling\n\n"
            "_No model node in this pipeline._ Add a `BasicTraining` or "
            "`AdvancedTuning` node on the canvas, or wire up your own sklearn "
            "estimator below.\n"
        ),
        code_cell(
            "# Example: drop in your own estimator.\n"
            "# from sklearn.linear_model import LogisticRegression\n"
            "# model = LogisticRegression(max_iter=200)\n"
            "# model.fit(X_train, y_train)\n"
        ),
    ]


def _modeling_cell_basic(model: _NodeIn, algo: str, config_json: str, metrics_var: str) -> str:
    return (
        "from skyulf.data.dataset import SplitDataset\n"
        "from skyulf.modeling.base import StatefulEstimator\n"
        "from skyulf.registry import NodeRegistry\n"
        "\n"
        f"model_config = {config_json}\n"
        f"# Registry key: algorithm name (e.g. 'xgboost_classifier'); the\n"
        f"# canvas step_type ({model.step_type!r}) is a job-router label, not\n"
        f"# a registry key.\n"
        f"ALGORITHM = {algo!r}\n"
        "model_calc = NodeRegistry.get_calculator(ALGORITHM)()\n"
        "model_apply = NodeRegistry.get_applier(ALGORITHM)()\n"
        f"estimator = StatefulEstimator(node_id={model.node_id!r}, "
        "calculator=model_calc, applier=model_apply)\n"
        "\n"
        "dataset = SplitDataset(\n"
        "    train=X_train.assign(**{TARGET_COLUMN: y_train}),\n"
        "    test=X_test.assign(**{TARGET_COLUMN: y_test}),\n"
        "    validation=None,\n"
        ")\n"
        "_ = estimator.fit_predict(\n"
        "    dataset=dataset, target_column=TARGET_COLUMN, config=model_config,\n"
        "    log_callback=print,\n"
        ")\n"
        f"{metrics_var} = estimator.evaluate(dataset=dataset, target_column=TARGET_COLUMN)\n"
        f"_summarize_metrics({metrics_var}).style.format('{{:.4f}}', na_rep='-')"
        f".background_gradient(cmap='RdYlGn', axis=0)"
        f".set_caption('{metrics_var} \u2014 train vs test')\n"
    )


_TUNING_TRIAL_CALLBACK_SRC = (
    "def _on_trial(current, total, score=None, params=None):\n"
    "    msg = f'Trial {current}/{total}'\n"
    "    if score is not None:\n"
    "        msg += f' \u2014 score={score:.4f}'\n"
    "    print(msg)\n"
)


def _modeling_cell_tuning(model: _NodeIn, algo: str, config_json: str, metrics_var: str) -> str:
    """Render an `advanced_tuning` cell that mirrors the engine's runner.

    The engine's `_run_advanced_tuning` wraps the base calculator/applier in
    `TuningCalculator`/`TuningApplier` before invoking `fit_predict`. Without
    that wrap the base estimator silently ignores `tuning_config` and runs a
    single default-param fit. We also pass `log_callback=print` and a small
    `progress_callback` so each trial is visible in the notebook output.
    """
    return (
        "from skyulf.data.dataset import SplitDataset\n"
        "from skyulf.modeling.base import StatefulEstimator\n"
        "from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator\n"
        "from skyulf.registry import NodeRegistry\n\n"
        f"tuning_config = {config_json}\n"
        f"# Registry key: algorithm name; canvas step_type ({model.step_type!r}) is\n"
        "# a job-router label, not a registry key.\n"
        f"ALGORITHM = {algo!r}\n"
        "base_calc = NodeRegistry.get_calculator(ALGORITHM)()\n"
        "base_apply = NodeRegistry.get_applier(ALGORITHM)()\n"
        "# Tuning wrappers — required for `tuning_config` to drive an Optuna /\n"
        "# grid / random search instead of a single default fit.\n"
        "tuner_calc = TuningCalculator(base_calc)\n"
        "tuner_apply = TuningApplier(base_apply)\n"
        f"estimator = StatefulEstimator(node_id={model.node_id!r}, "
        "calculator=tuner_calc, applier=tuner_apply)\n\n"
        "dataset = SplitDataset(\n"
        "    train=X_train.assign(**{TARGET_COLUMN: y_train}),\n"
        "    test=X_test.assign(**{TARGET_COLUMN: y_test}),\n"
        "    validation=None,\n"
        ")\n\n" + _TUNING_TRIAL_CALLBACK_SRC + "\n_ = estimator.fit_predict(\n"
        "    dataset=dataset, target_column=TARGET_COLUMN,\n"
        "    config=tuning_config['tuning_config'],\n"
        "    progress_callback=_on_trial, log_callback=print,\n"
        ")\n"
        f"{metrics_var} = estimator.evaluate(dataset=dataset, target_column=TARGET_COLUMN)\n"
        f"_summarize_metrics({metrics_var}).style.format('{{:.4f}}', na_rep='-')"
        ".background_gradient(cmap='RdYlGn', axis=0)"
        f".set_caption('{metrics_var} \u2014 train vs test')\n"
    )


def modeling_cells(
    model: Optional[_NodeIn], in_branch: bool = False, branch_letter: Optional[str] = None
) -> List[Dict[str, Any]]:
    if model is None:
        return _no_model_cells(in_branch)
    params = strip_internal_params(model.params)
    config_json = _to_py_literal(params)
    algo = _model_algorithm(model)
    h = "###" if in_branch else "## 6."
    metrics_var = f"metrics_{branch_letter}" if branch_letter else "report"
    is_tuning = model.step_type == "advanced_tuning"
    cell_src = (
        _modeling_cell_tuning(model, algo, config_json, metrics_var)
        if is_tuning
        else _modeling_cell_basic(model, algo, config_json, metrics_var)
    )
    intro = (
        "Hyperparameter search runs **here**: each trial is printed as it "
        "completes (Optuna also logs to stderr). "
        if is_tuning
        else ""
    )
    return [
        md_cell(
            f"{h} Train & evaluate — `{model.step_type}` (`{algo}`)\n\n"
            f"{intro}Each branch trains **independently** from its own split. "
            f"Results are stored in `{metrics_var}`.\n"
        ),
        code_cell(cell_src),
    ]


def _feat_target_cells(
    feat_target: Optional[_NodeIn], in_branch: bool = False
) -> List[Dict[str, Any]]:
    h = "###" if in_branch else "## 4."
    if feat_target is not None:
        target_col = feat_target.params.get("target_column", "<target_column>")
        return [
            md_cell(f"{h} Feature / target split — `{target_col}`\n"),
            code_cell(
                f'TARGET_COLUMN = "{target_col}"\n'
                "X = df.drop(columns=[TARGET_COLUMN])\n"
                "y = df[TARGET_COLUMN]\n"
                "X.shape, y.shape\n"
            ),
        ]
    return [
        md_cell(f"{h} Feature / target split\n"),
        code_cell(
            'TARGET_COLUMN = "<target_column>"  # set this manually\n'
            "X = df.drop(columns=[TARGET_COLUMN])\n"
            "y = df[TARGET_COLUMN]\n"
        ),
    ]


def _train_test_cells(
    train_test: Optional[_NodeIn], in_branch: bool = False
) -> List[Dict[str, Any]]:
    h = "###" if in_branch else "## 5."
    if train_test is not None:
        p = train_test.params
        return [
            md_cell(f"{h} Train / test split\n"),
            code_cell(
                "from sklearn.model_selection import train_test_split\n\n"
                "X_train, X_test, y_train, y_test = train_test_split(\n"
                f"    X, y, test_size={p.get('test_size', 0.2)}, "
                f"random_state={p.get('random_state', 42)},\n"
                f"    stratify=y if {bool(p.get('stratify', False))} else None,\n"
                ")\n"
                "X_train.shape, X_test.shape\n"
            ),
        ]
    return [
        md_cell(f"{h} Train / test split\n"),
        code_cell(
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, "
            "test_size=0.2, random_state=42)\n"
        ),
    ]


def split_cells(
    feat_target: Optional[_NodeIn],
    train_test: Optional[_NodeIn],
    in_branch: bool = False,
) -> List[Dict[str, Any]]:
    return _feat_target_cells(feat_target, in_branch) + _train_test_cells(train_test, in_branch)


def persist_cells(preprocess: List[_NodeIn]) -> List[Dict[str, Any]]:
    artifact_dict = "".join(
        f"    {i:>2}: ({n.step_type!r}, step{i:02d}_artifact),\n"
        for i, n in enumerate(preprocess, start=1)
    )
    return [
        md_cell(
            "## 7. Persist artifacts for inference\n\n"
            "Pickle the per-step artifacts together with the model so a "
            "downstream service can re-apply the exact same transformations.\n"
        ),
        code_cell(
            "import pickle\n\n"
            "artifacts = {\n" + artifact_dict + "}\n"
            'with open("skyulf_artifacts.pkl", "wb") as fh:\n'
            '    pickle.dump({"artifacts": artifacts}, fh)\n'
        ),
        md_cell(
            "## 8. Predict on new data (replay loop)\n\n"
            "Re-applies each saved artifact in order. Use this as the template "
            "for a production scoring service.\n"
        ),
        code_cell(
            "def predict_new(new_df: pd.DataFrame) -> pd.DataFrame:\n"
            "    out = new_df.copy()\n"
            "    for _idx, (step_type, artifact) in artifacts.items():\n"
            "        applier = NodeRegistry.get_applier(step_type)()\n"
            "        out = applier.apply(out, artifact)\n"
            "    return out\n"
            "\n"
            "# new_df = pd.read_csv('new_data.csv')\n"
            "# predict_new(new_df).head()\n"
        ),
    ]


def full_intro_cells(data_path: str, resolved_from_db: bool = False) -> List[Dict[str, Any]]:
    return [
        md_cell("## 1. Imports\n"),
        code_cell(
            "import pandas as pd\n"
            "import numpy as np  # noqa: F401\n"
            "import skyulf  # noqa: F401  # populates the registry\n"
            "from skyulf.registry import NodeRegistry\n"
        ),
        md_cell(_data_path_guidance_md(data_path, resolved_from_db)),
        code_cell(
            "# EDIT THIS to the absolute or relative path of your CSV.\n"
            '# Use a raw string (r"...") on Windows to avoid escaping backslashes.\n'
            f'DATA_PATH = r"{data_path}"\n'
            "\n"
            "df = pd.read_csv(DATA_PATH)\n"
            "print(f'Loaded shape: {df.shape}')\n"
            "df.head()\n"
        ),
        metrics_helper_cell(),
        md_cell(
            "## 3. Apply preprocessing steps\n\n"
            "Each cell below mirrors a single canvas node. The `df` variable is "
            "threaded through; every step writes its fitted artifact to a "
            "`stepNN_artifact` variable so you can inspect what was learned.\n"
        ),
    ]
