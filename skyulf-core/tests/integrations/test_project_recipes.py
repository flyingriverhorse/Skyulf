"""Named project recipes remain independent across branches and saved artifacts."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.inference.project_code import load_project_module
from skyulf.integrations.databricks.project import load_project_workflow

CUSTOM = (
    Path(__file__).resolve().parents[2]
    / "templates/databricks/template/{{.project_name}}/src/features/custom"
)


def _project(tmp_path):
    """Build one package whose inactive default registers no custom classes."""
    root = tmp_path / "features"
    root.mkdir()
    for filename in ("preprocessing_custom.py", "pre_split_custom.py"):
        shutil.copyfile(CUSTOM / filename, root / filename)
    root.joinpath("__init__.py").write_text(
        "from .preprocessing_custom import frequency_encoding\n"
        "from .pre_split_custom import minimum_completeness\n"
        "def build_preprocessing(recipe='default'):\n"
        "    if recipe == 'default': return []\n"
        "    if recipe not in ('category', 'other'): raise ValueError('Unknown recipe')\n"
        "    return [frequency_encoding([recipe])]\n"
        "def build_pre_split_steps(recipe='default'):\n"
        "    if recipe == 'default': return []\n"
        "    if recipe not in ('category', 'other'): raise KeyError(recipe)\n"
        "    return [minimum_completeness([recipe], min_present=1)]\n"
        "def build_scoring():\n"
        "    return {'reuse_pre_split': True, 'skip_target_steps': False}\n",
        encoding="utf-8",
    )
    return root


def _config(engine="pandas", column="category"):
    """Keep each branch's feature and target declarations explicit."""
    return {
        "engine": engine,
        "input_columns": [column, "amount"],
        "target_column": "target",
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
    }


def test_independent_selections_are_bound_into_saved_source(tmp_path):
    """Zero-argument replay must recreate both selected recipes with distinct identities."""
    root = _project(tmp_path)
    configs = [
        load_project_workflow(
            _config(column=column),
            root,
            preprocessing_recipe=column,
            pre_split_recipe=column,
        )
        for column in ("category", "other")
    ]
    sources = [config["pipeline"]["project_python_source"] for config in configs]
    assert sources[0] != sources[1]
    for config, column in zip(configs, ("category", "other"), strict=True):
        pipeline = config["pipeline"]
        assert pipeline["feature_recipes"] == {"preprocessing": column, "pre_split": column}
        module = load_project_module(pipeline["project_python_source"])
        assert module.build_preprocessing() == pipeline["preprocessing"]
        assert module.build_pre_split_steps() == config["pre_split_steps"]
        assert pipeline["preprocessing"][0]["params"]["params"]["columns"] == [column]
        assert pipeline["project_scoring"]["pre_split"]["steps"] == config["pre_split_steps"]
    default = load_project_workflow(_config(), root)["pipeline"]
    assert default["preprocessing"] == []
    assert default["feature_recipes"] == {"preprocessing": "default", "pre_split": "default"}


def test_phase_selections_do_not_select_each_other(tmp_path):
    """Choosing either phase must leave the other phase's default builder unchanged."""
    root = _project(tmp_path)
    preprocessing = load_project_workflow(_config(), root, preprocessing_recipe="category")
    pre_split = load_project_workflow(_config(), root, pre_split_recipe="category")
    assert preprocessing["pipeline"]["preprocessing"]
    assert preprocessing["pre_split_steps"] == []
    assert pre_split["pipeline"]["preprocessing"] == []
    assert pre_split["pre_split_steps"]


def test_saved_branch_plan_restores_selected_custom_registrations(tmp_path):
    """Fresh replay must register nondefault custom classes from both branch selections."""
    from skyulf.integrations.databricks.local_branches import (
        TrainingBranch,
        branch_training_payload,
    )
    from skyulf.integrations.databricks.local_retraining import LocalTrainingSpec

    root = _project(tmp_path)
    branches = []
    for column in ("category", "other"):
        config = load_project_workflow(
            _config(column=column),
            root,
            preprocessing_recipe=column,
            pre_split_recipe=column,
        )
        spec = LocalTrainingSpec(
            table="workspace.test.rows",
            version=0,
            record_key_columns=("id",),
            input_columns=(column, "amount"),
            target_column=f"{column}_target",
            max_rows=30,
            max_bytes=100000,
            pre_split_steps=tuple(config["pre_split_steps"]),
            drop_missing_labels=True,
        )
        branches.append(
            TrainingBranch(
                name=column,
                spec=spec,
                pipeline=config["pipeline"],
                model_name=f"workspace.test.{column}",
                metric="heldout_rmse",
            )
        )
    payload = branch_training_payload(tuple(branches))
    saved = tmp_path / "plan.json"
    saved.write_text(json.dumps(payload), encoding="utf-8")
    for path in root.glob("*.py"):
        path.write_text("raise RuntimeError('edited recipe')\n", encoding="utf-8")
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from skyulf.integrations.databricks.local_branches import "
        "restore_training_branches, branch_training_payload\n"
        "from skyulf.registry import NodeRegistry\n"
        "payload = json.loads(Path(sys.argv[1]).read_text())\n"
        "branches = restore_training_branches(payload)\n"
        "for branch in branches:\n"
        "    for step in (*branch.pipeline['preprocessing'], *branch.spec.pre_split_steps):\n"
        "        NodeRegistry.get_calculator(step['transformer'])\n"
        "print(json.dumps(branch_training_payload(branches)))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code, str(saved)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert json.loads(loaded.stdout) == payload


def test_branch_entry_selects_each_recipe(tmp_path, workflow_config):
    """Notebook branch entries pass both independent names through to captured code."""
    from skyulf.integrations.databricks.branch_notebook import _branch_config

    source = tmp_path / "src"
    source.mkdir()
    _project(source)
    modeling = source / "modeling"
    modeling.mkdir()
    base = {**workflow_config, **_config()}
    base.update(promotion_policy="manual_approval", score_handoff="disabled")
    values = {
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    config = _branch_config(
        base,
        {
            "workflow": {},
            "preprocessing_recipe": "category",
            "pre_split_recipe": "category",
        },
        values,
        modeling,
    )
    assert config["pipeline"]["preprocessing"][0]["params"]["params"]["columns"] == ["category"]
    assert config["pre_split_steps"][0]["params"]["columns"] == ["category"]
    with pytest.raises(ValueError, match="pre_split_recipe"):
        _branch_config(base, {"workflow": {}, "pre_split_recipe": "missing"}, values, modeling)


@pytest.mark.parametrize("option", ["preprocessing_recipe", "pre_split_recipe"])
@pytest.mark.parametrize("recipe", ["", " ", False, 12, [], "missing"])
def test_invalid_selection_fails_clearly(tmp_path, option, recipe):
    """Invalid and unknown names cannot silently execute a default recipe."""
    with pytest.raises(ValueError, match=option):
        load_project_workflow(_config(), _project(tmp_path), **{option: recipe})


@pytest.mark.parametrize("option", ["preprocessing_recipe", "pre_split_recipe"])
def test_legacy_zero_argument_factories_require_no_selection(tmp_path, option):
    """Old user projects keep working but cannot pretend to select a named recipe."""
    source = tmp_path / "preprocessing.py"
    text = "def build_preprocessing():\n    return []\n"
    source.write_text(text, encoding="utf-8", newline="\n")
    assert load_project_workflow(_config(), source)["pipeline"]["project_python_source"] == text
    with pytest.raises(ValueError, match=option):
        load_project_workflow(_config(), source, **{option: "default"})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("transport", ["local", "mlflow"])
def test_selected_recipes_fit_and_reload_in_fresh_process(tmp_path, monkeypatch, engine, transport):
    """Selected custom filters and learned frequencies survive edits without heldout leakage."""
    if transport == "mlflow":
        pytest.importorskip("mlflow")
    from skyulf.data.dataset import SplitDataset
    from skyulf.integrations.databricks.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.local_retraining import (
        LocalTrainingSpec,
        split_labeled_snapshot,
    )

    monkeypatch.chdir(tmp_path)
    root = _project(tmp_path)
    config = load_project_workflow(
        _config(engine), root, preprocessing_recipe="category", pre_split_recipe="category"
    )
    rows = pd.DataFrame(
        {
            "id": range(20),
            "category": ["A"] * 10 + ["B"] * 8 + [None] * 2,
            "amount": np.arange(1, 21, dtype=float),
            "target": np.arange(1, 21) * 2.0 + 1,
        }
    )
    spec = LocalTrainingSpec(
        table="workspace.test.rows",
        version=0,
        record_key_columns=("id",),
        input_columns=("category", "amount"),
        target_column="target",
        max_rows=30,
        max_bytes=100000,
        pre_split_steps=tuple(config["pre_split_steps"]),
    )
    train, heldout, _ = split_labeled_snapshot(rows, spec, engine=engine)
    assert set(train.amount) | set(heldout.amount) == set(range(1, 19))
    data = SplitDataset(
        train=pl.from_pandas(train) if engine == "polars" else train,
        test=pl.from_pandas(heldout) if engine == "polars" else heldout,
    )
    artifact = fit_local_workflow(
        config["pipeline"],
        data,
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=30,
        max_bytes=100000,
    )
    state = artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]
    assert state["state"]["category"] == train.category.value_counts(normalize=True).to_dict()
    saved_path = str(tmp_path / "artifact")
    if transport == "mlflow":
        saved_path = _log_model(tmp_path)
    for path in root.glob("*.py"):
        path.write_text("raise RuntimeError('edited recipe')\n", encoding="utf-8")
    code = (
        "import json, sys, pandas as pd\n"
        "from skyulf.inference.local_pipeline import load_local_pipeline\n"
        "from skyulf.inference.local_scoring import score_local_pipeline\n"
        "rows = pd.DataFrame({'category':['A','NEW',None], 'amount':[5.,10.,15.]})\n"
        "if sys.argv[2] == 'mlflow':\n"
        "    import mlflow\n    result = mlflow.pyfunc.load_model(sys.argv[1]).predict(rows)\n"
        "else:\n    result = score_local_pipeline(rows, load_local_pipeline(sys.argv[1]))\n"
        "print(json.dumps({'status': result.scoring_status.tolist(), "
        "'predictions': result.prediction.iloc[:2].tolist()}))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code, saved_path, transport],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert loaded.returncode == 0, loaded.stderr
    result = json.loads(loaded.stdout)
    assert result["status"] == ["predicted", "predicted", "excluded"]
    np.testing.assert_allclose(result["predictions"], [11.0, 21.0], atol=1e-8)


def _log_model(tmp_path):
    """Exercise the actual MLflow artifact packaging and fresh-process pyfunc loader."""
    import mlflow

    from skyulf.integrations.mlflow.local_model import log_local_model
    from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run

    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    with track_run(
        TrackingConfig(enabled=True, tracking_uri=uri, experiment_name="recipes"),
        run_name="recipes",
    ) as run:
        assert run.run_id is not None
        model_uri = log_local_model(
            tmp_path / "artifact", run_id=run.run_id, artifact_path="model", tracking_uri=uri
        )
    return mlflow.artifacts.download_artifacts(artifact_uri=model_uri, tracking_uri=uri)
