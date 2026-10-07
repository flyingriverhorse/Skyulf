"""Feature packages must survive training and replay without editable source files."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import predict_local_pipeline
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow


def _package(root):
    """Use synthetic custom pairs in a nested module with relative recipe imports."""
    root.mkdir()
    (root / "custom").mkdir()
    (root / "custom/__init__.py").write_text("", encoding="utf-8")
    example = Path(__file__).resolve().parents[3] / "tests/fixtures/custom_recipe.py"
    (root / "custom/steps.py").write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    (root / "__init__.py").write_text(
        "from .pre_split import build_pre_split_steps\n"
        "from .preprocessing import build_preprocessing\n",
        encoding="utf-8",
    )
    (root / "pre_split.py").write_text(
        "from .custom.steps import example_custom_pre_split\n"
        "def build_pre_split_steps():\n    return [example_custom_pre_split('is_test')]\n",
        encoding="utf-8",
    )
    (root / "preprocessing.py").write_text(
        "from .custom.steps import example_custom_step\n"
        "def build_preprocessing():\n    return [example_custom_step('x')]\n",
        encoding="utf-8",
    )
    return root


def _config():
    """Keep the model bounded while exercising actual project feature registration."""
    return {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_package_artifact_loads_without_editable_modules(tmp_path, engine):
    """Relative-imported custom classes must load from saved code in a fresh interpreter."""
    root = _package(tmp_path / "features")
    (root / "groups").mkdir()
    (root / "groups/company.py").write_text("raise AssertionError('Spark producer')\n")
    config = load_project_workflow(_config(), root)
    assert config["pre_split_steps"][0]["pre_split"]["required_columns"] == ["is_test"]
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    if engine == "polars":
        rows = pl.from_pandas(rows)
    artifact = fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=rows, test=rows[:0]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=10,
        max_bytes=10000,
    )
    expected = predict_local_pipeline(pd.DataFrame({"x": [5.0, 6.0]}), artifact)["prediction"]
    assert artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]["mean"] == 2.5
    for path in root.rglob("*.py"):
        path.write_text("raise RuntimeError('editable project must not load')\n", encoding="utf-8")
    code = (
        "import json, sys, pandas as pd\n"
        "from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline\n"
        "artifact = load_local_pipeline(sys.argv[1])\n"
        "print(json.dumps(predict_local_pipeline(pd.DataFrame({'x':[5.,6.]}), artifact)['prediction'].tolist()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "artifact")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(json.loads(result.stdout), expected)


def test_package_helper_changes_isolate_registered_versions(tmp_path):
    """A helper edit must change the whole package identity without replacing old classes."""
    root = _package(tmp_path / "features")
    before = load_project_workflow(_config(), root)
    helper = root / "custom/steps.py"
    helper.write_text(helper.read_text(encoding="utf-8") + "\nREVISION = 2\n", encoding="utf-8")
    after = load_project_workflow(_config(), root)
    assert before["pipeline"]["project_python_source"] != after["pipeline"]["project_python_source"]
    assert (
        before["pipeline"]["preprocessing"][0]["transformer"]
        != after["pipeline"]["preprocessing"][0]["transformer"]
    )


def test_spark_groups_do_not_change_model_source_or_consume_its_budget(tmp_path):
    """Upstream producers must not enter saved inference code or its source-size limit."""
    root = _package(tmp_path / "features")
    before = load_project_workflow(_config(), root)
    (root / "groups").mkdir()
    (root / "groups/company.py").write_text(
        "raise AssertionError('Spark producer is not a model hook')\n#" + "x" * 65536,
        encoding="utf-8",
    )
    after = load_project_workflow(_config(), root)
    assert after["pipeline"]["project_python_source"] == before["pipeline"]["project_python_source"]
    assert after["pipeline"]["preprocessing"] == before["pipeline"]["preprocessing"]


def test_model_helpers_can_still_use_nested_groups_packages(tmp_path):
    """Only the reserved top-level producer directory is excluded from feature snapshots."""
    from skyulf.inference.project_code import load_project_module

    root = _package(tmp_path / "features")
    helpers = root / "custom/groups"
    helpers.mkdir()
    (helpers / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    init = root / "__init__.py"
    init.write_text(init.read_text(encoding="utf-8") + "\nfrom .custom.groups import VALUE\n")
    config = load_project_workflow(_config(), root)
    saved = load_project_module(config["pipeline"]["project_python_source"])
    assert saved.VALUE == 7


def test_non_feature_package_still_captures_groups_modules(tmp_path):
    """Composition and generic callers must not silently lose helpers named groups."""
    from skyulf.inference.project_code import load_project_module
    from skyulf.integrations.databricks.model_sets.model_set_project import (
        capture_set_composition,
    )

    root = tmp_path / "src/composition"
    (root / "groups").mkdir(parents=True)
    (root / "__init__.py").write_text("from .groups import VALUE\n")
    (root / "groups/__init__.py").write_text("VALUE = 11\n")
    source = capture_set_composition({"config_path": str(tmp_path / "config/training.yml")})
    assert load_project_module(source).VALUE == 11


def test_package_snapshot_rejects_missing_init_and_oversize(tmp_path):
    """Incomplete or oversized packages cannot silently drop source from model delivery."""
    root = tmp_path / "features"
    root.mkdir()
    with pytest.raises(ValueError, match="__init__"):
        load_project_workflow(_config(), root)
    (root / "__init__.py").write_text("#" * (64 * 1024 + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="64 KiB"):
        load_project_workflow(_config(), root)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_package_custom_filter_and_fold_learning_remain_separate(tmp_path, monkeypatch, engine):
    """Filtering precedes splitting while custom fitted means use each CV training fold."""
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        LocalTrainingSpec,
        split_labeled_snapshot,
    )
    from skyulf.integrations.databricks.training.tuning.local_cv import (
        LocalCVSpec,
        evaluate_training_cv,
    )
    from skyulf.preprocessing.base import BaseCalculator
    from skyulf.registry import NodeRegistry

    root = _package(tmp_path / "features")
    config = load_project_workflow(_config(), root)
    frame = pd.DataFrame(
        {
            "id": range(16),
            "x": np.arange(16, dtype=float),
            "target": np.arange(16, dtype=float) * 2,
            "is_test": [False] * 12 + [True] * 4,
        }
    )
    spec = LocalTrainingSpec(
        table="workspace.test.labels",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=30,
        max_bytes=100000,
        pre_split_steps=tuple(config["pre_split_steps"]),
    )
    train, held, _ = split_labeled_snapshot(frame, spec, engine=engine)
    assert set(train.x) | set(held.x) == set(range(12))
    assert held.attrs["pre_split_filter_counts"][0]["excluded_rows"] == 4
    calculator = NodeRegistry.get_calculator(config["pipeline"]["preprocessing"][0]["transformer"])
    assert issubclass(calculator, BaseCalculator)
    original = calculator.fit
    observed = []

    def record_fit(self, X, params):
        """Record fitted means without altering the real custom implementation."""
        state = original(self, X, params)
        observed.append(state["mean"])
        return state

    monkeypatch.setattr(calculator, "fit", record_fit)
    eligible = frame.iloc[:12][["x", "target"]]
    if engine == "polars":
        eligible = pl.from_pandas(eligible)
    report = evaluate_training_cv(
        eligible,
        config["pipeline"],
        LocalCVSpec(enabled=True, folds=3, shuffle=False),
        target_column="target",
    )
    assert report is not None
    assert sorted(observed) == [3.5, 5.5, 7.5]


def test_organized_modeling_hooks_are_both_applied(tmp_path):
    """Moving hooks must retain composition before the selected-model search override."""
    root = _package(tmp_path / "features")
    modeling = tmp_path / "modeling"
    modeling.mkdir()
    (modeling / "ensemble.py").write_text(
        "def build_ensemble_params(model_type):\n"
        "    return {'base_estimators': ['linear_regression', 'ridge'], 'weights': [2, 1]}\n",
        encoding="utf-8",
    )
    (modeling / "tuning.py").write_text(
        "def build_search_space(model_type, strategy, params):\n"
        "    assert params['weights'] == [2, 1]\n"
        "    return {'ridge__alpha': [0.1, 1.0]}\n",
        encoding="utf-8",
    )
    config = _config()
    config["pipeline"]["modeling"] = {
        "type": "hyperparameter_tuner",
        "strategy": "grid",
        "base_model": {"type": "voting_regressor", "params": {}},
    }
    result = load_project_workflow(config, root)["pipeline"]
    assert result["modeling"]["base_model"]["params"]["weights"] == [2, 1]
    assert result["modeling"]["search_space"] == {"ridge__alpha": [0.1, 1.0]}
    assert result["ensemble_python_sha256"] and result["search_python_sha256"]


def test_package_failed_import_removes_partially_loaded_modules(tmp_path):
    """A failed package import must not leave stale children to satisfy a future load."""
    from skyulf.inference.project_code import project_source_digest
    from skyulf.integrations.databricks.projects._project_files import project_source

    root = _package(tmp_path / "features")
    source = (root / "__init__.py").read_text(
        encoding="utf-8"
    ) + "\nraise RuntimeError('broken package')\n"
    (root / "__init__.py").write_text(source, encoding="utf-8")
    prefix = "_skyulf_project_" + project_source_digest(project_source(root))
    with pytest.raises(RuntimeError, match="broken package"):
        load_project_workflow(_config(), root)
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)


def test_package_lazy_import_uses_saved_helper_after_edit(tmp_path):
    """Helpers first imported at prediction time must still use the pinned source."""
    from skyulf.inference.project_code import load_project_module

    root = _package(tmp_path / "features")
    (root / "custom/values.py").write_text("VALUE = 7\n", encoding="utf-8")
    init = root / "__init__.py"
    init.write_text(
        init.read_text(encoding="utf-8")
        + "\ndef value():\n    from .custom.values import VALUE\n    return VALUE\n",
        encoding="utf-8",
    )
    config = load_project_workflow(_config(), root)
    (root / "custom/values.py").write_text("raise RuntimeError('changed')\n", encoding="utf-8")
    saved = load_project_module(config["pipeline"]["project_python_source"])
    assert saved.value() == 7
