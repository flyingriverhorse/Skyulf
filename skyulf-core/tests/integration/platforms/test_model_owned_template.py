"""Generated models own independent editable parameters and search spaces."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"


def test_new_template_does_not_install_shared_model_overrides():
    """New projects must have one authoritative settings location per model."""
    modeling = ROOT / "template/{{.project_name}}/src/modeling"
    assert not (modeling / "ensemble.py").exists()
    assert not (modeling / "tuning.py").exists()


@pytest.mark.skipif(not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI opt-in")
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("family", ["voting", "stacking"])
def test_single_ensemble_questions_materialize_model_owned_settings(tmp_path, task, family):
    """Bundle selections must become visible editable params and selected-member search axes."""
    from test_databricks_bundle_generation import _generate_project

    from skyulf.integrations.databricks.training.tuning.cv import CVSpec
    from skyulf.integrations.databricks.training.tuning.search import prepare_search_pipeline

    suffix = "classifier" if task == "classification" else "regressor"
    project = _generate_project(
        tmp_path,
        task=task,
        search_n_trials="1",
        **{
            f"{task}_model": f"{family}_{suffix}",
            f"single_ensemble_{task}_base_1": "decision_tree",
            f"single_ensemble_{task}_base_2": "random_forest",
            "single_ensemble_cv": "2",
        },
    )
    from test_databricks_bundle_generation import _read_modeling

    pipeline = {"preprocessing": [], "modeling": _read_modeling(project)}
    model = pipeline["modeling"]
    assert model["base_model"]["params"]["base_estimators"] == ["decision_tree", "random_forest"]
    assert model["search_space"]
    assert any(key.startswith("decision_tree__") for key in model["search_space"])
    assert any(key.startswith("random_forest__") for key in model["search_space"])
    prepared = prepare_search_pipeline(
        pipeline, CVSpec(folds=2), target_column="target", event_column=None
    )
    assert prepared["modeling"]["search_space"]
    assert not (project / "src/modeling/ensemble.py").exists()
    assert not (project / "src/modeling/tuning.py").exists()


@pytest.mark.skipif(not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI opt-in")
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_generated_mixed_branches_fit_and_score_heldout_without_edits(tmp_path, engine):
    """Standalone and ensemble branches must train from their own generated model settings."""
    import math

    import numpy as np
    import pandas as pd
    import polars as pl
    from test_databricks_bundle_generation import _generate_project
    from test_guided_branches import _load_configs

    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.pipeline_evaluation import evaluate_holdout
    from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec
    from skyulf.integrations.databricks.training.tuning.search import prepare_search_pipeline

    choices = [
        ("regression", "ridge_regression"),
        ("classification", "logistic_regression"),
        ("regression", "voting_regressor"),
        ("classification", "stacking_classifier"),
    ]
    settings = {}
    for slot, (task, model) in enumerate(choices, 1):
        settings.update(
            {
                f"branch_{slot}_task": task,
                f"branch_{slot}_{task}_model": model,
                f"branch_{slot}_cv_enabled": "true",
                f"branch_{slot}_cv_folds": "2",
                f"branch_{slot}_search_n_trials": "1",
            }
        )
    project = _generate_project(
        tmp_path,
        training_layout="multi_target",
        engine=engine,
        branch_count="4",
        branch_1_preprocessing_recipe="example_imputer",
        branch_4_ensemble_classification_base_2="decision_tree",
        branch_4_ensemble_cv="2",
        **settings,
    )
    x = np.linspace(-3, 3, 80)
    fitted = {}
    for name, config in _load_configs(project).items():
        target = config["target_column"]
        y = (x > 0).astype(int) if config["task"] == "classification" else 2 * x + 1
        frame = pd.DataFrame({"feature_value": x, target: y})
        holdout = frame.iloc[::5].copy()
        train = frame.drop(holdout.index).copy()
        if engine == "polars":
            train, holdout = pl.from_pandas(train), pl.from_pandas(holdout)
        pipeline = prepare_search_pipeline(
            config["pipeline"],
            CVSpec.from_workflow(config),
            target_column=target,
            event_column=None,
        )
        artifact = fit_workflow(
            pipeline,
            SplitDataset(train=train, test=train.head(0)),
            target_column=target,
            artifact_path=tmp_path / name,
            max_rows=100,
            max_bytes=1_000_000,
        )
        metrics = evaluate_holdout(artifact, holdout, target_column=target)
        fitted[name] = artifact.manifest.task
        assert math.isfinite(metrics[config["metric"]])
        assert artifact.manifest.input_columns == ("feature_value",)
    assert list(fitted.values()) == [task for task, _ in choices]


@pytest.mark.skipif(not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI opt-in")
def test_generated_preview_shows_independent_branch_settings(tmp_path):
    """Offline preview must describe branch models instead of the unused root placeholder."""
    import subprocess
    import sys

    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(
        tmp_path,
        training_layout="multi_target",
        branch_1_name="revenue",
        branch_1_regression_model="ridge_regression",
        branch_2_name="risk",
        branch_2_task="classification",
        branch_2_classification_model="logistic_regression",
    )
    result = subprocess.run(
        [sys.executable, str(project / "src/tools/preview.py"), "--action", "train"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Branch: revenue" in result.stdout
    assert "Branch: risk" in result.stdout
    assert "ridge_regression" in result.stdout
    assert "logistic_regression" in result.stdout
