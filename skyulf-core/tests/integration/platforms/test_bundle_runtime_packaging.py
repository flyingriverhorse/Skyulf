"""Exercise generated requirements installation without a local Core source or wheel."""

import os
import shutil
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def core_requirement():
    """Read the source pin whose release alignment is checked by the branch-template contract."""
    template = (
        Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
    )
    return next(
        line
        for line in (template / "deployment/requirements.txt.tmpl").read_text().splitlines()
        if line.startswith("skyulf-core[")
    )


@pytest.mark.skipif(
    os.environ.get("SKYULF_BUNDLE_OFFLINE_CLI") != "1" or not shutil.which("databricks"),
    reason="Requires explicitly enabled offline Bundle CLI generation.",
)
@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
def test_generated_jobs_install_core_from_one_shared_requirement(
    tmp_path, layout, compute, core_requirement
):
    """Every generated job must obtain the same release without a local Core checkout or wheel."""
    from packaging.requirements import Requirement
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(
        tmp_path,
        training_layout=layout,
        compute_mode=compute,
        include_online_publication="true",
        # Core supports online serving only for single-model and competition training.
        include_online_serving=str(layout != "multi_target").lower(),
        include_daily_rollout="true",
    )
    requirements = (project / "deployment/requirements.txt").read_text().splitlines()
    core = [Requirement(line) for line in requirements if line.startswith("skyulf-core")]
    assert len(core) == 1
    assert str(core[0]) == core_requirement
    assert core[0].extras == {"mlflow"}
    assert core[0].url is None
    assert not (project / "deployment/artifact.json").exists()
    assert not (project / "src/tools/build_wheel.py").exists()
    bundle = yaml.safe_load((project / "databricks.yml").read_text())
    assert "artifacts" not in bundle
    assert not any(".whl" in item for item in bundle["sync"]["include"])
    jobs = {}
    for resource in (project / "resources").glob("*.yml"):
        jobs.update(yaml.safe_load(resource.read_text())["resources"]["jobs"])
    online_serving = {"online_serving_setup"} if layout != "multi_target" else set()
    assert set(jobs) == {
        "train",
        "score",
        "monitoring",
        "online_publication",
        "daily_rollout",
        *online_serving,
    }
    for role, job in jobs.items():
        filename = (
            "train-requirements.txt"
            if role == "train"
            else "serving-requirements.txt"
            if role in {"online_publication", "online_serving_setup", "daily_rollout"}
            else "requirements.txt"
        )
        path = "${workspace.file_path}/deployment/" + filename
        if filename != "requirements.txt":
            assert "-r requirements.txt" in (project / "deployment" / filename).read_text()
        for environment in job.get("environments", []):
            dependencies = environment["spec"]["dependencies"]
            assert "-r " + path in dependencies
            assert not any(".whl" in item for item in dependencies)
        for task in job["tasks"]:
            if "libraries" in task:
                assert {"requirements": path} in task["libraries"]
                assert not any("whl" in item for item in task["libraries"])


@pytest.mark.skipif(
    os.environ.get("SKYULF_BUNDLE_OFFLINE_CLI") != "1" or not shutil.which("databricks"),
    reason="Requires explicitly enabled offline Bundle CLI generation.",
)
def test_publication_without_serving_inherits_feature_client_requirement(
    tmp_path, core_requirement
):
    """Publication needs its native feature client even when endpoint setup is disabled."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(
        tmp_path, include_online_publication="true", include_online_serving="false"
    )
    shared = (project / "deployment/requirements.txt").read_text()
    serving = (project / "deployment/serving-requirements.txt").read_text()
    assert core_requirement in shared.splitlines()
    assert "databricks-feature-engineering==0.18.1" not in shared
    assert "-r requirements.txt" in serving
    assert "databricks-feature-engineering==0.18.1" in serving
    resource = yaml.safe_load((project / "resources/online_publication.job.yml").read_text())
    dependencies = resource["resources"]["jobs"]["online_publication"]["environments"][0]["spec"][
        "dependencies"
    ]
    assert dependencies == ["-r ${workspace.file_path}/deployment/serving-requirements.txt"]


@pytest.mark.skipif(
    not (
        os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE")
        or os.environ.get("SKYULF_BUNDLE_OFFLINE_CLI") == "1"
    )
    or not shutil.which("databricks"),
    reason="Requires explicitly enabled real Bundle CLI generation.",
)
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
@pytest.mark.parametrize(
    "settings,expected,absent",
    [
        ({}, {"mlflow"}, {"optuna", "xgboost", "lightgbm"}),
        (
            {
                "branch_8_search_strategy": "optuna",
                "competition_classification_model_8": "xgboost_classifier",
            },
            {"mlflow"},
            {"optuna", "xgboost"},
        ),
        (
            {
                "sample_weight_enabled": "true",
                "classification_model_weighted": "xgboost_classifier",
            },
            {"xgboost"},
            {"lightgbm"},
        ),
        ({"search_strategy": "optuna"}, {"optuna", "optuna-integration", "cmaes"}, {"xgboost"}),
        ({"classification_model": "xgboost_classifier"}, {"xgboost"}, {"lightgbm"}),
        (
            {
                "training_layout": "multi_target",
                "branch_2_search_strategy": "optuna",
                "branch_2_regression_model": "lgbm_regressor",
            },
            {"optuna", "lightgbm"},
            {"xgboost"},
        ),
        (
            {
                "training_layout": "model_competition",
                "competition_candidate_count": "2",
                "competition_classification_model_2": "xgboost_classifier",
            },
            {"xgboost"},
            {"lightgbm"},
        ),
        (
            {
                "classification_model": "voting_classifier",
                "single_ensemble_classification_base_1": "lightgbm",
            },
            {"lightgbm"},
            {"xgboost"},
        ),
    ],
)
def test_selected_runtime_dependencies_reach_both_job_types(
    tmp_path, compute, settings, expected, absent
):
    """Selected optional features must install without borrowing packages from a prepared cluster."""
    from packaging.requirements import Requirement
    from test_databricks_bundle_generation import _generate_project, _read_jobs

    project = _generate_project(tmp_path, compute_mode=compute, task="classification", **settings)
    requirements = (project / "deployment/requirements.txt").read_text()
    names = {
        Requirement(line).name
        for line in requirements.splitlines()
        if line.strip() and not line.startswith(("#", "-r"))
    }
    assert names >= expected
    assert not names.intersection(absent)
    assert "-r ../src/features/requirements.txt" in requirements
    jobs = _read_jobs(project)
    for role, job in jobs.items():
        requirements_file = "train-requirements.txt" if role == "train" else "requirements.txt"
        requirements_path = "${workspace.file_path}/deployment/" + requirements_file
        if compute == "serverless" or role == "monitoring":
            dependencies = job["environments"][0]["spec"]["dependencies"]
            assert "-r " + requirements_path in dependencies
            assert not any(".whl" in item for item in dependencies)
        else:
            for task in job["tasks"]:
                if "notebook_task" in task:
                    assert {"requirements": requirements_path} in task["libraries"]
                    assert not any("whl" in item for item in task["libraries"])
    bundle = yaml.safe_load((project / "databricks.yml").read_text())
    assert "artifacts" not in bundle
