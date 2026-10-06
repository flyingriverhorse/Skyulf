"""Exercise generated dependency installation inputs and checked wheel preparation."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"


def _wheel(path: Path, version: str = "0.9.1", name: str = "skyulf-core") -> Path:
    """Create a metadata fixture without depending on the installed development package."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"skyulf_core-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
    return path


def _build_project(tmp_path: Path, source: Path, version: str = "0.9.1") -> Path:
    """Copy the actual shipped build command into a project with explicit artifact inputs."""
    project = tmp_path / "project"
    (project / "src/tools").mkdir(parents=True)
    (project / "deployment").mkdir()
    shutil.copyfile(TEMPLATE / "src/tools/build_wheel.py", project / "src/tools/build_wheel.py")
    (project / "deployment/artifact.json").write_text(
        json.dumps({"source": str(source), "version": version}), encoding="utf-8"
    )
    return project


def _build(project: Path) -> subprocess.CompletedProcess[str]:
    """Execute the generated command outside the repository's import context."""
    return subprocess.run(
        [sys.executable, "-I", str(project / "src/tools/build_wheel.py")],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_build_copies_matching_wheel_and_records_its_digest(tmp_path):
    """A deploy must use the declared wheel bytes, with an independently verifiable receipt."""
    import hashlib

    source = _wheel(tmp_path / "release/skyulf_core-0.9.1-py3-none-any.whl")
    project = _build_project(tmp_path, source)
    result = _build(project)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((project / "dist/skyulf/build.json").read_text())
    output = project / "dist/skyulf" / receipt["filename"]
    assert output.read_bytes() == source.read_bytes()
    assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["version"] == "0.9.1"


@pytest.mark.parametrize("suffix", [".py", ".pyc"])
def test_reorganized_wheel_rejects_shadowing_build_cache(tmp_path, suffix):
    """Old build files must not shadow the relocated runtime compatibility aliases."""
    import runpy

    source = _wheel(tmp_path / "skyulf_core-0.9.1-py3-none-any.whl")
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("skyulf/integrations/databricks/jobs/shared/job_runtime.py", "")
        archive.writestr(f"skyulf/integrations/databricks/job_runtime{suffix}", "")
    validate = runpy.run_path(str(TEMPLATE / "src/tools/build_wheel.py"))["validate_wheel"]
    with pytest.raises(ValueError, match="stale flat Databricks modules"):
        validate(source, "0.9.1")


def test_reorganized_wheel_keeps_private_compatibility_aliases(tmp_path):
    """A clean package with compatibility aliases must remain deployable."""
    import runpy

    source = _wheel(tmp_path / "skyulf_core-0.9.1-py3-none-any.whl")
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("skyulf/integrations/databricks/__init__.py", "")
        archive.writestr("skyulf/integrations/databricks/jobs/shared/job_runtime.py", "")
        archive.writestr("skyulf/integrations/databricks/_compat/job_runtime.py", "")
    validate = runpy.run_path(str(TEMPLATE / "src/tools/build_wheel.py"))["validate_wheel"]
    assert validate(source, "0.9.1") is None


@pytest.mark.parametrize("suffix", [".py", ".pyc"])
def test_nested_wheel_rejects_previous_group_build_cache(tmp_path, suffix):
    """Intermediate build files must not create a second copy of runtime globals."""
    import runpy

    source = _wheel(tmp_path / "skyulf_core-0.9.1-py3-none-any.whl")
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("skyulf/integrations/databricks/jobs/shared/job_runtime.py", "")
        archive.writestr(f"skyulf/integrations/databricks/jobs/job_runtime{suffix}", "")
    validate = runpy.run_path(str(TEMPLATE / "src/tools/build_wheel.py"))["validate_wheel"]
    with pytest.raises(ValueError, match="stale duplicate Databricks modules"):
        validate(source, "0.9.1")


@pytest.mark.parametrize("suffix", [".py", ".pyc"])
def test_reorganized_mlflow_wheel_rejects_shadowing_build_cache(tmp_path, suffix):
    """Stale MLflow adapters must not bypass the released-path compatibility aliases."""
    import runpy

    source = _wheel(tmp_path / "skyulf_core-0.9.1-py3-none-any.whl")
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("skyulf/integrations/mlflow/runs/tracking.py", "")
        archive.writestr(f"skyulf/integrations/mlflow/tracking{suffix}", "")
    validate = runpy.run_path(str(TEMPLATE / "src/tools/build_wheel.py"))["validate_wheel"]
    with pytest.raises(ValueError, match="stale flat MLflow modules"):
        validate(source, "0.9.1")


def test_reorganized_mlflow_wheel_accepts_private_aliases(tmp_path):
    """Compatibility modules belong in a clean wheel alongside canonical adapters."""
    import runpy

    source = _wheel(tmp_path / "skyulf_core-0.9.1-py3-none-any.whl")
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("skyulf/integrations/mlflow/runs/tracking.py", "")
        archive.writestr("skyulf/integrations/mlflow/_compat/tracking.py", "")
    validate = runpy.run_path(str(TEMPLATE / "src/tools/build_wheel.py"))["validate_wheel"]
    assert validate(source, "0.9.1") is None


def test_changed_release_bytes_get_distinct_deployment_identity(tmp_path):
    """Same-version code changes must invalidate caches without breaking saved model requirements."""
    from packaging.utils import parse_wheel_filename

    source = _wheel(tmp_path / "release/skyulf_core-0.9.1-py3-none-any.whl")
    project = _build_project(tmp_path, source)
    first = _build(project)
    assert first.returncode == 0, first.stderr
    old = json.loads(first.stdout)
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("skyulf/probe.py", "VALUE = 2\n")
    second = _build(project)
    assert second.returncode == 0, second.stderr
    new = json.loads(second.stdout)
    assert new["filename"] != old["filename"]
    _, version, build_tag, _ = parse_wheel_filename(new["filename"])
    assert str(version) == "0.9.1" and build_tag
    assert not (project / "dist/skyulf" / old["filename"]).exists()
    third = _build(project)
    assert third.returncode == 0, third.stderr
    assert json.loads(third.stdout) == new


@pytest.mark.parametrize("version,name", [("0.9.0", "skyulf-core"), ("0.9.1", "foreign")])
def test_build_rejects_wrong_release_before_replacing_existing_output(tmp_path, version, name):
    """A mislabeled wheel must not replace an already prepared deployment artifact."""
    source = _wheel(tmp_path / "release/skyulf_core-0.9.1-py3-none-any.whl", version, name)
    project = _build_project(tmp_path, source)
    prior = _wheel(project / "dist/skyulf/skyulf_core-0.9.1-py3-none-any.whl")
    original = prior.read_bytes()
    result = _build(project)
    assert result.returncode != 0
    assert "wheel" in result.stderr.lower()
    assert prior.read_bytes() == original


def test_build_rejects_source_inside_owned_output_without_deleting_it(tmp_path):
    """Rebuilding must not remove the release input needed by the next deployment."""
    source = _wheel(tmp_path / "project/dist/skyulf/skyulf_core-0.9.1-py3-none-any.whl")
    project = _build_project(tmp_path, source)
    original = source.read_bytes()
    result = _build(project)
    assert result.returncode != 0
    assert "outside dist/skyulf" in result.stderr
    assert source.read_bytes() == original


@pytest.mark.skipif(
    not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE") or not shutil.which("databricks"),
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
            assert "../dist/skyulf/*.whl" in dependencies
        else:
            for task in job["tasks"]:
                if "notebook_task" in task:
                    assert {"requirements": requirements_path} in task["libraries"]
                    assert {"whl": "../dist/skyulf/*.whl"} in task["libraries"]
    bundle = yaml.safe_load((project / "databricks.yml").read_text())
    assert bundle["artifacts"]["skyulf"]["type"] == "whl"
    assert not bundle["artifacts"]["skyulf"].get("dynamic_version", False)
    assert "build_wheel.py" in bundle["artifacts"]["skyulf"]["build"]
