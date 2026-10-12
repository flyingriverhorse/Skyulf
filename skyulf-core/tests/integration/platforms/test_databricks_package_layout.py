"""Runtime modules own their implementation and serialize current canonical names."""

import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = "skyulf.integrations.databricks"
PACKAGE_ROOT = Path(importlib.import_module(PACKAGE).__file__).resolve().parent
GROUPS = {
    "shared",
    "jobs",
    "training",
    "scoring",
    "observability",
    "model_sets",
    "projects",
    "data",
    "lifecycle",
}


@pytest.mark.parametrize(
    "canonical, legacy",
    [
        ("jobs.shared.job_runtime", "job_runtime"),
        ("jobs.training.training_nodes", "training_nodes"),
        ("observability.monitoring.spark.spark_monitoring_metrics", "spark_monitoring_metrics"),
        ("observability.monitoring.performance.performance_policy", "performance_policy"),
        ("shared._contracts", "_contracts"),
        ("data.delta_io.delta", "delta"),
        ("scoring.batch.batch", "batch"),
    ],
)
def test_responsibility_subpackages_replace_flat_imports(canonical, legacy):
    """The direct implementation remains importable after its flat alias is retired."""
    assert importlib.import_module(f"{PACKAGE}.{canonical}").__name__ == f"{PACKAGE}.{canonical}"
    assert importlib.util.find_spec(f"{PACKAGE}.{legacy}") is None


def test_runtime_root_contains_only_package_entrypoint():
    """Operators can find runtime responsibilities without scanning flat modules."""
    assert {path.name for path in PACKAGE_ROOT.glob("*.py")} == {"__init__.py"}
    assert {path.name for path in PACKAGE_ROOT.iterdir() if path.is_dir()} >= GROUPS
    assert not list((PACKAGE_ROOT / "_compat").glob("*.py"))
    assert all(Path(path).name != "_compat" for path in importlib.import_module(PACKAGE).__path__)


@pytest.mark.parametrize("promotion_first", [True, False])
def test_canonical_import_order_preserves_pickled_class_identity(promotion_first):
    """Canonical imports and serialized classes must work in fresh processes in either order."""
    script = "import skyulf.integrations.mlflow.lifecycle.promotion\n" if promotion_first else ""
    script += (
        "import importlib, pickle\n"
        f"module = importlib.import_module('{PACKAGE}.training.fitting.candidate')\n"
        f"from {PACKAGE} import TrainingSpec\n"
        "assert module.TrainingSpec is TrainingSpec\n"
        "assert TrainingSpec.__module__ == module.__name__\n"
        "assert pickle.loads(pickle.dumps(TrainingSpec)) is TrainingSpec\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PACKAGE_ROOT.parents[2])
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "retired",
    [
        "_batch_manifest",
        "_contracts",
        "_lifecycle_data",
        "_lifecycle_state",
        "_project_files",
        "_project_recipes",
        "admission",
        "batch",
        "branch_notebook",
        "branch_tasks",
        "cdf_recovery",
        "competition_evaluation",
        "competition_project",
        "competition_training",
        "decision_thresholds",
        "delta",
        "delta_admission",
        "evaluation_chart_data",
        "evaluation_chart_report",
        "evaluation_chart_runs",
        "evaluation_chart_task",
        "evaluation_charts",
        "explanation_report",
        "job_output",
        "job_runtime",
        "lifecycle_tasks",
        "model_set_batch",
        "model_set_output",
        "model_set_project",
        "model_set_quality",
        "model_set_release",
        "model_set_stages",
        "monitoring_config",
        "monitoring_model_set",
        "monitoring_output",
        "monitoring_reference",
        "monitoring_registration",
        "monitoring_source_evidence",
        "monitoring_sources",
        "monitoring_store",
        "monitoring_tasks",
        "notebook_diagnostics",
        "performance_actions",
        "performance_policy",
        "performance_retraining",
        "prediction_output",
        "project",
        "project_checks",
        "retraining_data",
        "retraining_requests",
        "retraining_task",
        "scoring_pre_split",
        "scoring_recovery",
        "spark_monitoring_drift",
        "spark_monitoring_job",
        "spark_monitoring_metrics",
        "spark_monitoring_probability",
        "spark_monitoring_quality",
        "spark_monitoring_reference",
        "spark_monitoring_sources",
        "spark_monitoring_windows",
        "spark_retraining_data",
        "spark_scoring",
        "threshold_cv",
        "threshold_training",
        "training_dates",
        "training_node_notebook",
        "training_node_output",
        "training_nodes",
        "training_parameters",
        "weight_config",
        "workflow_config",
    ],
)
def test_retired_flat_modules_do_not_resolve(retired):
    """Removed notebook import paths must fail instead of silently routing to another module."""
    name = f"{PACKAGE}.{retired}"
    assert importlib.util.find_spec(name) is None
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(name)
    assert error.value.name == name


@pytest.mark.parametrize(
    "module, name",
    [("local_retraining", "LocalTrainingSpec"), ("monitoring_config", "MonitorConfig")],
)
def test_retired_training_pickle_path_is_not_redirected(module, name):
    """Old saved class paths fail clearly rather than silently reviving removed modules."""
    with pytest.raises(ModuleNotFoundError) as error:
        pickle.loads(f"c{PACKAGE}.{module}\n{name}\n.".encode())
    assert error.value.name == f"{PACKAGE}.{module}"


@pytest.mark.parametrize(
    "canonical, retired",
    [
        ("shared._frames", "shared._local_frames"),
        ("observability.monitoring.monitoring", "observability.monitoring.local.monitoring"),
        (
            "observability.monitoring.monitoring_metrics",
            "observability.monitoring.local.monitoring_metrics",
        ),
        (
            "observability.monitoring.monitoring_performance",
            "observability.monitoring.local.monitoring_performance",
        ),
    ],
)
def test_shared_frame_and_monitoring_modules_have_direct_names(canonical, retired):
    """Shared pandas/Polars and Spark helpers must not hide behind local-only paths."""
    name = f"{PACKAGE}.{canonical}"
    assert importlib.util.find_spec(name) is not None
    module = importlib.import_module(name)
    assert module.__name__ == name
    basename = retired.rsplit(".", 1)[-1]
    for suffix in (retired, basename, f"_compat.{basename}"):
        old_name = f"{PACKAGE}.{suffix}"
        with pytest.raises(ModuleNotFoundError) as error:
            importlib.import_module(old_name)
        assert error.value.name and (
            old_name == error.value.name or old_name.startswith(error.value.name + ".")
        )
