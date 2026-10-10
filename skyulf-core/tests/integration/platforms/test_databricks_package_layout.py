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
def test_responsibility_subpackages_preserve_legacy_identity(canonical, legacy):
    """Deeper package boundaries must not fork objects used by deployed clients."""
    assert importlib.import_module(f"{PACKAGE}.{canonical}") is importlib.import_module(
        f"{PACKAGE}.{legacy}"
    )


def test_runtime_root_contains_only_package_entrypoint():
    """Operators can find runtime responsibilities without scanning flat modules."""
    assert {path.name for path in PACKAGE_ROOT.glob("*.py")} == {"__init__.py"}
    assert {path.name for path in PACKAGE_ROOT.iterdir() if path.is_dir()} >= GROUPS


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


def test_all_legacy_modules_resolve_to_canonical_modules():
    """Compatibility imports must share module globals used by monkeypatch callers."""
    aliases = list((PACKAGE_ROOT / "_compat").glob("*.py"))
    assert aliases
    for alias in aliases:
        if alias.stem == "__init__":
            continue
        legacy = importlib.import_module(f"{PACKAGE}.{alias.stem}")
        assert legacy.__name__.removeprefix(PACKAGE + ".").split(".")[0] in GROUPS
        canonical = importlib.import_module(legacy.__name__)
        assert legacy is canonical


def test_retired_training_pickle_path_is_not_redirected():
    """Old saved class paths fail clearly rather than silently reviving removed modules."""
    with pytest.raises(ModuleNotFoundError) as error:
        pickle.loads(b"cskyulf.integrations.databricks.local_retraining\nLocalTrainingSpec\n.")
    assert error.value.name == f"{PACKAGE}.local_retraining"


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
