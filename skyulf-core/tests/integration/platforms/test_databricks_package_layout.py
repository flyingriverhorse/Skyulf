"""Runtime packages stay navigable without breaking deployed module references."""

import importlib
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
        ("training.fitting.local_retraining", "local_retraining"),
        ("training.tuning.local_cv", "local_cv"),
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


@pytest.mark.parametrize("legacy_first", [True, False])
@pytest.mark.parametrize("promotion_first", [True, False])
def test_legacy_import_order_preserves_class_identity(legacy_first, promotion_first):
    """Old notebooks and new runtime code must share one class in either order."""
    names = [f"{PACKAGE}.local_retraining", f"{PACKAGE}.training.fitting.local_retraining"]
    if not legacy_first:
        names.reverse()
    script = "import skyulf.integrations.mlflow.lifecycle.promotion\n" if promotion_first else ""
    script += (
        "import importlib, pickle\n"
        f"first = importlib.import_module({names[0]!r})\n"
        f"second = importlib.import_module({names[1]!r})\n"
        "assert first is second\n"
        "assert first.LocalTrainingSpec is second.LocalTrainingSpec\n"
        "legacy = pickle.loads(b'cskyulf.integrations.databricks.local_retraining"
        "\\nLocalTrainingSpec\\n.')\n"
        "assert legacy is first.LocalTrainingSpec\n"
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


def test_legacy_monkeypatch_and_pickled_class_use_canonical_module(monkeypatch):
    """Existing patch paths and pickles keep addressing the executing runtime."""
    legacy = importlib.import_module(f"{PACKAGE}.local_retraining")
    canonical = importlib.import_module(f"{PACKAGE}.training.fitting.local_retraining")
    sentinel = object()
    monkeypatch.setattr(legacy, "train_local_candidate", sentinel)
    restored = pickle.loads(
        b"cskyulf.integrations.databricks.local_retraining\nLocalTrainingSpec\n."
    )
    assert canonical.train_local_candidate is sentinel
    assert restored is canonical.LocalTrainingSpec
