"""MLflow responsibility packages retain saved artifacts and lazy optional imports."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = "skyulf.integrations.mlflow"
ROOT = Path(importlib.import_module(PACKAGE).__file__).resolve().parent


@pytest.mark.parametrize(
    "canonical, legacy",
    [
        ("models.local_model", "local_model"),
        ("spark.spark_model", "spark_model"),
        ("lifecycle.promotion", "promotion"),
        ("registration.registry", "registry"),
        ("runs.tracking", "tracking"),
        ("shared._client", "_client"),
    ],
)
def test_nested_mlflow_modules_share_legacy_identity(canonical, legacy):
    """Old notebooks and canonical callers must patch the same executing globals."""
    pytest.importorskip("mlflow")
    assert importlib.import_module(f"{PACKAGE}.{canonical}") is importlib.import_module(
        f"{PACKAGE}.{legacy}"
    )


def test_mlflow_root_is_a_small_optional_entrypoint():
    """Responsibility groups must keep implementation out of the package root."""
    assert {path.name for path in ROOT.glob("*.py")} == {"__init__.py"}


def test_all_mlflow_aliases_preserve_module_and_saved_class_identity():
    """Pickled public adapter classes must resolve from their released module names."""
    pytest.importorskip("mlflow")
    import pickle

    aliases = [path.stem for path in (ROOT / "_compat").glob("*.py") if path.stem != "__init__"]
    assert len(aliases) == 17
    classes = []
    for name in aliases:
        legacy = importlib.import_module(f"{PACKAGE}.{name}")
        assert legacy is importlib.import_module(legacy.__name__)
        for class_name, value in vars(legacy).items():
            if isinstance(value, type) and value.__module__ == legacy.__name__:
                restored = pickle.loads(f"c{PACKAGE}.{name}\n{class_name}\n.".encode())
                assert restored is value
                classes.append(class_name)
    assert {"SkyulfPythonModel", "SkyulfLocalPythonModel", "ResolvedModel"} <= set(classes)


@pytest.mark.parametrize("legacy_first", [False, True])
def test_mlflow_alias_import_order_stays_optional(legacy_first):
    """Import order cannot pull MLflow into tracking/registry-only package imports."""
    names = [f"{PACKAGE}.tracking", f"{PACKAGE}.runs.tracking"]
    if not legacy_first:
        names.reverse()
    script = """
import importlib
import importlib.abc
import sys
class BlockMlflow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlflow' or fullname.startswith('mlflow.'):
            raise ModuleNotFoundError('optional MLflow intentionally unavailable')
sys.meta_path.insert(0, BlockMlflow())
"""
    script += f"first = importlib.import_module({names[0]!r})\n"
    script += f"second = importlib.import_module({names[1]!r})\n"
    script += "assert first is second\nassert 'mlflow' not in sys.modules\n"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT.parents[2])
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
