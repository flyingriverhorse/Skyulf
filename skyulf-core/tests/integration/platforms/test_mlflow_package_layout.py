"""MLflow responsibility packages retain saved artifacts and lazy optional imports."""

import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = "skyulf.integrations.mlflow"
ROOT = Path(importlib.import_module(PACKAGE).__file__).resolve().parent


@pytest.mark.parametrize(
    "canonical, legacy",
    [
        ("spark.spark_model", "spark_model"),
        ("lifecycle.promotion", "promotion"),
        ("registration.registry", "registry"),
        ("runs.tracking", "tracking"),
        ("shared._client", "_client"),
    ],
)
def test_nested_mlflow_modules_replace_flat_imports(canonical, legacy):
    """Canonical imports stay usable without retaining the retired module names."""
    assert importlib.import_module(f"{PACKAGE}.{canonical}").__name__ == f"{PACKAGE}.{canonical}"
    assert importlib.util.find_spec(f"{PACKAGE}.{legacy}") is None


def test_mlflow_root_is_a_small_optional_entrypoint():
    """Responsibility groups must keep implementation out of the package root."""
    assert {path.name for path in ROOT.glob("*.py")} == {"__init__.py"}
    assert not list((ROOT / "_compat").glob("*.py"))
    assert all(Path(path).name != "_compat" for path in importlib.import_module(PACKAGE).__path__)


def test_canonical_mlflow_classes_preserve_saved_identity():
    """Current adapter class pickles must resolve directly to their owning implementation."""
    pytest.importorskip("mlflow")
    modules = [
        "shared._client",
        "shared._model_metadata",
        "shared._nullable_transport",
        "spark._spark_environment",
        "spark._spark_output",
        "lifecycle.challenger",
        "models.model",
        "models.model_set",
        "lifecycle.model_set_challenger",
        "lifecycle.model_set_lifecycle",
        "lifecycle.promotion",
        "registration.registry",
        "lifecycle.rejection",
        "spark.spark_model",
        "runs.tracking",
        "lifecycle.validation",
    ]
    classes = []
    for name in modules:
        module = importlib.import_module(f"{PACKAGE}.{name}")
        for class_name, value in vars(module).items():
            if isinstance(value, type) and value.__module__ == module.__name__:
                assert pickle.loads(pickle.dumps(value)) is value
                classes.append(class_name)
    assert {"SkyulfPythonModel", "SkyulfModelSetPythonModel", "ResolvedModel"} <= set(classes)


@pytest.mark.parametrize(
    "retired",
    [
        "_client",
        "_model_metadata",
        "_nullable_transport",
        "_spark_environment",
        "_spark_output",
        "challenger",
        "model",
        "model_set",
        "model_set_challenger",
        "model_set_lifecycle",
        "promotion",
        "registry",
        "rejection",
        "spark_model",
        "tracking",
        "validation",
    ],
)
def test_retired_mlflow_flat_modules_do_not_resolve(retired):
    """A removed import must fail even when its optional dependencies are installed."""
    name = f"{PACKAGE}.{retired}"
    assert importlib.util.find_spec(name) is None
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(name)
    assert error.value.name == name


@pytest.mark.parametrize(
    "module, name",
    [
        ("model", "SkyulfPythonModel"),
        ("model_set", "SkyulfModelSetPythonModel"),
        ("registry", "ResolvedModel"),
    ],
)
def test_retired_model_pickle_globals_fail_before_context_loading(module, name):
    """Old outer adapter pickles can fail before the inner artifact runtime guard executes."""
    with pytest.raises(ModuleNotFoundError) as error:
        pickle.loads(f"c{PACKAGE}.{module}\n{name}\n.".encode())
    assert error.value.name == f"{PACKAGE}.{module}"


@pytest.mark.parametrize("root_first", [False, True])
def test_mlflow_import_order_stays_optional(root_first):
    """Import order cannot pull MLflow into tracking/registry-only package imports."""
    names = [PACKAGE, f"{PACKAGE}.runs.tracking"]
    if not root_first:
        names.reverse()
    script = """
import importlib
import importlib.util
import pickle
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
    script += (
        "assert first.TrackingConfig is second.TrackingConfig\nassert 'mlflow' not in sys.modules\n"
    )
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
