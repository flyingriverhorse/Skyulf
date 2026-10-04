"""Integration consumers use declared helpers without reaching into module privates."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


def test_integration_imports_do_not_borrow_private_symbols():
    """New orchestration must not silently expand another module's private API."""
    root = Path(__file__).parents[3] / "skyulf" / "integrations"
    violations = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_bindings = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_bindings.update(
                    name.asname or name.name
                    for name in node.names
                    if name.name.startswith("skyulf.")
                )
            if not isinstance(node, ast.ImportFrom):
                continue
            base = path.parents[node.level - 1] if node.level else root.parents[1]
            if node.module:
                base = base.joinpath(*node.module.split("."))
            for name in node.names:
                if (base / (name.name + ".py")).is_file():
                    module_bindings.add(name.asname or name.name)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and ast.unparse(node.value) in module_bindings
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
            ):
                violations.append(f"{path.relative_to(root)}:{node.lineno}: {node.attr}")
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            violations.extend(
                f"{path.relative_to(root)}:{node.lineno}: {name.name}"
                for name in node.names
                if name.name.startswith("_")
            )
    assert not violations, "Undeclared private imports:\n" + "\n".join(violations)


def test_shared_helpers_have_one_definition_name():
    """Private/public aliases must not leave monkeypatches targeting an inactive binding."""
    root = Path(__file__).parents[3] / "skyulf"
    paths = list((root / "integrations").rglob("*.py"))
    paths += [
        root / "inference/bundle.py",
        root / "inference/local_pipeline.py",
        root / "modeling/_tuning/cv_policy.py",
    ]
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Name)
                and not node.value.id.isupper()
            ):
                violations.extend(
                    f"{path.name}:{node.lineno}: duplicate binding"
                    for target in node.targets
                    if isinstance(target, ast.Name)
                    and target.id.startswith("_") != node.value.id.startswith("_")
                )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.level or (node.module or "").startswith("skyulf.")
            ):
                violations.extend(
                    f"{path.name}:{node.lineno}: private import alias"
                    for name in node.names
                    if name.asname and name.asname.startswith("_") and not name.name.startswith("_")
                )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    "first",
    [
        "skyulf.integrations.mlflow.promotion",
        "skyulf.integrations.databricks.training_nodes",
        "skyulf.integrations.databricks.local_workflow",
    ],
)
def test_internal_helpers_preserve_optional_imports(first):
    """Import order must not introduce an eager MLflow dependency or a cycle."""
    code = f"""
import importlib
import importlib.abc
import sys

class BlockMlflow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlflow' or fullname.startswith('mlflow.'):
            raise ModuleNotFoundError('MLflow intentionally unavailable')

sys.meta_path.insert(0, BlockMlflow())
importlib.import_module({first!r})
from skyulf.integrations.mlflow._client import require_mlflow
from skyulf.integrations.mlflow.registry import RegistryDependencyError
try:
    require_mlflow()
except RegistryDependencyError as exc:
    assert str(exc) == "MLflow registry support requires the optional 'mlflow' extra."
else:
    raise AssertionError('Missing optional dependency was accepted')
assert 'mlflow' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
