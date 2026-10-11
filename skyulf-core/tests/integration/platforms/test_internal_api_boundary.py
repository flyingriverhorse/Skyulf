"""Integration consumers use declared helpers without reaching into module privates."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


def _is_module_import(path, root, node, name):
    """Recognize an actual source module, never a similarly named helper binding."""
    base = path.parents[node.level - 1] if node.level else root.parents[1]
    if node.module:
        base = base.joinpath(*node.module.split("."))
    return (base / (name.name + ".py")).is_file() or (base / name.name / "__init__.py").is_file()


def _private_imports(path, root):
    """Find private symbol imports and private access through imported modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_bindings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_bindings.update(
                name.asname or name.name for name in node.names if name.name.startswith("skyulf.")
            )
        if isinstance(node, ast.ImportFrom):
            module_bindings.update(
                name.asname or name.name
                for name in node.names
                if _is_module_import(path, root, node, name)
            )
    violations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and ast.unparse(node.value) in module_bindings
            and node.attr.startswith("_")
            and not node.attr.startswith("__")
        ):
            violations.append(f"{path.relative_to(root)}:{node.lineno}: {node.attr}")
        if isinstance(node, ast.ImportFrom):
            violations.extend(
                f"{path.relative_to(root)}:{node.lineno}: {name.name}"
                for name in node.names
                if name.name.startswith("_") and not _is_module_import(path, root, node, name)
            )
    return violations


def _helper_aliases(path, root):
    """Reject duplicate helper bindings while allowing canonical module imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
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
                if name.asname
                and name.asname.startswith("_")
                and not name.name.startswith("_")
                and not _is_module_import(path, root, node, name)
            )
    return violations


def test_integration_imports_do_not_borrow_private_symbols():
    """New orchestration must not silently expand another module's private API."""
    root = Path(__file__).parents[3] / "skyulf" / "integrations"
    violations = [
        item for path in sorted(root.rglob("*.py")) for item in _private_imports(path, root)
    ]
    assert not violations, "Undeclared private imports:\n" + "\n".join(violations)


def test_shared_helpers_have_one_definition_name():
    """Private/public aliases must not leave monkeypatches targeting an inactive binding."""
    root = Path(__file__).parents[3] / "skyulf"
    paths = list((root / "integrations").rglob("*.py"))
    paths += [
        root / "inference/bundle.py",
        root / "inference/fitted_pipeline.py",
        root / "modeling/_tuning/cv_policy.py",
    ]
    violations = [item for path in paths for item in _helper_aliases(path, root / "integrations")]
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    ("source", "private", "alias"),
    [
        ("from skyulf.integrations import _owner as owner\nowner.shared()", False, False),
        ("from . import named as _implementation", False, False),
        ("from skyulf.integrations import named as _implementation", False, False),
        ("from . import _private_package", False, False),
        ("from ._owner import _private", True, False),
        ("from . import _package_helper", True, False),
        ("from ._owner import shared as _shared", False, True),
        ("from . import _owner as owner\nowner._private()", True, False),
        ("import skyulf.integrations._owner as owner\nowner._private()", True, False),
        ("from . import _private_package\n_private_package._private()", True, False),
        ("from ._owner import shared\n_private = shared", False, True),
    ],
)
def test_boundary_scanners_distinguish_modules_from_helper_bindings(
    tmp_path, source, private, alias
):
    """Compatibility imports must not create exemptions for borrowed or aliased functions."""
    root = tmp_path / "skyulf" / "integrations"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("_package_helper = None\n", encoding="utf-8")
    for name in ("_owner.py", "named.py"):
        (root / name).write_text("def shared(): pass\ndef _private(): pass\n", encoding="utf-8")
    package = root / "_private_package"
    package.mkdir()
    (package / "__init__.py").write_text("def _private(): pass\n", encoding="utf-8")
    path = root / "consumer.py"
    path.write_text(source, encoding="utf-8")
    assert (bool(_private_imports(path, root)), bool(_helper_aliases(path, root))) == (
        private,
        alias,
    )


@pytest.mark.parametrize(
    "first",
    [
        "skyulf.integrations.mlflow.lifecycle.promotion",
        "skyulf.integrations.databricks.jobs.training.training_nodes",
        "skyulf.integrations.databricks.lifecycle.workflow",
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
from skyulf.integrations.mlflow.shared._client import require_mlflow
from skyulf.integrations.mlflow.registration.registry import RegistryDependencyError
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
