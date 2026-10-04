"""Import trusted saved project packages without an editable filesystem directory."""

import importlib.abc
import importlib.util
import sys
from base64 import b64decode
from types import ModuleType
from typing import Any

from .project_dependencies import verify_project_requirements


class _SavedSourceLoader(importlib.abc.Loader):
    """Execute one module from the immutable source attached to its model version."""

    def __init__(self, source: str, filename: str) -> None:
        self.source = source
        self.filename = filename

    def create_module(self, spec: Any) -> None:
        """Use Python's standard module construction."""
        return None

    def exec_module(self, module: ModuleType) -> None:
        """Compile the saved bytes with their diagnostic path, without reading disk."""
        module.__file__ = self.filename
        exec(compile(self.source, self.filename, "exec"), module.__dict__)  # nosec B102 - trusted model source


class _SavedPackageFinder(importlib.abc.MetaPathFinder):
    """Resolve only digest-qualified packages explicitly installed by model loading."""

    def __init__(self) -> None:
        self.packages: dict[str, dict[str, str]] = {}
        self.assets: dict[str, dict[str, bytes]] = {}
        self.requirements: dict[str, tuple[str, ...]] = {}

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        """Leave unrelated imports to Python's normal finders."""
        root, separator, suffix = fullname.partition(".")
        files = self.packages.get(root)
        if files is None or not separator:
            return None
        stem = suffix.replace(".", "/")
        package_path = stem + "/__init__.py"
        is_package = package_path in files
        relative = package_path if is_package else stem + ".py"
        if relative not in files:
            return None
        loader = _SavedSourceLoader(files[relative], f"<{root}/{relative}>")
        return importlib.util.spec_from_loader(fullname, loader, is_package=is_package)


_FINDER = _SavedPackageFinder()


def install_project_package(
    name: str,
    files: dict[str, str],
    *,
    assets: dict[str, str] | None = None,
    requirements: tuple[str, ...] = (),
) -> None:
    """Populate a digest-specific root module and enable its relative imports.

    Called by a generated source snapshot, already verified by the model's source
    digest. Third-party imports still require installed environment dependencies.
    """
    verify_project_requirements(requirements)
    module = sys.modules[name]
    module.__package__ = name
    module.__path__ = []
    _FINDER.packages[name] = dict(files)
    _FINDER.assets[name] = {
        relative: b64decode(encoded, validate=True) for relative, encoded in (assets or {}).items()
    }
    _FINDER.requirements[name] = requirements
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    _SavedSourceLoader(files["__init__.py"], f"<{name}/__init__.py>").exec_module(module)


def read_project_asset(package: str, relative: str) -> bytes:
    """Read declared immutable bytes by package-root-relative path.

    Pass ``__package__`` from a saved feature module. This also works inside
    nested modules; paths always start at the feature package root. Filesystem
    paths, undeclared files and edits to the original checkout are never read.
    """
    root = package.partition(".")[0]
    try:
        return _FINDER.assets[root][relative]
    except KeyError as exc:
        raise ValueError(
            f"Project asset is not declared in the saved package: {relative}."
        ) from exc


def get_project_requirements(package: str) -> tuple[str, ...]:
    """Return saved exact pins for an installed feature package or legacy module."""
    return _FINDER.requirements.get(package.partition(".")[0], ())


def discard_project_package(name: str) -> None:
    """Remove a partially imported package after a failed trusted-source load."""
    _FINDER.packages.pop(name, None)
    _FINDER.assets.pop(name, None)
    _FINDER.requirements.pop(name, None)
    for module_name in tuple(sys.modules):
        if module_name == name or module_name.startswith(name + "."):
            sys.modules.pop(module_name, None)
