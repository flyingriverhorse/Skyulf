"""Credential-free project checks that never import or execute project hooks."""

import ast
from pathlib import Path

from ..lifecycle.local_workflow import resolve_target_config
from ._project_files import project_source
from .workflow_config import validate_project_settings
from .yaml_config import read_training_config, read_workflow_config
from .yaml_models import static_workflows


def _check_config(root: Path, bindings: dict[str, str]) -> dict:
    """Attach the editable file path to invalid or incomplete workflow settings."""
    try:
        config = read_workflow_config(root / "config/workflow.json")
        if not isinstance(config, dict):
            raise ValueError("Workflow configuration must be an object.")
        yaml = read_training_config(root / "config")
        configs = static_workflows(yaml, config) if yaml is not None else [config]
        checked = [
            validate_project_settings(resolve_target_config(item, bindings)) for item in configs
        ]
        return checked[0]
    except KeyError as exc:
        raise ValueError(f"config/workflow.json: missing required setting {exc}.") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"config/workflow.json: {exc}") from exc


def _check_packages(root: Path) -> list[str]:
    """Reuse training's inert snapshot validation without importing project packages."""
    checked = []
    for relative in ("src/features", "src/composition"):
        path = root / relative
        if not path.exists() and not path.is_symlink():
            continue
        try:
            if not path.is_dir() or not path.resolve().is_relative_to(root):
                raise ValueError("Project package must be a directory inside the project.")
            project_source(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{relative}: {exc}") from exc
        checked.append(relative)
    return checked


def check_project(project_path: str | Path, bindings: dict[str, str]) -> dict[str, object]:
    """Check settings, syntax and package snapshots without training or external I/O.

    This static smoke check deliberately does not execute editable recipes. It
    cannot certify hook behavior, source data, cloud permissions or fitted models.
    Those checks require runtime verification; sampling is not a no-write mode.
    Present features/composition packages use the same snapshot checks as training;
    declared dependencies are parsed, never installed or imported.
    """
    root = Path(project_path).resolve(strict=True)
    config = _check_config(root, bindings)
    sources = sorted((root / "src").rglob("*.py"))
    if not sources:
        raise ValueError("Project src must contain Python source files.")
    for path in sources:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"{path}: project source must remain within the project directory.")
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path.relative_to(root).as_posix()}: {exc}") from exc
    return {
        "status": "passed",
        "scope": "static_config_and_python_syntax",
        "python_files": len(sources),
        "project_packages": _check_packages(root),
        "training_layout": config.get("training_layout", "single_model"),
        "project_hooks_executed": False,
        "remote_operations": False,
    }
