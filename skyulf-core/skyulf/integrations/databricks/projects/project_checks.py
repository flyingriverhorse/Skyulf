"""Credential-free project checks that never import or execute project hooks."""

import ast
import json
from pathlib import Path

from ..lifecycle.local_workflow import resolve_target_config
from .workflow_config import validate_project_settings


def check_project(project_path: str | Path, bindings: dict[str, str]) -> dict[str, object]:
    """Read workflow settings and Python syntax without training or external I/O.

    This static smoke check deliberately does not execute editable recipes. It
    cannot certify hook behavior, source data, cloud permissions or fitted models.
    Those checks require separate runtime verification; sampling is not a no-write mode.
    """
    root = Path(project_path).resolve(strict=True)
    config = json.loads((root / "config/workflow.json").read_text(encoding="utf-8"))
    resolved = resolve_target_config(config, bindings)
    validate_project_settings(resolved)
    sources = sorted((root / "src").rglob("*.py"))
    if not sources:
        raise ValueError("Project src must contain Python source files.")
    for path in sources:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Project source must remain within the project directory.")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        "status": "passed",
        "scope": "static_config_and_python_syntax",
        "python_files": len(sources),
        "training_layout": config.get("training_layout", "single_model"),
        "project_hooks_executed": False,
        "remote_operations": False,
    }
