"""Credential-free project checks that never import or execute project hooks."""

import ast
from pathlib import Path
from typing import Any

from ..jobs.serving_endpoint_job import read_serving_endpoint_settings
from ..jobs.serving_job import validate_serving_section
from ..lifecycle.workflow import resolve_target_config
from ..training.tuning.cv import CVSpec
from ..training.tuning.search import prepare_search_pipeline
from ._project_files import project_source
from .workflow_config import validate_project_settings
from .yaml_config import (
    project_config_path,
    read_training_config,
    read_workflow_config,
    read_yaml_mapping,
)
from .yaml_models import static_workflows
from .yaml_recipes import feature_project_source

_SERVING_SECTIONS = ("online_feature_lookup", "rollout", "online_publication")


def _check_search(checked: dict[str, Any]) -> None:
    """Run preview's search and competition-metric admission on static settings."""
    if checked.get("training_layout") == "model_competition":
        from ..training.competition.competition_evaluation import (  # noqa: PLC0415
            competition_metric,
        )

        competition_metric(checked["metric"], checked["task"])
    prepare_search_pipeline(
        checked["pipeline"],
        CVSpec.from_workflow(checked),
        target_column=checked["target_column"],
        event_column=checked.get("event_column"),
    )


def _check_config(root: Path, bindings: dict[str, str]) -> dict:
    """Attach the editable file path to invalid or incomplete workflow settings."""
    path = project_config_path(root)
    try:
        config = read_workflow_config(path)
        if not isinstance(config, dict):
            raise ValueError("Workflow configuration must be an object.")
        yaml = read_training_config(root / "config")
        configs = static_workflows(yaml, config) if yaml is not None else [config]
        checked = [
            validate_project_settings(resolve_target_config(item, bindings)) for item in configs
        ]
        for item in checked:
            _check_search(item)
        return checked[0]
    except KeyError as exc:
        raise ValueError(f"config/{path.name}: missing required setting {exc}.") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"config/{path.name}: {exc}") from exc


def _check_permission_targets(root: Path, settings: Any) -> None:
    """Catch mistyped target keys, which would otherwise silently grant nothing."""
    permissions = settings.get("permissions") if isinstance(settings, dict) else None
    if not isinstance(permissions, dict):
        return
    targets = read_yaml_mapping(root / "deployment" / "targets.yml").get("targets") or {}
    unknown = sorted(str(name) for name in set(permissions) - set(targets))
    if unknown:
        raise ValueError(f"serving_endpoint.permissions names unknown targets: {unknown}.")


def _check_serving(root: Path, bindings: dict[str, str]) -> None:
    """Validate optional serving jobs' settings offline, as each job does before remote work."""
    path = root / "config" / "serving.yml"
    if not path.exists():
        return
    label = "config/serving.yml"
    try:
        document = read_yaml_mapping(path)
        if "online_serving" in document:
            raise ValueError("online_serving was renamed to online_feature_lookup.")
        if "serving_endpoint" in document:
            _check_permission_targets(root, document["serving_endpoint"])
            read_serving_endpoint_settings(
                {
                    **bindings,
                    "serving_config_path": str(path),
                    "bundle_project": "static_preview",
                    "bundle_target": bindings.get("bundle_target", "static_preview"),
                    "cost_tag_key": "CostCenter",
                    "cost_tag_value": "static_preview",
                }
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    for section in _SERVING_SECTIONS:
        if section not in document:
            continue  # A job that was not generated has no section to read.
        try:
            validate_serving_section(section, document[section])
        except KeyError as exc:
            raise ValueError(f"{label}: {section}: missing required setting {exc}.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: {section}: {exc}") from exc


def _check_feature_targets(root: Path) -> None:
    """Keep the feature job's per-target identity in step with training, in targets.yml."""
    from ..features.graph import (  # noqa: PLC0415
        feature_targets_snippet,
        missing_feature_controls,
        stale_feature_targets,
    )

    missing = missing_feature_controls(root)
    if missing:
        raise ValueError(
            "deployment/targets.yml: the feature job needs the training job's run_as/"
            "permissions in these targets; merge:\n" + feature_targets_snippet(missing)
        )
    stale = stale_feature_targets(root)
    if stale:
        raise ValueError(
            "deployment/targets.yml: remove resources.jobs.features from "
            f"{', '.join(stale)}; feature groups are off, so Bundle would create an empty job."
        )


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
            if relative == "src/features":
                feature_project_source(path)
            else:
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
    declared dependencies are parsed, never installed or imported. Search and
    competition-metric settings get preview's checks, and an optional
    config/serving.yml gets the serving jobs' own pre-remote checks.
    """
    root = Path(project_path).resolve(strict=True)
    config = _check_config(root, bindings)
    _check_serving(root, bindings)
    _check_feature_targets(root)
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
