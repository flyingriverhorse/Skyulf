"""Run multi-target candidates and optional coherent set lifecycle operations."""

import html
import json
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...inference.project_code import load_project_module
from ._project_files import read_source, renamed_modeling_hook
from .job_runtime import (
    lifecycle_widget_context,
    notebook_output,
    operator_options,
    read_notebook_config,
)
from .local_workflow import resolve_target_config
from .model_set_project import (
    capture_set_rules,
    load_project_model_set,
    package_training_model_set,
    project_endpoints,
    render_model_set_result,
)
from .project import load_project_workflow
from .weight_config import capture_branch_weights


def _training_only(config: dict[str, Any], *, allow_set_handoff: bool = False) -> None:
    """Keep component activation manual while allowing parent-level score orchestration."""
    if config.get("promotion_policy") != "manual_approval":
        raise ValueError("Multi-target training requires promotion_policy=manual_approval.")
    allowed = {"disabled", "after_alias_change"} if allow_set_handoff else {"disabled"}
    if config.get("score_handoff") not in allowed:
        raise ValueError("Multi-target training requires score_handoff=disabled.")


def _branch_entries(path: Path) -> tuple[dict[str, Any], str]:
    """Load the trusted branch factory separately from each saved feature package."""
    source = read_source(path)
    module = load_project_module(source)
    factory = getattr(module, "build_training_branches", None)
    if not callable(factory):
        raise ValueError(f"{path.name} must define build_training_branches().")
    entries = factory()
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"Configure a nonempty branch mapping in src/modeling/{path.name}.")
    return entries, source


def _branch_config(
    base: dict[str, Any],
    entry: Any,
    values: dict[str, str],
    modeling: Path,
    weight_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace explicit top-level settings and capture one branch's feature recipe."""
    allowed = {"workflow", "features_path", "preprocessing_recipe", "pre_split_recipe"}
    if not isinstance(entry, dict) or set(entry) - allowed:
        raise ValueError(
            "Each branch requires workflow and optional features_path, "
            "preprocessing_recipe, pre_split_recipe."
        )
    overlay = entry.get("workflow")
    if not isinstance(overlay, dict):
        raise ValueError("Each branch workflow must be a configuration overlay.")
    config = {**deepcopy(base), "score_handoff": "disabled", **deepcopy(overlay)}
    config["training_layout"] = "multi_target"
    _training_only(config)
    bindings = {
        name: values[name]
        for name in (
            "catalog",
            "input_schema",
            "output_schema",
            "metadata_schema",
            "resource_suffix",
        )
    }
    config = resolve_target_config(config, bindings)
    relative = entry.get("features_path", "../features")
    if not isinstance(relative, str) or not relative:
        raise ValueError("features_path must be a nonempty path relative to src/modeling.")
    path = (modeling / relative).resolve()
    if not path.is_relative_to(modeling.parent.resolve()):
        raise ValueError("Branch features_path must stay inside the project's src directory.")
    return load_project_workflow(
        config,
        path,
        preprocessing_recipe=entry.get("preprocessing_recipe"),
        pre_split_recipe=entry.get("pre_split_recipe"),
        weight_settings=weight_settings,
    )


def load_training_branch_configs(values: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Resolve independent full workflows with the deployed target's ownership rules.

    Workflow overlays replace whole top-level values, including ``pipeline``.
    They never merge nested model settings from a different target. All feature
    packages are captured before the service validates and reads remote data.
    """
    base = read_notebook_config(values)
    _training_only(base, allow_set_handoff=True)
    modeling = Path(values["config_path"]).parent.parent / "src/modeling"
    entries, source = _branch_entries(
        renamed_modeling_hook(modeling / "multi_model.py", "branches.py")
    )
    weights = capture_branch_weights(source, entries)
    return {
        name: _branch_config(base, entry, values, modeling, weights[name])
        for name, entry in entries.items()
    }


def render_branch_result(payload: dict[str, Any]) -> str:
    """Show saved component evidence without presenting unavailable activation controls."""
    return (
        "<h2>Multi-target training results</h2>"
        "<p>Candidates registered and compared. No aliases changed; scoring was not run. "
        "Enable src/modeling/model_set.py for coherent activation and scoring.</p><pre>"
        + html.escape(json.dumps(payload, indent=2, default=str, allow_nan=False))
        + "</pre>"
    )


def run_branch_training_notebook(
    spark: Any,
    dbutils: Any,
    *,
    display_html: Callable[[str], Any] | None = None,
    exit_notebook: bool = True,
) -> str:
    """Train branch candidates or explicitly operate an enabled saved model set."""
    values = dbutils.widgets.getAll()
    lifecycle_widget_context(values)
    # Legacy sequential notebooks have no child score task; require the new graph.
    _training_only(read_notebook_config(values))
    action = values.get("lifecycle_action", "train")
    settings = load_project_model_set(values)
    if settings is None and action != "train":
        raise ValueError(
            "Multi-target entrypoint supports only train unless a model set is enabled."
        )
    options = operator_options(action, values)
    if action != "train":
        return _set_operator_output(
            spark,
            dbutils,
            values,
            settings,
            options,
            display_html=display_html,
            exit_notebook=exit_notebook,
        )
    source = ""
    if settings is not None:
        settings, source = capture_set_rules(values, settings)
    configs = load_training_branch_configs(values)
    champion_versions = None
    if settings is not None:
        from .model_set_release import pin_model_set_baseline  # noqa: PLC0415

        settings, champion_versions = pin_model_set_baseline(
            settings, configs, project_endpoints(next(iter(configs.values())))
        )
    from .local_branches import (  # noqa: PLC0415 - load training services after preflight
        prepare_training_branches,
        train_local_branches,
    )

    branches = prepare_training_branches(spark, configs, champion_versions=champion_versions)
    base = next(iter(configs.values()))
    with tempfile.TemporaryDirectory(prefix="skyulf-branches-") as directory:
        outcome = train_local_branches(
            spark,
            branches,
            **project_endpoints(base),
            experiment_name=values["experiment_name"],
            artifact_path=directory,
        )
    payload = asdict(outcome)
    if settings is not None:
        candidate = package_training_model_set(
            spark,
            branches,
            outcome,
            settings,
            composition_source=source,
            **project_endpoints(base),
        )
        payload["model_set_candidate"] = {
            "name": candidate.name,
            "version": candidate.version,
            "digest": candidate.digest,
        }
        from .model_set_release import automatic_model_set_release  # noqa: PLC0415

        payload.update(automatic_model_set_release(spark, candidate, settings, base))
        payload["next_actions"] = set_next_actions(payload)
    return notebook_output(
        payload,
        dbutils,
        render=render_model_set_result if settings is not None else render_branch_result,
        display_html=display_html,
        exit_notebook=exit_notebook,
        explanation_tracking_uri=(
            project_endpoints(base)["tracking_uri"]
            if any(config["pipeline"].get("explainability") for config in configs.values())
            else None
        ),
    )


def set_next_actions(payload: dict) -> list[str]:
    """Offer a valid next step without suggesting manual approval can bypass failed gates."""
    if payload.get("alias_change") is not None:
        return ["Run the score job using the approved model set."]
    if payload.get("quality", {}).get("passed") is False:
        return [
            "Inspect failed component quality decisions; the champion is unchanged.",
            "Adjust the models or quality policy and train a new candidate set.",
        ]
    return [
        "Review component comparisons and set outputs.",
        "Run approve with candidate_version and expected_champion_version.",
        "Run the score job after explicit approval.",
    ]


def _set_operator_output(
    spark: Any,
    dbutils: Any,
    values: dict[str, str],
    settings: dict[str, Any] | None,
    options: dict[str, Any],
    *,
    display_html: Callable[[str], Any] | None,
    exit_notebook: bool,
) -> str:
    """Keep legacy projects train only and route enabled saved-set operations."""
    if settings is None:
        raise ValueError(
            "Multi-target entrypoint supports only train unless a model set is enabled."
        )
    if values["lifecycle_action"] not in {"approve", "reject", "rollback"}:
        raise ValueError("Model sets support train, approve, reject and rollback.")
    from .model_set_project import run_model_set_operator  # noqa: PLC0415

    payload = run_model_set_operator(spark, values, settings, options)
    return notebook_output(
        payload,
        dbutils,
        render=render_model_set_result,
        display_html=display_html,
        exit_notebook=exit_notebook,
    )
