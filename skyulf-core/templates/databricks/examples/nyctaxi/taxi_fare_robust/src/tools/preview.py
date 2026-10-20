"""Inspect this project's Core configuration without starting a cloud job."""

import argparse
import json
from functools import partial
from pathlib import Path

from skyulf.integrations.databricks.jobs.training.branch_notebook import (
    load_training_branch_configs,
)
from skyulf.integrations.databricks.lifecycle.workflow import (
    bind_target_name,
    resolve_target_config,
)
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.projects.source_validation import (
    query_source_sql,
    validate_project_sources,
)
from skyulf.integrations.databricks.projects.workflow_config import preview_workflow_config
from skyulf.integrations.databricks.projects.yaml_config import (
    project_config_path,
    read_workflow_config,
)
from skyulf.modeling.base import BaseModelCalculator
from skyulf.registry import NodeRegistry


def _preview(config, action):
    """Show every requested candidate, including its independent preprocessing and search."""
    if config.get("training_layout") != "model_competition":
        return preview_workflow_config(config, action=action)
    lines = [
        "Model competition: shared source, target and CV; separate candidate training tasks.",
        "Only the CV winner receives the final holdout check and registration.",
    ]
    for name, candidate in config["competition"]["candidates"].items():
        lines.append(f"\nCandidate: {name}")
        selected = {**config, "pipeline": candidate["pipeline"]}
        lines.append(preview_workflow_config(selected, action=action))
    return "\n".join(lines)


def _schedule_warning(config):
    """Disclose a fixed data window independently of the job's trigger clock."""
    scheduled = "manual" == "scheduled"
    if scheduled and config.get("training_window_mode") == "fixed_window":
        return "Warning: every scheduled run uses the same fixed training window. Choose rolling_days or rolling_calendar to advance it.\n"
    return ""


def _project_preview(config, path, bindings, action):
    """Preview each branch's own settings using the same loader as training."""
    if config.get("training_layout") == "multi_target":
        branches = load_training_branch_configs(
            {
                "config_path": str(path),
                "workflow_contract": "3",
                "deployed_score_handoff": config["score_handoff"],
                "deployed_auto_rebuild_on_cdf_expiry": "false",
                **bindings,
            }
        )
        return "\n\n".join(
            f"Branch: {name}\n{_schedule_warning(branch)}{preview_workflow_config(branch, action=action)}"
            for name, branch in branches.items()
        )
    loaded = load_project_workflow(config, path.parents[1] / "src/features")
    return _schedule_warning(loaded) + _preview(resolve_target_config(loaded, bindings), action)


def _source_preview(config, path, bindings, args):
    """Read current source contracts using an explicitly selected workspace and warehouse."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415 - online validation only

    if not args.profile or not args.warehouse_id:
        raise ValueError("--action sources requires --profile and --warehouse-id.")
    client = WorkspaceClient(profile=args.profile)
    query = partial(query_source_sql, client, args.warehouse_id)
    if config.get("training_layout") == "multi_target":
        configs = load_training_branch_configs(
            {
                "config_path": str(path),
                "workflow_contract": "3",
                "deployed_score_handoff": config["score_handoff"],
                "deployed_auto_rebuild_on_cdf_expiry": "false",
                **bindings,
            }
        )
    else:
        loaded = load_project_workflow(config, path.parents[1] / "src/features")
        configs = {"main": resolve_target_config(loaded, bindings)}
    label = args.label_table
    if label:
        label = "{catalog}.{input_schema}." + label if "." not in label else label
        label = bind_target_name("label_table", label, bindings, "")
    return json.dumps(
        {
            name: validate_project_sources(
                value,
                query,
                label_table=label or None,
                label_available_at_column=args.label_available_at_column or None,
            )
            for name, value in configs.items()
        },
        indent=2,
    )


def main():
    """Show the selected pipeline or discover installed Core model/node metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=["score", "train", "approve", "sources"],
        help="Preview scoring, training, saved-model approval, or source contracts; default uses approval for endpoint-only delivery and scoring otherwise.",
    )
    parser.add_argument("--profile")
    parser.add_argument("--warehouse-id")
    parser.add_argument("--label-table", default="")
    parser.add_argument("--label-available-at-column", default="")
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--input-schema", default="taxi_fare_robust")
    parser.add_argument("--output-schema", default="taxi_fare_robust")
    parser.add_argument("--metadata-schema", default="taxi_fare_robust")
    parser.add_argument("--resource-suffix", default="")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-preprocessors", action="store_true")
    args = parser.parse_args()
    path = project_config_path(Path(__file__).resolve().parents[2])
    try:
        config = read_workflow_config(path)
    except ValueError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    if args.action is None:
        args.action = "approve" if config.get("prediction_delivery") == "endpoint" else "score"
    if args.list_models or args.list_preprocessors:
        for node_id, meta in sorted(NodeRegistry.get_all_metadata().items()):
            calculator = NodeRegistry.get_calculator(node_id)
            is_model = issubclass(calculator, BaseModelCalculator)
            if is_model:
                if not args.list_models:
                    continue
                if (
                    config.get("training_layout") != "multi_target"
                    and calculator().problem_type != config["task"]
                ):
                    continue
            elif not args.list_preprocessors:
                continue
            print(f"{node_id}: {meta['name']} | {meta['description']}")
            print("  Metadata defaults:", json.dumps(meta.get("params", {})))
        print("Availability depends on installed dependencies and the selected node options.")
        return
    bindings = {
        key: getattr(args, key)
        for key in (
            "catalog",
            "input_schema",
            "output_schema",
            "metadata_schema",
            "resource_suffix",
        )
    }
    try:
        print(
            _source_preview(config, path, bindings, args)
            if args.action == "sources"
            else _project_preview(config, path, bindings, args.action)
        )
    except ValueError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")


if __name__ == "__main__":
    main()
