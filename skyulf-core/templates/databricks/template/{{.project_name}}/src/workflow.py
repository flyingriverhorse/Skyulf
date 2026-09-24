# Databricks notebook source
"""Read job settings and call the installed Skyulf workflow services."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from skyulf.integrations.databricks.local_workflow import resolve_target_config, run_action


def main() -> None:
    """Read deployed configuration and job widgets only at the notebook boundary."""
    widgets = globals()["dbutils"].widgets
    config = json.loads(Path(widgets.get("config_path")).read_text(encoding="utf-8"))
    config = resolve_target_config(
        config,
        {
            name: widgets.get(name)
            for name in (
                "catalog",
                "input_schema",
                "output_schema",
                "metadata_schema",
                "resource_suffix",
            )
        },
    )
    action = widgets.get("action")
    with tempfile.TemporaryDirectory(prefix="skyulf-bundle-") as directory:
        result = run_action(
            globals()["spark"],
            config,
            action,
            experiment_name=widgets.get("experiment_name") if action.startswith("train") else None,
            artifact_path=Path(directory) / "artifact" if action.startswith("train") else None,
        )
    output = json.dumps(asdict(result), default=str, allow_nan=False)
    print(output)
    globals()["dbutils"].notebook.exit(output)


if __name__ == "__main__":
    main()
