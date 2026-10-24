"""Refresh named training/SHAP tasks after editing the model list, then redeploy.

Run from any directory with the same Python environment used for this project:
    python src/tools/refresh_training_graph.py

Only the generated model tasks, their join dependencies and the model-list guard
are changed. Job schedules, compute, ownership and operator settings are retained.
"""

import json
import re
import runpy
from pathlib import Path

from skyulf.integrations.databricks.projects.yaml_config import (
    project_config_path,
    read_training_config,
    read_workflow_config,
)


def refresh(project):
    """Synchronize the generated graph with the project's trusted model declarations."""
    config = read_workflow_config(project_config_path(project))
    layout = config["training_layout"]
    if layout == "single_model":
        return []
    module = "multi_model" if layout == "multi_target" else "model_competition"
    yaml = read_training_config(project / "config")
    names = list(
        yaml["models"]
        if yaml is not None
        else runpy.run_path(str(project / f"src/modeling/{module}.py"))["MODELS"]
    )
    if not names or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,89}", name) for name in names):
        raise ValueError("Use 1-90 character model identifiers for named job tasks.")
    path = project / "resources/train.job.yml"
    source = path.read_text(encoding="utf-8")
    start = source.index("        # BEGIN MODEL TASKS")
    end = source.index("        # END MODEL TASKS", start)
    block = source[start:end]
    entries = re.split(r"(?=^        - <<:)", block, flags=re.MULTILINE)[1:]
    train = entries[0]
    original = re.search(r'model_key: "([^"]+)"', train).group(1)
    shap = next(
        (entry for entry in entries if "notebook_path: ../src/jobs/shap_report.py" in entry), ""
    )
    generated = "        # BEGIN MODEL TASKS\n"
    for name in names:
        generated += _renamed(train, original, name)
        if shap:
            shap_name = re.search(r"task_key: shap_(\w+)", shap).group(1)
            generated += _renamed(shap, shap_name, name)
    source = source[:start] + generated + source[end:]
    source = re.sub(
        r"(?m)^              model_keys_json: .*?$",
        lambda _: "              model_keys_json: '" + json.dumps(names) + "'",
        source,
    )
    start = source.index("            # BEGIN MODEL DEPENDENCIES")
    end = source.index("            # END MODEL DEPENDENCIES", start)
    deps = "            # BEGIN MODEL DEPENDENCIES\n" + "".join(
        f"            - task_key: train_{name}\n" for name in names
    )
    path.write_text(source[:start] + deps + source[end:], encoding="utf-8")
    return names


def _renamed(block, old, new):
    """Change identifiers and task-value references without rewriting unrelated options."""
    for field, before, after in (
        ("task_key", f"train_{old}", f"train_{new}"),
        ("task_key", f"shap_{old}", f"shap_{new}"),
        ("model_key", old, new),
    ):
        pattern = (
            rf"(?m)^([ \t]*(?:-[ \t]+)?{field}:[ \t]*)(['\"]?){re.escape(before)}"
            rf"\2(?=[ \t]*(?:#.*)?$)"
        )
        block = re.sub(pattern, rf"\g<1>\g<2>{after}\g<2>", block)
    for prefix in ("train", "shap"):
        block = block.replace(f"{{{{tasks.{prefix}_{old}.", f"{{{{tasks.{prefix}_{new}.")
    return block


if __name__ == "__main__":
    models = refresh(Path(__file__).resolve().parents[2])
    print("Training graph models:", ", ".join(models) or "single model (unchanged)")
    print("Validate and redeploy the Bundle before running it.")
