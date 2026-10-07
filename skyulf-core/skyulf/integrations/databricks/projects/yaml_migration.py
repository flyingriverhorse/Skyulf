"""Explicit, reversible migration of generated static declarations to optional YAML."""

import ast
import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..observability.charts.evaluation_chart_data import chart_settings
from ..training.tuning.local_cv import CV_FIELDS, LocalCVSpec
from ._project_files import read_source
from .workflow_config import validate_workflow_pipeline
from .yaml_config import INFERENCE_FIELDS, TRAINING_FIELDS, read_workflow_config
from .yaml_models import static_workflows

_UNIQUE_BODY = """
result = {}
for name, value in pairs:
    if name in result:
        raise ValueError(f"Duplicate branch declaration or setting: {name}")
    result[name] = value
return result
"""


def _body(nodes: list[ast.stmt]) -> list[ast.stmt]:
    """Ignore documentation while comparing exact generated executable statements."""
    if nodes and isinstance(nodes[0], ast.Expr) and isinstance(nodes[0].value, ast.Constant):
        return nodes[1:]
    return nodes


def _same_body(nodes: list[ast.stmt], expected: str) -> bool:
    """Match generated factory code structurally without importing the project."""
    return ast.dump(ast.Module(body=_body(nodes), type_ignores=[])) == ast.dump(ast.parse(expected))


def _factory_is_generated(node: ast.FunctionDef, task: str) -> bool:
    """Permit only known factory bodies; preserve custom logic through manual migration."""
    expected = {
        "build_modeling": "return deepcopy(MODELING)",
        "build_training_branches": "return deepcopy(MODELS)",
        "_unique_object": _UNIQUE_BODY,
        "build_candidates": (
            f"if task != {task!r}:\n"
            "    raise ValueError('Candidate task differs from the task selected during Bundle setup.')\n"
            "if not MODELS:\n"
            "    raise ValueError('Choose model_competition during Bundle setup to generate candidates.')\n"
            "return deepcopy(MODELS)"
        ),
    }
    arguments = {"build_candidates": "task", "_unique_object": "pairs"}
    return (
        node.name in expected
        and _plain_signature(node, arguments.get(node.name, ""))
        and _same_body(node.body, expected[node.name])
    )


def _plain_signature(node: ast.FunctionDef, arguments: str = "") -> bool:
    """Refuse executable defaults, decorators or annotations in a generated factory."""
    expected = ast.parse(f"def factory({arguments}): pass").body[0]
    if not isinstance(expected, ast.FunctionDef):
        raise ValueError("Invalid generated factory signature.")
    return (
        not node.decorator_list
        and node.returns is None
        and not node.type_params
        and ast.dump(node.args) == ast.dump(expected.args)
    )


def _static_assignment(node: ast.Assign) -> tuple[str, Any]:
    """Read literal declarations, including the generated duplicate-checking model list."""
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        raise ValueError("Only generated static model assignments can be migrated.")
    name = node.targets[0].id
    if name not in {"MODELING", "MODELS", "WEIGHT_COLUMN", "DECISION_THRESHOLD"}:
        raise ValueError(f"Cannot migrate custom model declaration {name}.")
    value = node.value
    _check_literal_keys(value)
    if isinstance(value, ast.Call) and ast.unparse(value.func) == "_unique_object":
        if len(value.args) != 1 or value.keywords:
            raise ValueError("Only generated static MODELS can be migrated.")
        pairs = ast.literal_eval(value.args[0])
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate model names in generated MODELS.")
        return name, result
    return name, ast.literal_eval(value)


def _check_literal_keys(value: ast.AST) -> None:
    """Reject duplicate static dictionary keys before literal evaluation drops them."""
    for node in ast.walk(value):
        if isinstance(node, ast.Dict):
            keys = [ast.literal_eval(key) for key in node.keys if key is not None]
            if len(keys) != len(set(keys)):
                raise ValueError("Duplicate dictionary key in static model declaration.")


def _static_models(path: Path, task: str) -> dict[str, Any]:
    """Reject custom top-level code and factories before reading generated literals."""
    declarations = {}
    for node in _body(ast.parse(read_source(path)).body):
        if isinstance(node, ast.Assign):
            name, value = _static_assignment(node)
            if name in declarations:
                raise ValueError(f"Duplicate generated declaration: {name}.")
            declarations[name] = value
        elif (
            isinstance(node, ast.ImportFrom) and ast.unparse(node) == "from copy import deepcopy"
        ) or (isinstance(node, ast.FunctionDef) and _factory_is_generated(node, task)):
            continue
        else:
            raise ValueError(f"{path.name} contains custom code; migrate it manually.")
    return declarations


def _static_model_set(path: Path) -> dict[str, Any]:
    """Read the generated model-set literal without executing its function."""
    nodes = _body(ast.parse(read_source(path)).body)
    if len(nodes) != 1 or not isinstance(nodes[0], ast.FunctionDef):
        raise ValueError("model_set.py contains custom code; migrate it manually.")
    function = nodes[0]
    body = _body(function.body)
    if function.name != "build_model_set" or len(body) != 3 or not _plain_signature(function):
        raise ValueError("Only the generated model_set.py factory can be migrated.")
    if not _same_body(body[:2], "enabled = True\nif not enabled:\n    return None"):
        raise ValueError("Only the generated enabled model set can be migrated.")
    if not isinstance(body[-1], ast.Return) or body[-1].value is None:
        raise ValueError("Generated model-set factory must return literal settings.")
    _check_literal_keys(body[-1].value)
    return ast.literal_eval(body[-1].value)


def _pipeline_entry(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Use compact model/tuning sections while retaining all selected model parameters."""
    if pipeline.get("preprocessing") or set(pipeline) - {
        "preprocessing",
        "modeling",
        "decision_threshold",
        "explainability",
    }:
        raise ValueError(
            "Migrate custom pipeline settings manually; only generated pipelines are supported."
        )
    modeling = deepcopy(pipeline["modeling"])
    result = {}
    if modeling.get("type") == "hyperparameter_tuner":
        result["model"] = modeling.pop("base_model")
        modeling.pop("type")
        result["tuning"] = modeling
    else:
        result["model"] = modeling
    for key in ("decision_threshold", "explainability"):
        if key in pipeline:
            result[key] = deepcopy(pipeline[key])
    return result


def _declarations(root: Path, workflow: dict[str, Any]) -> tuple[dict, dict, list[Path]]:
    """Extract the selected layout and preserve model-set settings separately."""
    layout = workflow.get("training_layout", "single_model")
    filenames = {
        "single_model": "single_model.py",
        "model_competition": "model_competition.py",
        "multi_target": "multi_model.py",
    }
    if layout not in filenames:
        raise ValueError("Unknown training_layout for YAML migration.")
    path = root / "src/modeling" / filenames[layout]
    declared = _static_models(path, workflow.get("task", "regression"))
    files = [path]
    models = _converted_models(layout, declared, workflow)
    model_set = root / "src/modeling/model_set.py"
    inference = {}
    if model_set.exists():
        inference["model_set"] = _static_model_set(model_set)
        files.append(model_set)
    return models, inference, files


def _converted_models(layout: str, declared: dict, workflow: dict) -> dict:
    """Convert each layout without importing or executing editable Python source."""
    if layout == "single_model":
        pipeline = {**workflow["pipeline"], "modeling": declared["MODELING"]}
        if "DECISION_THRESHOLD" in declared:
            pipeline["decision_threshold"] = declared["DECISION_THRESHOLD"]
        models = {"main": _pipeline_entry(pipeline)}
    elif layout == "model_competition":
        models = {
            name: {
                **_pipeline_entry({"modeling": item["modeling"]}),
                **{key: value for key, value in item.items() if key != "modeling"},
            }
            for name, item in declared["MODELS"].items()
        }
    else:
        models = _converted_branches(declared["MODELS"])
    if "WEIGHT_COLUMN" in declared:
        for item in models.values():
            item["weight_column"] = declared["WEIGHT_COLUMN"]
    return models


def _converted_branches(entries: dict) -> dict:
    """Retain each branch's independent workflow, feature selection and estimator."""
    models = {}
    for name, item in entries.items():
        overlay = deepcopy(item["workflow"])
        if overlay.pop("score_handoff", "disabled") != "disabled":
            raise ValueError("Generated branches must keep score_handoff disabled.")
        pipeline = overlay.pop("pipeline")
        models[name] = {
            **overlay,
            **_pipeline_entry(pipeline),
            **{key: value for key, value in item.items() if key != "workflow"},
        }
    return models


def _shared_defaults(workflow: dict, models: dict) -> dict:
    """Factor identical declarations into defaults without silently merging nested params."""
    defaults = {key: workflow.pop(key) for key in list(workflow) if key in TRAINING_FIELDS}
    first = next(iter(models.values()))
    for key, value in list(first.items()):
        if all(key in item and item[key] == value for item in models.values()):
            defaults[key] = deepcopy(value)
            for item in models.values():
                item.pop(key)
    return defaults


def _compact_defaults(defaults: dict[str, Any]) -> None:
    """Omit only values identical to the runtime defaults, retaining every nondefault choice."""
    LocalCVSpec.from_workflow(defaults)
    cv_defaults = LocalCVSpec()
    for key, field in CV_FIELDS.items():
        if key != "cv_enabled" and defaults.get(key) == getattr(cv_defaults, field):
            defaults.pop(key, None)
    charts = defaults.get("evaluation_charts")
    chart_settings(charts)
    chart_defaults = chart_settings({"enabled": True})
    if isinstance(charts, dict) and chart_defaults is not None:
        for key in chart_defaults:
            if key != "enabled" and charts.get(key) == chart_defaults[key]:
                charts.pop(key, None)
    _compact_source_options(defaults)


def _compact_source_options(defaults: dict[str, Any]) -> None:
    """Retain active source controls while removing explicit nulls with identical .get defaults."""
    for key in (
        "training_sample_rows",
        "training_version",
        "risk_category",
        "champion_version",
        "quality_threshold",
    ):
        if defaults.get(key) is None:
            defaults.pop(key, None)
    if defaults.get("training_sample_rows") is None and defaults.get("training_sample_seed") == 42:
        defaults.pop("training_sample_seed")
    parsing_default = {"format": None, "timezone": None, "date_only": "reject"}
    for column, parsing in (
        ("event_column", "event_time_parsing"),
        ("result_available_at_column", "result_time_parsing"),
    ):
        if defaults.get(column) is None and defaults.get(parsing) == parsing_default:
            defaults.pop(parsing)


def _migration_payload(root: Path) -> tuple[dict[str, str], list[Path]]:
    """Validate staged ownership before touching any original project declaration."""
    import yaml  # noqa: PLC0415

    workflow = read_workflow_config(root / "config/workflow.json")
    models, inference, files = _declarations(root, workflow)
    defaults = _shared_defaults(workflow, models)
    _compact_defaults(defaults)
    training = {"version": 1, "defaults": defaults, "models": models}
    inference = {
        "version": 1,
        **inference,
        **{key: workflow.pop(key) for key in list(workflow) if key in INFERENCE_FIELDS},
    }
    workflow["pipeline"]["modeling"] = {}
    workflow["pipeline"].pop("decision_threshold", None)
    if "explainability" in defaults:
        workflow["pipeline"].pop("explainability", None)
    payload = {
        "config/workflow.json": json.dumps(workflow, indent=2) + "\n",
        "config/training.yml": yaml.safe_dump(training, sort_keys=False),
        "config/inference.yml": yaml.safe_dump(inference, sort_keys=False),
    }
    with tempfile.TemporaryDirectory(prefix=".skyulf-yaml-stage-", dir=root) as temporary:
        staged = Path(temporary)
        (staged / "config").mkdir()
        for relative, content in payload.items():
            (staged / relative).write_text(content, encoding="utf-8")
        config = read_workflow_config(staged / "config/workflow.json")
        for candidate in static_workflows(training, config):
            validate_workflow_pipeline(candidate, candidate.get("task", "regression"))
    return payload, files


def _publish(root: Path, payload: dict[str, str], files: list[Path]) -> None:
    """Back up originals, publish outputs, and restore originals on any write failure."""
    backup = root / ".skyulf-yaml-backup"
    originals = [root / "config/workflow.json", *files]
    for path in originals:
        target = backup / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    created = []
    try:
        for relative, content in payload.items():
            path = root / relative
            if path.name == "workflow.json":
                continue
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(content)
        (root / "config/workflow.json").write_text(
            payload["config/workflow.json"], encoding="utf-8"
        )
        for path in files:
            path.unlink()
        read_workflow_config(root / "config/workflow.json")
    except BaseException:
        for path in originals:
            shutil.copy2(backup / path.relative_to(root), path)
        for path in created:
            path.unlink(missing_ok=True)
        raise


def migrate_project_yaml(project: str | Path) -> dict[str, Any]:
    """Move generated static settings to YAML with byte-exact originals retained locally.

    Custom functions and feature packages stay in Python. Projects with custom model
    factories must migrate manually. Existing YAML or a prior backup is never replaced.
    Run smoke, preview and graph refresh, then validate and redeploy after migration.
    """
    root = Path(project).resolve(strict=True)
    for relative in ("config/training.yml", "config/inference.yml", ".skyulf-yaml-backup"):
        if (root / relative).exists() or (root / relative).is_symlink():
            raise ValueError(f"Refusing to replace existing {relative}.")
    for relative in ("config", "config/workflow.json", "src/modeling"):
        if not (root / relative).resolve().is_relative_to(root):
            raise ValueError("Configuration and model declarations must stay inside the project.")
    payload, files = _migration_payload(root)
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Model declarations must stay inside the project.")
    _publish(root, payload, files)
    return {"status": "migrated", "files": list(payload), "backup": ".skyulf-yaml-backup"}
