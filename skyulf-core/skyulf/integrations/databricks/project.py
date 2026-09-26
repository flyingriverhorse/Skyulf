"""Resolve a trusted project Python preprocessing recipe into existing Core config."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from ...config_validation import validate_preprocessing_steps
from ...inference.project_code import MAX_PROJECT_SOURCE_BYTES, load_project_module


def load_project_workflow(config: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Use Python-defined steps for training/preview and capture their exact source.

    Score and lifecycle approval load the saved artifact instead of this file.
    A nonempty JSON chain is rejected rather than silently overwritten.
    """
    if config.get("pipeline", {}).get("preprocessing"):
        raise ValueError("Configure preprocessing in the Python file; leave the JSON list empty.")
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_PROJECT_SOURCE_BYTES + 1)
    if len(payload) > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError("Project preprocessing source exceeds 64 KiB.")
    source = payload.decode("utf-8")
    module = load_project_module(source)
    factory = getattr(module, "build_preprocessing", None)
    if not callable(factory):
        raise ValueError("preprocessing.py must define build_preprocessing().")
    steps = factory()
    if not isinstance(steps, list):
        raise ValueError("build_preprocessing() must return a list of Core steps.")
    validate_preprocessing_steps(steps)
    result = deepcopy(config)
    result["pipeline"]["preprocessing"] = steps
    result["pipeline"]["project_python_source"] = source
    return result
