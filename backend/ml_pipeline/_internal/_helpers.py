"""Small label helpers shared across the ML-pipeline router (E9).

Pure functions only — no FastAPI / DB / engine dependencies. Kept
together so naming changes (e.g. branch label format) only touch one
file.
"""

from __future__ import annotations

import re

from backend.ml_pipeline._execution.schemas import PipelineConfig
from backend.ml_pipeline.constants import StepType


def prettify_model_type(model_type: str) -> str:
    """Convert a snake_case model id to a readable name.

    Mirrors the frontend `useBranchColors` logic so canvas badges and
    backend-generated branch labels stay aligned.
    """
    if not model_type:
        return ""
    cleaned = model_type
    for suffix in ("_classifier", "_regressor"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return " ".join(w.capitalize() for w in cleaned.split("_"))


def branch_label(index: int, sub_config: PipelineConfig, dup_suffix: str = "") -> str:
    """Build a 'Path A · <suffix>' label for a branch sub-pipeline.

    Suffix priority (most specific first):
      1. Model name when the terminal node itself is a training/tuning node.
      2. The leaf node's `_display_name` (canvas label, sent by the
         frontend converter) so preview tabs read "Encoding" /
         "Scaling" rather than the raw step type "LabelEncoder" /
         "RobustScaler".
      3. Friendly version of the leaf node's `step_type` as a last
         resort.

    Note: model_type is only read when the TERMINAL (last node) is a
    training/tuning node. If a DataPreview or other non-training terminal
    happens to have a training node as an ancestor (e.g. Best Model output
    wired into DataPreview), those training nodes are intentionally ignored
    so the label reflects the actual terminal, not an upstream model.

    `dup_suffix` is appended to disambiguate sibling branches that
    share the same `model_type` (e.g. two XGBoost training nodes get
    `#1` / `#2`) so tab labels match the canvas edge labels exactly.
    """
    letter = chr(ord("A") + index)
    model_type = ""
    leaf_step = ""
    leaf_display = ""
    if sub_config.nodes:
        leaf = sub_config.nodes[-1]
        if leaf.step_type in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}:
            # Only scan for model_type when the terminal IS a training/tuning node.
            for n in sub_config.nodes:
                if n.step_type in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}:
                    model_type = n.params.get("model_type") or n.params.get("algorithm") or ""
        else:
            leaf_step = str(leaf.step_type)
            leaf_display = str(leaf.params.get("_display_name") or "")
    pretty = prettify_model_type(str(model_type))
    suffix_tail = f" {dup_suffix}" if dup_suffix else ""
    if pretty:
        return f"Path {letter} · {pretty}{suffix_tail}"
    if leaf_display:
        return f"Path {letter} · {leaf_display}{suffix_tail}"
    if leaf_step:
        # "StandardScaler" → "Standard Scaler", "data_loader" → "Data Loader".
        if "_" in leaf_step:
            friendly = " ".join(w.capitalize() for w in leaf_step.split("_"))
        else:
            friendly = re.sub(r"(?<!^)(?=[A-Z])", " ", leaf_step).strip()
        return f"Path {letter} · {friendly}{suffix_tail}"
    return f"Path {letter}"


__all__ = ["prettify_model_type", "branch_label"]
