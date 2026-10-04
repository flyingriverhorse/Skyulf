"""Validate explicit, default-disabled classification decision policies."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from ...modeling._tuning.refit import resolve_threshold_metric
from ...registry import NodeRegistry


def _number(value: Any, name: str, *, maximum: float | None = None) -> float:
    """Reject coercions and nonfinite threshold values before fitting."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Decision threshold {name} must be a finite number.")
    if value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Decision threshold {name} is outside its allowed range.")
    return float(value)


def _label(value: Any) -> None:
    """Keep scalar labels typed across JSON and reject null or nonfinite labels."""
    if type(value) not in (str, int, float, bool) or (
        type(value) is float and not math.isfinite(value)
    ):
        raise ValueError("Decision threshold class labels must be finite scalar labels.")


def _manual(policy: dict[str, Any]) -> None:
    """Admit either a named binary cutoff or complete typed class entries."""
    if set(policy) == {"mode", "value", "positive_class"}:
        _number(policy["value"], "value", maximum=1)
        _label(policy["positive_class"])
        return
    if set(policy) != {"mode", "thresholds"}:
        raise ValueError("Manual decision threshold requires value/positive_class or thresholds.")
    entries = policy["thresholds"]
    if type(entries) is not list or len(entries) < 2:
        raise ValueError("Decision thresholds require an entry for every class.")
    labels = []
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"class", "value"}:
            raise ValueError("Decision threshold entries require class and value.")
        _label(entry["class"])
        if _number(entry["value"], "class value") <= 0:
            raise ValueError("Per-class decision thresholds must be positive.")
        labels.append(entry["class"])
    if len(set(labels)) != len(labels):
        raise ValueError("Decision threshold class entries must be distinct.")


def _automatic(policy: dict[str, Any]) -> None:
    """Validate a separate calibration split and an explicit hard-label objective."""
    allowed = {"mode", "metric", "validation_fraction", "random_state", "positive_class"}
    if set(policy) - allowed:
        raise ValueError("Unknown automatic decision threshold setting.")
    metric = policy.setdefault("metric", "balanced_accuracy")
    if metric not in {"balanced_accuracy", "f1", "f1_macro", "f1_weighted", "matthews_corrcoef"}:
        raise ValueError("Decision threshold metric must score class predictions.")
    fraction = _number(policy.setdefault("validation_fraction", 0.2), "validation_fraction")
    if not 0 < fraction < 1:
        raise ValueError("Decision threshold validation_fraction must be between zero and one.")
    seed = policy.setdefault("random_state", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Decision threshold random_state must be an unsigned integer.")
    if "positive_class" in policy:
        _label(policy["positive_class"])


def threshold_policy(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Copy and validate one policy without changing legacy pipelines."""
    value = pipeline.get("decision_threshold", {"mode": "off"})
    if type(value) is not dict or value.get("mode") not in {"off", "manual", "auto"}:
        raise ValueError("Decision threshold mode must be off, manual or auto.")
    policy = deepcopy(value)
    mode = policy["mode"]
    if mode == "off":
        if set(policy) != {"mode"}:
            raise ValueError("Disabled decision threshold accepts only mode.")
    elif mode == "manual":
        _manual(policy)
    else:
        _automatic(policy)
    modeling = pipeline.get("modeling", {})
    if mode != "off" and modeling.get("tune_threshold"):
        raise ValueError("Use decision_threshold or legacy tune_threshold, not both.")
    if mode != "off":
        selected = modeling.get("base_model", modeling)
        calculator = NodeRegistry.get_calculator(selected["type"])()
        if calculator.problem_type != "classification":
            raise ValueError("Decision thresholds require classification.")
    return policy


def needs_threshold_time(config: dict[str, Any]) -> bool:
    """Retain chronological split metadata when any workflow candidate calibrates."""
    if config.get("split_strategy") != "temporal":
        return False
    candidates = config.get("competition", {}).get("candidates", {}).values()
    pipelines = [config.get("pipeline", {}), *(entry["pipeline"] for entry in candidates)]
    return any(
        pipeline.get("decision_threshold", {}).get("mode") == "auto" for pipeline in pipelines
    )


def manual_thresholds(policy: dict[str, Any], classes: list[Any]) -> dict[Any, float]:
    """Resolve typed declarations against the actual probability-column labels."""
    if "positive_class" in policy:
        positive = policy["positive_class"]
        if len(classes) != 2 or positive not in classes:
            raise ValueError("Binary decision threshold positive_class must match a fitted class.")
        return {
            label: policy["value"] if label == positive else 1 - policy["value"]
            for label in classes
        }
    thresholds = {entry["class"]: entry["value"] for entry in policy["thresholds"]}
    if set(thresholds) != set(classes):
        raise ValueError("Decision thresholds must cover exactly the fitted classes.")
    return thresholds


def threshold_metric(policy: dict[str, Any], classes: list[Any]) -> Any:
    """Resolve the configured objective without silently changing its semantics."""
    metric = policy["metric"]
    positive = policy.get("positive_class", classes[-1])
    if metric == "f1" and (len(classes) != 2 or positive not in classes):
        raise ValueError("Binary threshold F1 requires a valid positive class.")
    scorer, resolved = resolve_threshold_metric(metric, None, pos_label=positive)
    if resolved != metric:
        raise ValueError("Decision threshold metric must score class predictions.")
    return scorer
