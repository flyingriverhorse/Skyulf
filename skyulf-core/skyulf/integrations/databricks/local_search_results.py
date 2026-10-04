"""Admit actual search folds and report bounded local tuning results."""

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import ShuffleSplit

from ...inference.local_pipeline import LocalPipelineArtifact
from ...modeling._tuning.schemas import TuningConfig, TuningResult
from ...registry import NodeRegistry
from .local_cv import LocalCVSpec, evaluate_training_cv, validate_fold_membership


def validate_search_membership(
    frame: pd.DataFrame | pl.DataFrame,
    pipeline: dict[str, Any],
    cv: LocalCVSpec,
    *,
    target_column: str,
    event_column: str | None,
) -> None:
    """Reject undersized actual tuning folds before any model fit."""
    modeling = pipeline.get("modeling", {})
    if modeling.get("type") != "hyperparameter_tuner":
        return
    if target_column not in frame.columns:
        raise ValueError("Search target_column is absent from training rows.")
    base = modeling["base_model"]
    problem_type = NodeRegistry.get_calculator(base["type"])().problem_type
    if not cv.enabled:
        labels = frame[target_column].to_numpy()
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=cv.random_state)
        for train, validation in splitter.split(np.arange(len(frame)), labels):
            if len(train) < 2 or len(validation) < 2:
                raise ValueError(
                    "Each search fold needs at least two training and validation rows."
                )
        return
    validate_fold_membership(frame, cv, target_column, problem_type, event_column)


def _json_value(value: Any) -> Any:
    """Convert Core scalar values without writing NaN or Infinity to JSON."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("Tuning evidence contains a non-JSON value.")


def _trial_evidence(result: TuningResult) -> list[dict[str, Any]]:
    """Normalize failed trial scores without allowing nonfinite JSON numbers."""
    trials = []
    for trial in result.trials:
        score = trial.get("score")
        finite = (
            isinstance(score, (int, float, np.integer, np.floating))
            and not isinstance(score, (bool, np.bool_))
            and math.isfinite(score)
        )
        trials.append(
            {
                **_json_value(trial),
                "score": _json_value(score) if finite else None,
                "status": "completed" if finite else "failed",
            }
        )
    return trials


def tuning_evidence(artifact: LocalPipelineArtifact) -> dict[str, Any] | None:
    """Return the fitted tuning result and effective search recipe as JSON-safe evidence."""
    config = artifact.pipeline.config
    modeling = config.get("modeling", {})
    if modeling.get("type") != "hyperparameter_tuner":
        return None
    estimator = artifact.pipeline.model_estimator
    fitted = estimator.model if estimator is not None else None
    if not isinstance(fitted, tuple) or len(fitted) != 2 or not isinstance(fitted[1], TuningResult):
        raise ValueError("Fitted search artifact lacks a Core TuningResult.")
    result = fitted[1]
    if not math.isfinite(result.best_score):
        raise ValueError("Fitted search has no finite best score.")
    trials = _trial_evidence(result)
    evidence = {
        "status": "completed",
        "modeling": _json_value(modeling),
        "requested_metric": modeling.get("metric"),
        "scoring_metric": result.scoring_metric,
        "score_direction": "higher_is_better; sklearn negative loss remains negative",
        "best_score": result.best_score,
        "best_params": _json_value(result.best_params),
        "n_trials": result.n_trials,
        "decision_thresholds": _json_value(result.decision_thresholds),
        "decision_threshold_metric": result.decision_threshold_metric,
        "trials": trials,
    }
    if getattr(result, "nested_cv", None) is not None:
        evidence["nested_cv"] = _json_value(result.nested_cv)
    source = config.get("search_python_source")
    if source is not None:
        if type(source) is not str:
            raise ValueError("search_python_source must be text.")
        evidence["source_code_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return evidence


def parameter_preview(value: Any, section: str, *, artifact_file: str = "tuning.json") -> str:
    """Keep parameter previews small while pointing to their complete artifact."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > 500:
        return f"See {artifact_file}: {section} (value exceeds parameter preview limit)"
    return encoded


def tuning_run_params(evidence: dict[str, Any]) -> dict[str, Any]:
    """Expose effective search settings and final selected parameters in Experiments.

    Requested trials are a configured budget; tuning_trials is the actual count
    reported by Core. For nested CV, best_params belongs to the final search on
    all training rows, while outer-fold selections remain in tuning.json.
    """
    modeling = evidence["modeling"]
    defaults = TuningConfig()
    params = {
        "tuning_strategy": modeling["strategy"],
        "tuning_metric": evidence["scoring_metric"],
        "tuning_trials": evidence["n_trials"],
        "tuning_model_type": modeling["base_model"]["type"],
        "tuning_requested_metric": evidence["requested_metric"],
        "tuning_requested_trials": modeling.get("n_trials", defaults.n_trials),
    }
    for name in ("timeout", "random_state", "n_jobs", "parallel_backend", "tune_threshold"):
        params[f"tuning_{name}"] = modeling.get(name, getattr(defaults, name))
    if "max_candidates" in modeling:
        params["tuning_max_candidates"] = modeling["max_candidates"]
    for name in ("strategy_params", "search_space"):
        params[f"tuning_{name}"] = parameter_preview(modeling.get(name, {}), f"modeling.{name}")
    params["tuning_best_params"] = parameter_preview(evidence["best_params"], "best_params")
    for name, value in evidence["best_params"].items():
        key = f"tuning_best_params.{name}"
        # Long names remain available in the complete artifact and summary above.
        if len(key) <= 250:
            params[key] = parameter_preview(value, f"best_params.{name}")
    return params


def post_selection_cv(
    frame: pd.DataFrame | pl.DataFrame,
    artifact: LocalPipelineArtifact,
    cv: LocalCVSpec,
    *,
    target_column: str,
    event_column: str | None = None,
    sample_weight: Any = None,
) -> dict[str, Any] | None:
    """Return stored nested search scores, retaining diagnostics for legacy artifacts."""
    modeling = artifact.pipeline.config.get("modeling", {})
    if not cv.enabled or cv.method != "nested_cv" or modeling.get("type") != "hyperparameter_tuner":
        return None
    evidence = tuning_evidence(artifact)
    assert evidence is not None
    if evidence.get("nested_cv") is not None:
        return evidence["nested_cv"]
    if cv.nested_type != "auto":
        raise ValueError("Fitted search lacks evidence for the requested nested split policy.")
    selected = dict(modeling["base_model"])
    selected["params"] = {**selected.get("params", {}), **evidence["best_params"]}
    selected["params"].pop("tune_base_models", None)
    task = artifact.manifest.task
    method = "stratified_k_fold" if task == "classification" else "k_fold"
    fixed_cv = replace(cv, method=method)
    fixed_pipeline = {
        "preprocessing": artifact.pipeline.config.get("preprocessing", []),
        "modeling": selected,
    }
    input_columns = [*artifact.manifest.input_columns, target_column]
    training_features = (
        frame.select(input_columns)
        if isinstance(frame, pl.DataFrame)
        else frame.loc[:, input_columns]
    )
    report = evaluate_training_cv(
        training_features,
        fixed_pipeline,
        fixed_cv,
        target_column=target_column,
        sample_weight=sample_weight,
    )
    assert report is not None
    return {
        **report,
        "status": "post_selection_diagnostic",
        "method_note": "Selected fixed model scored on training folds; no nested search was rerun.",
        "selected_model": selected,
        "cv_config": {**report["cv_config"], "method": method},
    }
