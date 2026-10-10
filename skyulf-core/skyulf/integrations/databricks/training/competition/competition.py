"""Pinned training plans and deterministic single-target candidate selection."""

import math
import statistics
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..fitting import candidate as training
from ..shared.training_evidence import evidence_digest
from ..thresholds.decision_thresholds import threshold_policy
from ..tuning.cv import CVSpec


def _bound_metric(pipeline: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Bind an omitted candidate search objective to the competition metric."""
    from .competition_evaluation import competition_metric  # noqa: PLC0415

    result = deepcopy(pipeline)
    if result["modeling"]["type"] == "hyperparameter_tuner":
        result["modeling"].setdefault(
            "metric", competition_metric(config["metric"], config["task"])
        )
    return result


def validate_competition_budget(config: dict[str, Any]) -> None:
    """Check aggregate search bounds before source/registry access."""
    from ..tuning.search import prepare_search_pipeline  # noqa: PLC0415

    cv = CVSpec.from_workflow(config)
    total = 0
    for candidate in config["competition"]["candidates"].values():
        effective = prepare_search_pipeline(
            _bound_metric(candidate["pipeline"], config),
            cv,
            target_column=config["target_column"],
            event_column=config.get("event_column"),
        )
        total += _pipeline_trial_bound(effective, cv)
    if total > config.get("competition_max_trials", 100):
        raise ValueError(f"Competition search budget exceeds competition_max_trials: {total}.")


def prepare_competition(
    config: dict[str, Any], spec: training.TrainingSpec, champion: str | None
) -> dict[str, Any]:
    """Freeze every effective recipe and reject excessive aggregate search work."""
    cv = CVSpec.from_workflow(config)
    candidates = {}
    total = 0
    for name, candidate in sorted(config["competition"]["candidates"].items()):
        pipeline = _bound_metric(candidate["pipeline"], config)
        effective = training.candidate_config(
            spec,
            pipeline,
            engine=config["engine"],
            cv=cv,
            metric=config["metric"],
            min_improvement=config["min_improvement"],
            champion_version=champion,
            quality_threshold=config.get("quality_threshold"),
            quality_gates=config.get("quality_gates"),
            risk_category=config.get("risk_category"),
        )
        bound = _pipeline_trial_bound(effective, cv)
        total += bound
        candidates[name] = {
            "pipeline": pipeline,
            "effective_config": effective,
            "trial_bound": bound,
        }
    if total > config.get("competition_max_trials", 100):
        raise ValueError(f"Competition search budget exceeds competition_max_trials: {total}.")
    return {"candidates": candidates, "trial_bound": total, "concurrency": 1}


def _trial_bound(model: dict[str, Any], cv: CVSpec) -> int:
    """Bound search configurations across nested searches, including halving rounds."""
    if model["type"] != "hyperparameter_tuner":
        return cv.folds + 1 if cv.method == "nested_cv" else 1
    strategy = model.get("strategy", "random")
    count = model.get("n_trials", 10)
    if strategy in {"grid", "halving_grid"}:
        count = math.prod(len(values) for values in model["search_space"].values())
    # Successive halving evaluates fewer survivors at every additional round.
    if strategy in {"halving_grid", "halving_random"}:
        factor = model.get("strategy_params", {}).get("factor", 3)
        rounds = 1 + math.floor(math.log(count, factor))
        remaining = count
        count = 0
        for _ in range(rounds):
            count += remaining
            remaining = math.ceil(remaining / factor)
    return count * (cv.folds + 1 if cv.method == "nested_cv" else 1)


def _pipeline_trial_bound(pipeline: dict[str, Any], cv: CVSpec) -> int:
    """Include independent outer policy searches in the admitted competition budget."""
    model = pipeline["modeling"]
    bound = _trial_bound(model, cv)
    if threshold_policy(pipeline)["mode"] == "off":
        return bound
    if model["type"] != "hyperparameter_tuner":
        return cv.folds + 1
    ordinary = replace(
        cv,
        method="k_fold",
        nested_type="auto",
        inner_folds=None,
        group_column=None,
        gap=0,
        test_size=None,
        max_train_size=None,
    )
    return bound + cv.folds * _trial_bound(model, ordinary)


def choose_winner(rows: list[dict[str, Any]], names: set[str]) -> dict[str, Any]:
    """Rank complete comparable evidence, resolving exact ties by candidate name."""
    if len(rows) != len(names) or {row["candidate"] for row in rows} != names:
        raise ValueError("Competition results are incomplete or contain duplicate candidates.")
    first = rows[0]
    for row in rows:
        _validate_score_row(row, first)
    sign = 1 if first["direction"] == "minimize" else -1
    ranked = sorted(rows, key=lambda row: (sign * row["mean"], row["candidate"]))
    return {
        "winner": ranked[0]["candidate"],
        "selection_metric": first["metric"],
        "direction": first["direction"],
        "tie_rule": "exact score ties use ascending candidate name",
        "candidate_count": len(rows),
        "leaderboard": ranked,
    }


def _validate_score_row(row: dict[str, Any], first: dict[str, Any]) -> None:
    """Reject incompatible policies, changed membership and non-finite scores."""
    fields = ("metric", "scoring_metric", "direction", "fold_membership_sha256", "split_policy")
    if any(row[field] != first[field] for field in fields):
        raise ValueError("Competition metrics, policies or fold membership differ.")
    if row["direction"] not in {"minimize", "maximize"}:
        raise ValueError("Competition metric direction is invalid.")
    if _evaluation_family(row) != _evaluation_family(first):
        raise ValueError("Competition evaluation policies differ.")
    _validate_summary(row)


def _evaluation_family(row: dict[str, Any]) -> str:
    """Keep bias disclosures while admitting fixed and tuned ordinary CV together."""
    modes = {
        "fixed_cv": "ordinary",
        "post_selection_cv": "ordinary",
        "nested_cv": "nested",
        "threshold_cv": "ordinary",
        "nested_threshold_cv": "nested",
    }
    if row["evaluation_mode"] not in modes:
        raise ValueError("Competition evaluation mode is invalid.")
    return modes[row["evaluation_mode"]]


def _validate_summary(row: dict[str, Any]) -> None:
    """Require finite complete fold scores and their actual population aggregates."""
    scores = row["fold_scores"]
    if len(scores) < 2 or not all(math.isfinite(value) for value in scores):
        raise ValueError("Competition requires a finite score for every fold.")
    if not math.isfinite(row["mean"]) or not math.isfinite(row["std"]):
        raise ValueError("Competition summary scores must be finite.")
    for field, expected in (("mean", statistics.mean(scores)), ("std", statistics.pstdev(scores))):
        if not math.isclose(row[field], expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Competition summary differs from fold scores.")


def selected_request(store: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover a winning recipe only from verified complete phase evidence."""
    request = store.request
    if "competition" not in request:
        return request["config"], request["effective_config"]
    fitted = store.receipt("train")["output"]
    saved = store.read("competition/selection.json")
    if evidence_digest(saved) != fitted["competition_sha256"]:
        raise ValueError("Competition selection differs from the saved phase receipt.")
    expected = choose_winner(saved["leaderboard"], set(request["competition"]["candidates"]))
    if expected != saved or fitted["competition"] != saved:
        raise ValueError("Competition winner differs from complete candidate evidence.")
    _validate_candidate_identities(saved, request["competition"]["candidates"], fitted)
    candidate = request["competition"]["candidates"][saved["winner"]]
    config = deepcopy(request["config"])
    config["pipeline"] = candidate["pipeline"]
    return config, candidate["effective_config"]


def _validate_candidate_identities(saved: dict, candidates: dict, fitted: dict) -> None:
    """Bind each score to its requested recipe and the winner to the promoted artifact."""
    for row in saved["leaderboard"]:
        recipe = candidates[row["candidate"]]
        if row["config_sha256"] != evidence_digest(recipe["effective_config"]):
            raise ValueError("Competition score belongs to another candidate recipe.")
        if row["candidate"] == saved["winner"] and row["model_digest"] != fitted["model_digest"]:
            raise ValueError("Competition winner artifact differs from the selected model.")


# Preserve class imports exposed by earlier module paths.
LocalCVSpec = CVSpec
