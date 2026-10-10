"""Fit bounded competition candidates without registering any losing model."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl

from .....inference.project_code import load_project_module
from ....mlflow.runs.tracking import TrackingRun
from ...observability.reports.explanation_report import copy_winner_explanations
from ..fitting import candidate as training
from ..shared.training_evidence import evidence_digest
from ..thresholds.decision_thresholds import needs_threshold_time
from ..tuning.cv import CVSpec
from ..tuning.search import base_model_config
from ..weights.weights import extract_training_weights
from .competition import choose_winner
from .competition_evaluation import evaluate_competition_candidate


def fit_competition(
    spark: Any,
    store: Any,
    spec: training.TrainingSpec,
    directory: Path,
    *,
    prepared_data: Any = None,
) -> tuple[Any, dict[str, Any], Path, dict[str, Any]]:
    """Train all requested candidates, preserve child evidence, then select one."""
    request = store.request
    config = request["config"]
    cv = CVSpec.from_workflow(config)
    partitions = prepared_data or training.read_training_partitions(
        spark,
        spec,
        temporal_cv=cv.temporal or needs_threshold_time(config),
        engine=config["engine"],
    )
    rows: list[dict[str, Any]] = []
    best = None
    candidates = request["competition"]["candidates"]
    for name, recipe in sorted(candidates.items()):
        fitted, row = fit_training_pipeline(
            spark, store, spec, partitions, directory / name, name, recipe
        )
        rows.append(row)
        store.log("competition/progress.json", {"completed": rows, "requested": list(candidates)})
        winner = choose_winner(rows, {item["candidate"] for item in rows})["winner"]
        if name == winner:
            best = (fitted, recipe["pipeline"], directory / name)
    selection = choose_winner(rows, set(candidates))
    assert best is not None
    store.log("competition/selection.json", selection)
    if best[1].get("explainability"):
        copy_winner_explanations(store, selection)
    store.run.set_tags(
        {"competition_winner": selection["winner"], "competition_count": str(len(rows))}
    )
    return (*best, selection)


def _restore_source(pipeline: dict[str, Any]) -> None:
    """Register custom steps from the immutable candidate feature package."""
    source = pipeline.get("project_python_source")
    if source is None:
        return
    module = load_project_module(source)
    for name in ("build_preprocessing", "build_pre_split_steps"):
        factory = getattr(module, name, None)
        if callable(factory):
            factory()


def fit_training_pipeline(
    spark: Any,
    store: Any,
    spec: training.TrainingSpec,
    partitions: Any,
    path: Path,
    name: str,
    recipe: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Finish or fail one child run while retaining exact fit and evaluation evidence."""
    config = store.request["config"]
    cv = CVSpec.from_workflow(config)
    created = store.client.create_run(
        store.client.get_run(store.run_id).info.experiment_id,
        run_name=name,
        tags={"mlflow.parentRunId": store.run_id, "competition_candidate": name},
    )
    child = TrackingRun(client=store.client, run_id=created.info.run_id, enabled=True)
    try:
        _restore_source(recipe["pipeline"])
        fitted = training.fit_candidate(
            spark,
            spec,
            recipe["pipeline"],
            run=child,
            pipeline_config=recipe["effective_config"],
            artifact_path=path,
            engine=config["engine"],
            cv=cv,
            risk_category=config.get("risk_category"),
            prepared_data=deepcopy(partitions),
            evaluate_cv=False,
        )
        training.log_fitted_candidate(
            child,
            fitted,
            recipe["pipeline"],
            engine=config["engine"],
            risk_category=config.get("risk_category"),
        )
        model_uri = training.log_local_model(
            path,
            run_id=created.info.run_id,
            tracking_uri=config["tracking_uri"],
            **training.feature_log_options(spark, fitted.spec),
        )
        frame = pl.from_pandas(partitions[1]) if config["engine"] == "polars" else partitions[1]
        frame, sample_weight = extract_training_weights(frame, spec.weight_column)
        row = evaluate_competition_candidate(
            frame,
            fitted.artifact,
            cv,
            target_column=spec.target_column,
            event_column=spec.event_column,
            metric=config["metric"],
            max_rows=spec.max_rows,
            max_bytes=spec.max_bytes,
            cv_results=fitted.cv_results,
            sample_weight=sample_weight,
        )
        row.update(
            candidate=name,
            model_type=base_model_config(recipe["pipeline"])["type"],
            strategy=recipe["pipeline"]["modeling"].get("strategy", "fixed"),
            run_id=child.run_id,
            model_uri=model_uri,
            model_digest=fitted.artifact.manifest.pipeline_sha256,
            config_sha256=evidence_digest(recipe["effective_config"]),
        )
        child.client.log_dict(child.run_id, row, "competition_evaluation.json")
        child.log_metrics({"competition_score": row["mean"], "competition_std": row["std"]})
        child.client.set_terminated(child.run_id, status="FINISHED")
        return fitted, row
    except BaseException:
        child.client.set_terminated(child.run_id, status="FAILED")
        raise


# Preserve class imports exposed by earlier module paths.
LocalCVSpec = CVSpec
