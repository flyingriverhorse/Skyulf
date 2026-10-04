"""Decision policies must survive complete durable training and registry replay."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")

from test_databricks_lifecycle_tasks import (  # noqa: F401 - real isolated registry fixture
    _call,
    staged,
)

from skyulf.inference.local_pipeline import predict_local_pipeline
from skyulf.integrations.mlflow.registry import load_run_local_pipeline


def _classification(staged, engine, classes, mode):
    """Reuse real phase storage while replacing only unavailable Spark source transport."""
    _, _, config, _, frame = staged
    frame.drop(frame.index, inplace=True)
    frame["id"] = np.arange(120)
    frame["x"] = np.arange(120, dtype=float)
    frame["target"] = np.arange(120) % classes
    policy = {"mode": mode}
    if mode == "manual":
        policy.update(
            thresholds=[{"class": index, "value": 0.2 + index * 0.3} for index in range(classes)]
        )
    config.update(
        engine=engine,
        task="classification",
        max_rows=1000,
        metric="heldout_balanced_accuracy",
        quality_threshold=0.0,
        stratify=True,
        cv_enabled=True,
        cv_folds=2,
        cv_type="stratified_k_fold",
    )
    config["pipeline"] = {
        "preprocessing": [],
        "modeling": {"type": "logistic_regression", "params": {"max_iter": 500}},
        "decision_threshold": policy,
    }
    return config


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("mode", ["off", "manual", "auto"])
def test_thresholds_survive_training_registration_and_decision(staged, engine, classes, mode):
    """A passing fit alone must not hide failure at configuration replay or registration."""
    config = _classification(staged, engine, classes, mode)
    _, client, _, _, frame = staged
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    registered = _call(staged, "train_register", prepared.reference)
    decided = _call(staged, "compare_decide", registered.reference)
    run_id = prepared.reference["run_id"]
    artifact = load_run_local_pipeline(
        f"runs:/{run_id}/model",
        digest=client.get_run(run_id).data.params["model_digest"],
        tracking_uri=config["tracking_uri"],
    )
    output = predict_local_pipeline(frame[["x"]], artifact)
    assert len(output) == len(frame)
    assert artifact.manifest.use_tuned_thresholds is (mode != "off")
    assert decided.reference["phase"] == "decide"
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "1"


@pytest.mark.parametrize("mode", ["manual", "auto"])
@pytest.mark.parametrize("metric", ["balanced_accuracy", "roc_auc_ovr_weighted", "pr_auc_weighted"])
def test_threshold_competition_selects_and_registers_winner(staged, mode, metric):
    """Policy scores, winner adoption and registry replay must agree across candidates."""
    config = _classification(staged, "pandas", 2, mode)
    config["metric"] = f"heldout_{metric}"
    _, client, _, _, _ = staged
    native = deepcopy(config["pipeline"])
    native["decision_threshold"] = {"mode": "off"}
    config.update(
        training_layout="model_competition",
        competition_max_trials=100,
        competition_max_candidates=8,
    )
    config["competition"] = {
        "candidates": {
            "policy": {"pipeline": deepcopy(config["pipeline"])},
            "native": {"pipeline": native},
        }
    }
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    trained = _call(staged, "train", split.reference)
    selected = _call(staged, "select_best_model", trained.reference)
    registered = _call(staged, "evaluate_register", selected.reference)
    decided = _call(staged, "compare_decide", registered.reference)
    assert registered.reference["phase"] == "evaluate_register"
    assert len(client.search_model_versions(f"name='{config['model_name']}'")) == 1
    assert selected.output["candidate_count"] == 2
    assert decided.reference["phase"] == "decide"
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "1"


def test_temporal_auto_without_cv_survives_separate_prepare_task(staged):
    """The durable prepared training frame must retain time metadata for calibration."""
    config = _classification(staged, "pandas", 2, "auto")
    _, client, _, _, frame = staged
    frame["event"] = pd.date_range("2020-01-01", periods=len(frame), freq="h", tz="UTC")
    config.update(
        cv_enabled=False,
        cv_type="k_fold",
        stratify=False,
        split_strategy="temporal",
        event_column="event",
        training_window_mode="fixed_window",
        start="2020-01-01T00:00:00Z",
        holdout_start="2020-01-05T00:00:00Z",
        cutoff="2020-01-06T00:00:00Z",
    )
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    trained = _call(staged, "train", split.reference)
    selected = _call(staged, "select_best_model", trained.reference)
    registered = _call(staged, "evaluate_register", selected.reference)
    assert registered.reference["phase"] == "evaluate_register"
    assert len(client.search_model_versions(f"name='{config['model_name']}'")) == 1


@pytest.mark.parametrize("search", [False, True])
def test_mixed_temporal_competition_keeps_metadata_out_of_native_candidates(staged, search):
    """Calibration-only timestamps cannot leak into an off candidate's fit or ordinary CV."""
    config = _classification(staged, "pandas", 2, "auto")
    _, client, _, _, frame = staged
    frame["event"] = pd.date_range("2020-01-01", periods=len(frame), freq="h", tz="UTC")
    config.update(
        cv_type="k_fold",
        stratify=False,
        split_strategy="temporal",
        event_column="event",
        training_window_mode="fixed_window",
        start="2020-01-01T00:00:00Z",
        holdout_start="2020-01-05T00:00:00Z",
        cutoff="2020-01-06T00:00:00Z",
        training_layout="model_competition",
        competition_max_trials=100,
        competition_max_candidates=8,
    )
    native = deepcopy(config["pipeline"])
    native["decision_threshold"] = {"mode": "off"}
    if search:
        native["modeling"] = {
            "type": "hyperparameter_tuner",
            "base_model": native["modeling"],
            "strategy": "grid",
            "search_space": {"C": [0.5]},
            "metric": "balanced_accuracy",
        }
    config["competition"] = {
        "candidates": {
            "policy": {"pipeline": deepcopy(config["pipeline"])},
            "native": {"pipeline": native},
        }
    }
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    trained = _call(staged, "train", split.reference)
    selected = _call(staged, "select_best_model", trained.reference)
    registered = _call(staged, "evaluate_register", selected.reference)
    assert registered.reference["phase"] == "evaluate_register"
    assert len(client.search_model_versions(f"name='{config['model_name']}'")) == 1


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["off", "manual", "auto"])
@pytest.mark.parametrize("encoder", ["LabelEncoder", "OrdinalEncoder"])
def test_original_target_labels_survive_registry_lifecycle(staged, engine, mode, encoder):
    """Training, registry reload and promotion must share the original multiclass labels."""
    config = _classification(staged, engine, 3, mode)
    _, client, _, _, frame = staged
    frame["target"] = frame.target.map({0: "alpha", 1: "beta", 2: "gamma"})
    params = {"columns": ["target"]}
    if encoder == "OrdinalEncoder":
        params["categories_order"] = "gamma,beta,alpha"
    config["pipeline"]["preprocessing"] = [
        {"name": "encode", "transformer": encoder, "params": params}
    ]
    if mode == "manual":
        config["pipeline"]["decision_threshold"]["thresholds"] = [
            {"class": label, "value": 0.2 + i * 0.2}
            for i, label in enumerate(["alpha", "beta", "gamma"])
        ]
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    registered = _call(staged, "train_register", prepared.reference)
    decided = _call(staged, "compare_decide", registered.reference)
    run_id = prepared.reference["run_id"]
    artifact = load_run_local_pipeline(
        f"runs:/{run_id}/model",
        digest=client.get_run(run_id).data.params["model_digest"],
        tracking_uri=config["tracking_uri"],
    )
    output = predict_local_pipeline(frame[["x"]], artifact)
    assert set(artifact.manifest.classes) == {"alpha", "beta", "gamma"}
    assert set(output.prediction).issubset(set(frame.target))
    assert decided.reference["phase"] == "decide"
    assert str(client.get_registered_model(config["model_name"]).aliases["champion"]) == "1"
