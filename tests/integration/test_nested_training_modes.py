"""Exercise real backend Basic/Advanced routes with ordinary and nested CV."""

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType


def _model_settings(model, classification):
    """Use small real ensembles with nondefault composition and two search candidates."""
    parameter = "C" if classification else "alpha"
    if not model.startswith(("voting", "stacking")):
        return {parameter: 0.5}, {parameter: [0.1, 1.0]}
    params = {
        "base_estimators": ["logistic_regression", "gaussian_nb"]
        if classification
        else ["linear_regression", "ridge"],
        "n_jobs": 1,
    }
    if model.startswith("voting"):
        params["weights"] = [2, 1]
        if classification:
            params["voting"] = "soft"
    else:
        params.update(
            cv=2,
            passthrough=True,
            final_estimator="logistic_regression" if classification else "ridge",
        )
    axis = "logistic_regression__C" if classification else "ridge__alpha"
    return params, {axis: [0.1, 1.0]}


@pytest.mark.parametrize("method", ["ordinary", "nested_cv"])
@pytest.mark.parametrize("mode", ["fixed", "tuned"])
@pytest.mark.parametrize(
    "model",
    [
        "logistic_regression",
        "ridge_regression",
        "voting_classifier",
        "voting_regressor",
        "stacking_classifier",
        "stacking_regressor",
    ],
)
def test_backend_training_modes_preserve_cv_and_artifacts(tmp_path, model, mode, method):
    """UI Basic/Advanced payloads must train, preserve fold isolation and reload results."""
    classification = model.endswith("classifier") or model == "logistic_regression"
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(120, 3)), columns=["x", "z", "w"])
    score = frame.x + frame.z - 0.2 * frame.w
    frame["target"] = (score > 0).astype(int) if classification else score
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    model_params, space = _model_settings(model, classification)
    cv = {
        "cv_enabled": True,
        "cv_folds": 2,
        "cv_type": "nested_cv"
        if method == "nested_cv"
        else ("stratified_k_fold" if classification else "k_fold"),
        "cv_shuffle": True,
        "cv_random_state": 42,
    }
    if method == "nested_cv":
        cv["cv_inner_folds"] = 2
    params = {"run_mode": mode, "algorithm": model, "target_column": "target"}
    if mode == "fixed":
        params.update(hyperparameters=model_params, **cv)
    else:
        params["tuning_config"] = {
            **model_params,
            **cv,
            "strategy": "grid",
            "metric": "accuracy" if classification else "r2",
            "search_space": space,
            "n_trials": 2,
        }
    nodes = [
        NodeConfig(
            node_id="data",
            step_type=StepType.DATA_LOADER,
            params={"source": "csv", "path": str(csv)},
        ),
        NodeConfig(
            node_id="split",
            step_type="TrainTestSplitter",
            inputs=["data"],
            params={"target_column": "target", "test_size": 0.2, "random_state": 42},
        ),
        NodeConfig(
            node_id="scale",
            step_type="StandardScaler",
            inputs=["split"],
            params={"columns": ["x", "z", "w"]},
        ),
        NodeConfig(
            node_id="training", step_type=StepType.TRAINING, inputs=["scale"], params=params
        ),
    ]
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())
    result = engine.run(PipelineConfig(pipeline_id="modes", nodes=nodes), job_id="modes")
    assert result.status == "success", {
        key: value.error for key, value in result.node_results.items()
    }
    metrics = result.node_results["training"].metrics
    assert "fold_refit_fallback" not in metrics
    assert metrics["fold_refit_audit"]["isolation_ok"] is True
    assert metrics["fold_refit_audit"]["train_rows"] == 96
    saved_model, saved_result = store.load("training")
    if model.startswith(("voting", "stacking")):
        assert [name for name, _ in saved_model.estimators] == model_params["base_estimators"]
        if model.startswith("voting"):
            assert saved_model.weights == [2, 1]
            if classification:
                assert saved_model.voting == "soft"
        else:
            assert saved_model.cv == 2 and saved_model.passthrough is True
    elif mode == "fixed":
        parameter = "C" if classification else "alpha"
        assert saved_model.get_params()[parameter] == 0.5
    expected_trials = 1 if mode == "fixed" else 2
    assert len(metrics["trials"]) == saved_result.n_trials == expected_trials
    assert np.isfinite(metrics["best_score"])
    split = store.load("scale")
    test_x, _test_y = split.test
    test_frame = test_x.to_pandas() if hasattr(test_x, "to_pandas") else test_x
    predictions = saved_model.predict(test_frame)
    assert len(predictions) == 24 and np.isfinite(predictions).all()
    if method == "nested_cv":
        nested = metrics["nested_cv"]
        assert saved_result.nested_cv == nested
        assert nested["outer_folds"] == nested["inner_folds"] == 2
        assert nested["total_trials"] == 3 * expected_trials
        assert all(f["train_rows"] == f["test_rows"] == 48 for f in nested["folds"])
        assert nested["mean_score"] == pytest.approx(
            np.mean([f["outer_score"] for f in nested["folds"]])
        )
        assert metrics[f"cv_{nested['scoring_metric']}_mean"] == nested["mean_score"]
    else:
        assert "nested_cv" not in metrics
        assert saved_result.nested_cv is None
        assert any(key.startswith("cv_") and key.endswith("_mean") for key in metrics)


@pytest.mark.parametrize(
    "model,policy",
    [
        (model, policy)
        for model in (
            "logistic_regression",
            "ridge_regression",
            "voting_classifier",
            "voting_regressor",
            "stacking_classifier",
            "stacking_regressor",
        )
        for policy in (
            "k_fold",
            "stratified_k_fold",
            "time_series_split",
            "group_k_fold",
            "stratified_group_k_fold",
        )
        if not policy.startswith("stratified")
        or model.endswith("classifier")
        or model == "logistic_regression"
    ],
)
@pytest.mark.parametrize("method", ["ordinary", "nested_cv"])
@pytest.mark.parametrize("mode", ["fixed", "tuned"])
def test_backend_policy_routes_preserve_metadata(tmp_path, policy, mode, method, model):
    """Single and ensemble routes must honor ordinary/nested CV without leaking metadata."""
    temporal = policy == "time_series_split"
    grouped = policy in {"group_k_fold", "stratified_group_k_fold"}
    classification = model.endswith("classifier") or model == "logistic_regression"
    count = 240
    rng = np.random.default_rng(76)
    frame = pd.DataFrame({"x": rng.normal(size=count), "target": np.tile([0, 1], count // 2)})
    if not classification:
        frame["target"] = 2 * frame.x + rng.normal(size=count) * 0.1
    if temporal or grouped:
        frame["metadata"] = (
            pd.date_range("2025-01-01", periods=count).astype(str)
            if temporal
            else np.repeat([f"customer_{i}" for i in range(24)], 10)
        )
    csv = tmp_path / "policy.csv"
    frame.to_csv(csv, index=False)
    cv = {
        "cv_enabled": True,
        "cv_type": "nested_cv" if method == "nested_cv" else policy,
        "cv_folds": 2,
        "cv_shuffle": False,
    }
    if method == "nested_cv":
        cv.update(cv_nested_type=policy, cv_inner_folds=2)
    if temporal or grouped:
        cv["cv_time_column" if temporal else "cv_group_column"] = "metadata"
    if temporal:
        cv.update(cv_gap=1, cv_test_size=15, cv_max_train_size=100)
    model_params, space = _model_settings(model, classification)
    params = {"run_mode": mode, "algorithm": model, "target_column": "target"}
    if mode == "fixed":
        params.update(hyperparameters=model_params, **cv)
    else:
        params["tuning_config"] = {
            **model_params,
            **cv,
            "strategy": "grid",
            "metric": "accuracy" if classification else "r2",
            "search_space": space,
            "n_trials": 2,
            "tune_threshold": classification and method == "nested_cv",
        }
    nodes = [
        NodeConfig(
            node_id="data",
            step_type=StepType.DATA_LOADER,
            params={"source": "csv", "path": str(csv)},
        ),
        NodeConfig(
            node_id="split",
            step_type="TrainTestSplitter",
            inputs=["data"],
            params={"target_column": "target", "test_size": 0.25, "shuffle": False},
        ),
        NodeConfig(
            node_id="scale", step_type="StandardScaler", inputs=["split"], params={"columns": ["x"]}
        ),
        NodeConfig(
            node_id="training", step_type=StepType.TRAINING, inputs=["scale"], params=params
        ),
    ]
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    result = PipelineEngine(store, catalog=FileSystemCatalog()).run(
        PipelineConfig(pipeline_id="policy", nodes=nodes), job_id="policy"
    )
    assert result.status == "success", {
        key: value.error for key, value in result.node_results.items()
    }
    metrics = result.node_results["training"].metrics
    assert metrics["fold_refit_audit"]["isolation_ok"] is True
    fitted, saved = store.load("training")
    assert fitted.n_features_in_ == 1
    assert np.isfinite(saved.best_score)
    heldout, _ = store.load("scale").test
    heldout = heldout.to_pandas() if hasattr(heldout, "to_pandas") else heldout
    predictions = fitted.predict(heldout[["x"]])
    assert len(predictions) == 60 and np.isfinite(predictions).all()
    if model.startswith(("voting", "stacking")):
        assert [name for name, _ in fitted.estimators] == model_params["base_estimators"]
    if method == "ordinary":
        assert "nested_cv" not in metrics
        assert saved.nested_cv is None
        assert any(key.startswith("cv_") and key.endswith("_mean") for key in metrics)
        return
    report = metrics["nested_cv"]
    assert report["split_policy"]["method"] == policy
    assert saved.nested_cv == report
    assert report["outer_folds"] == report["inner_folds"] == 2
    assert report["total_trials"] == (6 if mode == "tuned" else 3)
    if mode == "tuned" and classification:
        assert (
            saved.decision_thresholds and report["threshold_selection"]["selection"] == "inner_oof"
        )
        assert all(
            fold["threshold_selection"]["selection"] == "inner_oof" for fold in report["folds"]
        )
    else:
        assert saved.decision_thresholds is None
