"""F-15 stress coverage: rich preprocessing chains across model families.

Complements ``test_fold_preprocessing_refit.py`` (mechanism proofs with one
narrow WOE + logistic-regression setup) by exercising the always-on refit
through a realistic chain — imputation, scaling and target-aware encoding —
across classification, regression and ensemble algorithms, and by re-checking
score honesty on noise targets for a regressor and a tree ensemble.
"""

import math

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType

SPLITTER_STEP = {
    "name": "split",
    "transformer": "TrainTestSplitter",
    "params": {"target_column": "target", "test_size": 0.2},
}
IMPUTE_STEP = {
    "name": "impute_num2",
    "transformer": "SimpleImputer",
    "params": {"strategy": "mean", "columns": ["num2"]},
}
SCALE_STEP = {
    "name": "scale",
    "transformer": "StandardScaler",
    "params": {"columns": ["num1", "num2"]},
}
CLF_STEPS = [
    SPLITTER_STEP,
    IMPUTE_STEP,
    SCALE_STEP,
    {
        "name": "woe_cat",
        "transformer": "WOEEncoder",
        "params": {"columns": ["cat"], "regularization": 0.5},
    },
]
REG_STEPS = [
    SPLITTER_STEP,
    IMPUTE_STEP,
    SCALE_STEP,
    {
        "name": "target_encode_cat",
        "transformer": "TargetEncoder",
        "params": {"columns": ["cat"], "target_column": "target"},
    },
]


def _write_csv(tmp_path, df, name):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def signal_clf_csv(tmp_path):
    """Signal-bearing classification frame: NaN column + 10-category feature."""
    rng = np.random.default_rng(7)
    n = 400
    num1 = rng.normal(0, 1, n)
    num2 = rng.normal(0, 1, n)
    num2[rng.random(n) < 0.1] = np.nan
    df = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "cat": [f"c{v}" for v in rng.integers(0, 10, size=n)],
            "target": (num1 + rng.normal(0, 0.5, n) > 0).astype(int),
        }
    )
    return _write_csv(tmp_path, df, "clf.csv")


@pytest.fixture
def signal_reg_csv(tmp_path):
    rng = np.random.default_rng(11)
    n = 400
    num1 = rng.normal(0, 1, n)
    num2 = rng.normal(0, 1, n)
    num2[rng.random(n) < 0.1] = np.nan
    df = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "cat": [f"c{v}" for v in rng.integers(0, 10, size=n)],
            "target": 3.0 * num1 + rng.normal(0, 1.0, n),
        }
    )
    return _write_csv(tmp_path, df, "reg.csv")


@pytest.fixture
def noise_reg_csv(tmp_path):
    """Same features, pure-noise continuous target: TargetEncoder can only
    'predict' it by leaking validation rows into its per-category means."""
    rng = np.random.default_rng(13)
    n = 400
    num1 = rng.normal(0, 1, n)
    num2 = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "cat": [f"c{v}" for v in rng.integers(0, 200, size=n)],
            "target": rng.normal(0, 1, n),
        }
    )
    return _write_csv(tmp_path, df, "noise_reg.csv")


def _loader(node_id, path):
    return NodeConfig(
        node_id=node_id, step_type=StepType.DATA_LOADER, params={"source": "csv", "path": path}
    )


def _training(inputs, algorithm, hyperparameters, metric):
    return NodeConfig(
        node_id="node_training",
        step_type=StepType.TRAINING,
        inputs=inputs,
        params={
            "target_column": "target",
            "algorithm": algorithm,
            "hyperparameters": hyperparameters,
            "metric": metric,
            "cv_enabled": True,
            "cv_folds": 3,
            "evaluate": True,
        },
    )


def _run(tmp_path, csv_path, steps, training, job_id):
    logs: list[str] = []
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog(), log_callback=logs.append)
    result = engine.run(
        PipelineConfig(
            pipeline_id=job_id,
            nodes=[
                _loader("node_data", csv_path),
                NodeConfig(
                    node_id="node_features",
                    step_type=StepType.FEATURE_ENGINEERING,
                    inputs=["node_data"],
                    params={"steps": steps},
                ),
                training,
            ],
        ),
        job_id=job_id,
    )
    return result, logs


CLASSIFIERS = [
    ("logistic_regression", {"C": 1.0}),
    ("random_forest_classifier", {"n_estimators": 20, "random_state": 42}),
    ("gradient_boosting_classifier", {"n_estimators": 20, "random_state": 42}),
    (
        "voting_classifier",
        {"base_estimators": ["logistic_regression", "random_forest"]},
    ),
    (
        "stacking_classifier",
        {"base_estimators": ["logistic_regression", "random_forest"]},
    ),
]


@pytest.mark.parametrize("algorithm,hyperparameters", CLASSIFIERS)
def test_classification_chain_refit_scores_real_signal(
    signal_clf_csv, tmp_path, algorithm, hyperparameters
):
    """Imputer + scaler + WOE chain refit per fold; signal must survive it."""
    training = _training(["node_features"], algorithm, hyperparameters, "roc_auc")
    result, logs = _run(
        tmp_path, signal_clf_csv, CLF_STEPS, training, f"f15-stress-clf-{algorithm}"
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    metrics = result.node_results["node_training"].metrics
    auc = metrics["cv_roc_auc_mean"]
    assert math.isfinite(auc) and auc > 0.65, f"{algorithm}: expected real signal, got {auc}"


REGRESSORS = [
    ("ridge_regression", {}),
    ("gradient_boosting_regressor", {"n_estimators": 20, "random_state": 42}),
    ("voting_regressor", {"base_estimators": ["ridge", "random_forest"]}),
]


@pytest.mark.parametrize("algorithm,hyperparameters", REGRESSORS)
def test_regression_chain_refit_scores_real_signal(
    signal_reg_csv, tmp_path, algorithm, hyperparameters
):
    """Imputer + scaler + TargetEncoder chain refit per fold for regressors."""
    training = _training(["node_features"], algorithm, hyperparameters, "r2")
    result, logs = _run(
        tmp_path, signal_reg_csv, REG_STEPS, training, f"f15-stress-reg-{algorithm}"
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    metrics = result.node_results["node_training"].metrics
    r2 = metrics["cv_r2_mean"]
    assert math.isfinite(r2) and r2 > 0.5, f"{algorithm}: expected real signal, got {r2}"


def test_regression_noise_target_stays_near_zero(noise_reg_csv, tmp_path):
    """Honesty probe for the regression path: leaky TargetEncoder memorises
    per-category noise means and inflates R^2; per-fold refit keeps it ~0."""
    training = _training(["node_features"], "ridge_regression", {}, "r2")
    result, logs = _run(tmp_path, noise_reg_csv, REG_STEPS, training, "f15-stress-noise-reg")

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    r2 = result.node_results["node_training"].metrics["cv_r2_mean"]
    assert r2 < 0.15, f"per-fold refit R^2 on noise should sit near zero, got {r2}"


def test_random_forest_noise_target_stays_near_chance(noise_reg_csv, tmp_path):
    """Honesty probe through a tree ensemble on the classification noise trick:
    a memorising WOE step + forest on noise must stay near chance."""
    rng = np.random.default_rng(17)
    df = pd.read_csv(noise_reg_csv)
    df["target"] = rng.integers(0, 2, size=len(df))
    path = _write_csv(tmp_path, df, "noise_clf.csv")

    training = _training(
        ["node_features"],
        "random_forest_classifier",
        {"n_estimators": 20, "random_state": 42},
        "roc_auc",
    )
    result, logs = _run(tmp_path, path, CLF_STEPS, training, "f15-stress-noise-clf")

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    auc = result.node_results["node_training"].metrics["cv_roc_auc_mean"]
    assert max(auc, 1.0 - auc) < 0.65, f"forest on noise should sit near chance, got {auc}"
