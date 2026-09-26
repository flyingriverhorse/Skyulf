"""Pinned model evaluation must reproduce evidence without changing inference settings."""

import pandas as pd
import polars as pl
import pytest
from joblib import effective_n_jobs, parallel_config

from skyulf.data.dataset import SplitDataset
from skyulf.inference import local_evaluation
from skyulf.inference.local_pipeline import load_local_pipeline
from skyulf.integrations.databricks.local_batch import fit_local_workflow


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_parallel_forest_evidence_repeats_without_changing_saved_parallelism(
    tmp_path, engine, task
):
    """Parallel forest sum order must not invalidate otherwise identical approval evidence."""
    frame = pd.DataFrame(
        [
            {
                "x": None if i % 13 == 0 else float(i % 10),
                "z": float(i % 3),
                "target": float(3 * (i % 10) + i % 3 + 10)
                if task == "regression"
                else int(i % 10 >= 5),
            }
            for i in range(180)
        ]
    )
    train, holdout = frame.iloc[:125], frame.iloc[125:]
    if engine == "polars":
        train, holdout = pl.from_pandas(train), pl.from_pandas(holdout)
    config = {
        "preprocessing": [
            {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x", "z"]}},
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x", "z"]}},
        ],
        "modeling": {
            "type": "random_forest_regressor"
            if task == "regression"
            else "random_forest_classifier",
            "params": {"n_estimators": 16, "max_depth": 5, "random_state": 42, "n_jobs": 2},
        },
    }
    path = tmp_path / "artifact.pkl"
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=train, test=train.head(0)),
        target_column="target",
        artifact_path=path,
        max_rows=1000,
        max_bytes=1048576,
    )
    baseline = local_evaluation.evaluate_local_holdout(artifact, holdout, target_column="target")
    with parallel_config(backend="threading", n_jobs=2):
        for _ in range(8):
            loaded = load_local_pipeline(path)
            repeated = local_evaluation.evaluate_local_holdout(
                loaded, holdout, target_column="target"
            )
            assert repeated == baseline
            assert loaded.pipeline.model_estimator is not None
            assert loaded.pipeline.model_estimator.model is not None
            assert loaded.pipeline.model_estimator.model.n_jobs == 2
        assert effective_n_jobs(2) == 2


def test_evaluation_restores_callers_parallel_backend_on_failure(tmp_path, monkeypatch):
    """An evaluation error must not disable parallel training or scoring in the caller."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    artifact = fit_local_workflow(
        {"modeling": {"type": "linear_regression"}},
        SplitDataset(train=frame, test=frame.head(0)),
        target_column="target",
        artifact_path=tmp_path / "linear.pkl",
        max_rows=10,
        max_bytes=1048576,
    )
    observed = []

    def failing_prediction(*args, **kwargs):
        """Record the effective worker count at the real prediction boundary."""
        observed.append(effective_n_jobs(2))
        raise ValueError("prediction failed")

    monkeypatch.setattr(local_evaluation, "predict_local_pipeline", failing_prediction)
    with parallel_config(backend="threading", n_jobs=2):
        with pytest.raises(ValueError, match="prediction failed"):
            local_evaluation.evaluate_local_holdout(artifact, frame, target_column="target")
        assert effective_n_jobs(2) == 2
    assert observed == [1]
