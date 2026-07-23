"""Tests for SkyulfPipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import polars as pl

from skyulf.pipeline import SkyulfPipeline


def test_end_to_end_pipeline(sample_classification_data):
    """Test full pipeline execution, saving, and loading."""

    # Define Pipeline Config
    pipeline_config = {
        "preprocessing": [
            {
                "name": "imputer",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
            },
            {
                "name": "encoder",
                "transformer": "OneHotEncoder",
                "params": {"columns": ["category"]},
            },
            {
                "name": "scaler",
                "transformer": "StandardScaler",
                "params": {"columns": ["feature1", "feature2"]},
            },
        ],
        "modeling": {"type": "logistic_regression", "params": {"C": 1.0}},
    }

    # Initialize
    pipeline = SkyulfPipeline(pipeline_config)

    # Fit
    metrics = pipeline.fit(sample_classification_data, target_column="target")

    assert "preprocessing" in metrics
    assert "modeling" in metrics

    # Predict
    # Create new data (subset)
    new_data = sample_classification_data.drop(columns=["target"]).iloc[:10]
    predictions = pipeline.predict(new_data)

    assert len(predictions) == 10
    assert isinstance(predictions, pd.Series)

    # Save & Load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        save_path = tmp.name

    try:
        pipeline.save(save_path)

        loaded_pipeline = SkyulfPipeline.load(save_path)

        # Check if loaded pipeline works
        loaded_preds = loaded_pipeline.predict(new_data)

        pd.testing.assert_series_equal(predictions, loaded_preds)

    finally:
        if Path(save_path).exists():
            Path(save_path).unlink()


def test_train_test_split_step_actually_splits_a_raw_polars_dataframe():
    """Regression test: a raw (unwrapped) polars DataFrame fed straight into
    `SkyulfPipeline.fit()` — the advertised polars-native usage — must
    actually get split by a configured `TrainTestSplitter` step, not silently
    skipped.

    A raw `pl.DataFrame` satisfies neither `pd.DataFrame` nor the
    `SkyulfDataFrame` protocol (it lacks `.copy()`, using `.clone()`
    instead), so the internal "is this already split?" guard in
    `FeatureEngineer._run_step` previously failed to recognize it and
    silently skipped the splitter entirely. The pipeline then fit AND
    evaluated the model on the *entire* dataset with no held-out test
    set at all — a real, silent leakage bug, not merely a missing feature.
    """
    df = pl.DataFrame(
        {
            "feature1": list(range(100)),
            "feature2": [float(i) * 1.5 for i in range(100)],
            "target": [i % 2 for i in range(100)],
        }
    )

    pipeline_config = {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": 0.25, "random_state": 42},
            },
        ],
        "modeling": {"type": "logistic_regression", "params": {}},
    }

    pipeline = SkyulfPipeline(pipeline_config)
    metrics = pipeline.fit(df, target_column="target")

    splits = metrics["modeling"]["splits"]
    assert "train" in splits
    assert "test" in splits, (
        "TrainTestSplitter did not produce a held-out test split for a raw "
        "polars DataFrame input — the split step was silently skipped."
    )
