"""Tests for SkyulfPipeline."""

import os
import tempfile

import pandas as pd
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
        if os.path.exists(save_path):
            os.remove(save_path)
