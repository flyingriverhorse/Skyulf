"""Saved median statistics must be admitted without learning from inference batches."""

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.pipeline import SkyulfPipeline


@pytest.mark.parametrize("node", ["SimpleImputer", "GroupImputer"])
def test_saved_median_pipeline_admits_partitioned_predictions(tmp_path, node):
    """Worker admission must use learned medians and the same predictions for any batch size."""
    train = pd.DataFrame(
        {
            "x": [1.0, 4.0, 9.0, np.nan, 15.0, np.nan],
            "g": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    params = {"columns": ["x"], "strategy": "median"}
    if node == "GroupImputer":
        params["group_by"] = "g"
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "fill", "transformer": node, "params": params}],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "max_depth": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(SplitDataset(train=train, test=train.iloc[:0]), target_column="y")
    save_pipeline(pipeline, tmp_path / "median")
    restored = load_pipeline(tmp_path / "median")
    certificate = require_partition_safe_pipeline(restored)
    assert certificate.steps[0].action == "apply"
    sample = pd.DataFrame(
        {"x": [np.nan, np.nan, np.nan, 1000.0], "g": [0.0, 1.0, 42.0, 1.0]},
        index=[7, 2, 2, 0],
    )
    transformed = restored.pipeline.feature_engineer.transform(sample)
    expected_fills = [2.5, 12.0, 6.5, 1000.0] if node == "GroupImputer" else [6.5] * 3 + [1000.0]
    assert transformed["x"].tolist() == expected_fills
    whole = predict_pipeline(sample, restored)
    for positions in ([0], [1, 2], [3], [3, 2, 1, 0]):
        actual = predict_pipeline(sample.iloc[positions], restored)
        pd.testing.assert_frame_equal(actual, whole.iloc[positions], check_exact=True)
    pd.testing.assert_frame_equal(
        restored.pipeline.feature_engineer.transform(sample.iloc[:0]),
        transformed.iloc[:0],
        check_exact=True,
    )
    assert require_partition_safe_pipeline(restored) == certificate
