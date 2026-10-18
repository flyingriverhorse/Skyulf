"""New category nodes survive real fitted-pipeline persistence and engine changes."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["frequency", "count", "rare", "rare_no_imputer"])
def test_category_pipeline_roundtrip_chunks_and_empty(engine, mode, tmp_path, monkeypatch):
    """Exported inference uses training counts/membership even on unseen/null single rows."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    train = pd.DataFrame(
        {
            "cat": ["a"] * 12 + ["b"] * 6 + [None] * 4 + ["rare"] * 2,
            "target": [1.0] * 12 + [4.0] * 6 + [8.0] * 4 + [12.0] * 2,
        }
    )
    if mode.startswith("rare"):
        steps = [
            {
                "name": "group",
                "transformer": "RareCategoryGrouping",
                "params": {"columns": ["cat"], "min_count": 3},
            },
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["cat"], "strategy": "constant", "fill_value": "Missing"},
            },
            {
                "name": "encode",
                "transformer": "OneHotEncoder",
                "params": {"columns": ["cat"], "max_categories": None},
            },
        ]
        if mode == "rare_no_imputer":
            steps[0]["params"]["min_count"] = 5
            steps.pop(1)
    else:
        steps = [
            {
                "name": "encode",
                "transformer": "FrequencyEncoder",
                "params": {"columns": ["cat"], "mode": mode},
            }
        ]
    pipeline = SkyulfPipeline({"preprocessing": steps, "modeling": {"type": "linear_regression"}})
    pipeline.fit(pl.from_pandas(train) if engine == "polars" else train, target_column="target")
    saved = tmp_path / mode
    save_pipeline(pipeline, saved)
    restored = load_pipeline(saved)
    sample = pd.DataFrame({"cat": [None, np.nan, pd.NA, "a", "b", "rare", "unseen"]})
    other = pl.DataFrame({"cat": [None, None, None, "a", "b", "rare", "unseen"]})
    expected = predict_pipeline(sample, restored).to_numpy()
    np.testing.assert_allclose(predict_pipeline(other, restored).to_numpy(), expected)
    native = other if engine == "polars" else sample
    parts = [
        predict_pipeline(
            native[i : i + 1] if engine == "polars" else sample.iloc[[i]], restored
        ).to_numpy()
        for i in range(len(sample))
    ]
    np.testing.assert_allclose(np.concatenate(parts), expected)
    np.testing.assert_allclose(
        predict_pipeline(sample.iloc[::-1], restored).to_numpy(), expected[::-1]
    )
    assert predict_pipeline(native.head(0), restored).shape[0] == 0
