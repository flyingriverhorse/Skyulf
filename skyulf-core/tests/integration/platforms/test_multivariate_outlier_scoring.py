"""Saved multivariate filters publish positional exclusions instead of dropping requests."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["IsolationForest", "LocalOutlierFactor"])
def test_multivariate_saved_pipeline_keeps_exclusions_and_never_refits(
    engine, node, tmp_path, monkeypatch
):
    """Actual persisted scoring must retain every request's outcome and use the fixed detector."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    rng = np.random.default_rng(81)
    x = rng.normal(size=80)
    train = pd.DataFrame({"x": x, "z": x + rng.normal(scale=0.1, size=80), "target": x * 2})
    recipe = {"columns": ["x", "z"], "contamination": 0.1, "n_neighbors": 8, "n_estimators": 20}
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "joint", "transformer": node, "params": recipe}],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(
        pl.from_pandas(train) if engine == "polars" else train,
        target_column="target",
        sample_weight=np.arange(80) + 1.0,
    )
    save_pipeline(pipeline, tmp_path / "joint")
    restored = load_pipeline(tmp_path / "joint")

    def forbid_fit(*args, **kwargs):
        """A saved inference path must never relearn the detector from request rows."""
        raise AssertionError("unexpected detector refit")

    monkeypatch.setattr(IsolationForest, "fit", forbid_fit)
    monkeypatch.setattr(LocalOutlierFactor, "fit", forbid_fit)
    sample = pd.DataFrame(
        {"x": [0.1, 10.0, -10.0, 0.2], "z": [0.1, -10.0, 10.0, 0.2]}, index=[7, 7, 2, 7]
    )
    result = predict_pipeline(sample, restored)
    assert len(result) == len(sample) and result.index.tolist() == sample.index.tolist()
    assert result.scoring_status.tolist() == ["predicted", "excluded", "excluded", "predicted"]
    assert result.exclusion_reason.iloc[1:3].tolist() == ["outlier:joint"] * 2
    assert result.prediction.iloc[1:3].isna().all()
    pd.testing.assert_frame_equal(
        result, pd.concat([predict_pipeline(sample.iloc[[i]], restored) for i in range(4)])
    )
    other = predict_pipeline(pl.from_pandas(sample), restored)
    np.testing.assert_allclose(other.prediction, result.prediction, equal_nan=True)
    assert predict_pipeline(sample.head(0), restored).shape[0] == 0
