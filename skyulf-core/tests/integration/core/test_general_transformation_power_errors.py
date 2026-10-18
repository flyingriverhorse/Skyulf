"""Fitted power transforms must fail visibly instead of scoring raw request values."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline


def _native(frame, engine):
    """Use the same training and request values on both native engines."""
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("standardize", [False, True])
def test_saved_box_cox_pipeline_rejects_invalid_neighbors(
    engine, standardize, tmp_path, monkeypatch
):
    """A bad query row must never make a valid row reach the model without its fitted transform."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    values = np.arange(1.0, 41.0)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "power",
                    "transformer": "GeneralTransformation",
                    "params": {
                        "transformations": [
                            {"column": "value", "method": "box-cox", "standardize": standardize}
                        ]
                    },
                }
            ],
            "modeling": {"type": "ridge_regression", "params": {"alpha": 0.1}},
        }
    )
    pipeline.fit(
        _native(pd.DataFrame({"value": values, "target": np.log(values)}), engine),
        target_column="target",
    )
    valid = _native(pd.DataFrame({"value": [4.0]}), engine)
    expected = np.asarray(pipeline.predict(valid))
    save_pipeline(pipeline, tmp_path / "power")
    restored = load_pipeline(tmp_path / "power")
    np.testing.assert_allclose(predict_pipeline(valid, restored)["prediction"], expected)
    neighbors = _native(pd.DataFrame({"value": [4.0, 8.0]}), engine)
    np.testing.assert_allclose(
        predict_pipeline(neighbors, restored)["prediction"].iloc[:1], expected
    )

    for bad_value in (-1.0, 0.0):
        contaminated = _native(pd.DataFrame({"value": [4.0, bad_value]}), engine)
        with pytest.raises(ValueError, match="box-cox.*column.*value"):
            pipeline.predict(contaminated)
        with pytest.raises(ValueError, match="box-cox.*column.*value"):
            predict_pipeline(contaminated, restored)

    np.testing.assert_allclose(predict_pipeline(valid, restored)["prediction"], expected)
