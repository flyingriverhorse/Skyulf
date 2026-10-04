"""Exercise the public pipeline, fold tuning and serializer together after CCN extraction."""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.pipeline import SkyulfPipeline


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_tuned_regression_pipeline_roundtrip_preserves_formula(
    tmp_path, monkeypatch, engine, strategy
):
    """Replay casting, squaring and scaling once while preserving the saved pipeline seal."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    x = np.linspace(1, 30, 240)
    data = pd.DataFrame({"x": x.astype(str), "target": 3 * x**2 + 5})
    if engine == "polars":
        data = pl.from_pandas(data)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "cast",
                    "transformer": "Casting",
                    "params": {"column_types": {"x": "float"}},
                },
                {
                    "name": "split",
                    "transformer": "TrainTestSplitter",
                    "params": {"test_size": 0.25, "shuffle": False},
                },
                {
                    "name": "square",
                    "transformer": "SimpleTransformation",
                    "params": {"transformations": [{"column": "x", "method": "square"}]},
                },
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
            ],
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": "ridge_regression"},
                "strategy": strategy,
                "metric": "r2",
                "search_space": {"alpha": [1e-8, 0.01, 1.0]},
                "cv_folds": 3,
                "random_state": 42,
                "n_jobs": 1,
            },
        }
    )
    unfitted = pipeline.fingerprint()
    metrics = pipeline.fit(data, target_column="target")
    assert "modeling_error" not in metrics
    assert pipeline.is_fitted()
    fitted = pipeline.fingerprint()
    assert fitted != unfitted
    # Independent numeric oracle; these expected values do not call the fitted model.
    inputs = pd.DataFrame({"x": ["3", "9", "21"]})
    expected = np.array([32.0, 248.0, 1328.0])
    predictions = pipeline.predict(inputs)
    np.testing.assert_allclose(predictions, expected, rtol=1e-6, atol=1e-6)
    assert pipeline.fingerprint() == fitted
    model_path = tmp_path / "formula.pkl"
    pipeline.save(str(model_path))
    restored = SkyulfPipeline.load(str(model_path))
    np.testing.assert_allclose(restored.predict(inputs), predictions, rtol=0, atol=0)
    card = restored.export_model_card()
    (tmp_path / "model_card.json").write_text(
        json.dumps(card, indent=2, default=str), encoding="utf-8"
    )
    (tmp_path / "predictions.json").write_text(
        json.dumps(
            {
                "x": [3, 9, 21],
                "expected": expected.tolist(),
                "predicted": np.asarray(predictions).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert card["fingerprint"] == fitted == restored.fingerprint()
