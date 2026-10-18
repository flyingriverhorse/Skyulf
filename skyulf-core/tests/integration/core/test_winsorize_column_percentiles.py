"""Per-column percentile recipes retain learned clipping and exact integer boundaries."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.preprocessing.outliers.winsorize import WinsorizeApplier, WinsorizeCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_winsorize_column_percentiles_and_global_fallback(engine):
    """Each measurement learns its own tails while unspecified columns retain global defaults."""
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 100.0], "b": [0.0, 10.0, 20.0, 100.0]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    state = WinsorizeCalculator().fit(
        native,
        {
            "columns": ["a", "b"],
            "lower_percentile": 0,
            "upper_percentile": 100,
            "percentiles_by_column": {"a": {"lower": 25, "upper": 75}},
        },
    )
    np.testing.assert_allclose(list(state["bounds"]["a"].values()), frame.a.quantile([0.25, 0.75]))
    assert state["bounds"]["b"] == {"lower": 0.0, "upper": 100.0}
    assert state["percentiles_by_column"] == {
        "a": {"lower": 25, "upper": 75},
        "b": {"lower": 0, "upper": 100},
    }
    restored = pickle.loads(pickle.dumps(state))
    for request in (frame, pl.from_pandas(frame)):
        result = WinsorizeApplier().apply(request, restored)
        expected = frame.a.clip(*frame.a.quantile([0.25, 0.75]))
        np.testing.assert_allclose(result["a"].to_numpy(), expected)
        assert len(WinsorizeApplier().apply(request.head(0), restored)) == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"a": {"lower": 90, "upper": 10}},
        {"a": {"lower": True, "upper": 95}},
        {"a": {"lower": 5}},
        {"missing": {"lower": 0, "upper": 100}},
        {"a": {"lower": 0, "upper": float("inf")}},
    ],
)
def test_winsorize_rejects_invalid_override_maps(overrides):
    """Misspelled axes and invalid tails must not silently fall back to global percentiles."""
    with pytest.raises(ValueError, match="percentile"):
        WinsorizeCalculator().fit(
            pd.DataFrame({"a": [0.0, 1.0, 2.0]}),
            {"columns": ["a"], "percentiles_by_column": overrides},
        )


def test_winsorize_override_preserves_integer_casting_requirement():
    """New per-column controls cannot weaken the established integer precision contract."""
    with pytest.raises(ValueError, match="Casting"):
        WinsorizeCalculator().fit(
            pd.DataFrame({"a": [0, 1, 2, 10]}),
            {"columns": ["a"], "percentiles_by_column": {"a": {"lower": 25, "upper": 75}}},
        )


def test_winsorize_override_metadata_excludes_empty_columns_and_rejects_corruption():
    """Empty source columns produce warnings, while saved override axes match real learned bounds."""
    frame = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
    state = WinsorizeCalculator().fit(
        frame, {"columns": ["a", "b"], "percentiles_by_column": {"a": {"lower": 0, "upper": 100}}}
    )
    assert set(state["bounds"]) == set(state["percentiles_by_column"]) == {"b"}
    assert len(state["warnings"]) == 1
    WinsorizeApplier.validate_inference_state(dict(state))
    state["percentiles_by_column"]["a"] = {"lower": 0, "upper": 100}
    with pytest.raises(ValueError, match="axes"):
        WinsorizeApplier.validate_inference_state(dict(state))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_winsorize_column_percentiles_real_pipeline(engine, tmp_path, monkeypatch):
    """Saved scoring uses the learned distinct tails even for extreme single-row requests."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    frame = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) ** 2})
    train = frame.assign(target=frame.a + frame.b)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "tails",
                    "transformer": "Winsorize",
                    "params": {
                        "columns": ["a", "b"],
                        "percentiles_by_column": {"a": {"lower": 10, "upper": 90}},
                    },
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(pl.from_pandas(train) if engine == "polars" else train, target_column="target")
    save_pipeline(pipeline, tmp_path / "tails")
    restored = load_pipeline(tmp_path / "tails")
    sample = pd.DataFrame({"a": [-50.0, 100.0], "b": [-50.0, 1000.0]})
    expected = predict_pipeline(sample, restored).prediction.to_numpy()
    np.testing.assert_allclose(
        predict_pipeline(pl.from_pandas(sample), restored).prediction, expected
    )
    np.testing.assert_allclose(
        [predict_pipeline(sample.iloc[[i]], restored).prediction.iloc[0] for i in range(2)],
        expected,
    )
    assert len(predict_pipeline(sample.head(0), restored)) == 0
