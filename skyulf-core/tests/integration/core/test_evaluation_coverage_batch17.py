"""Expose the held-out population behind filtered evaluation metrics."""

import pickle

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.pipeline import FeatureEngineer


def _steps():
    """Exclude explicit missing features before splitting features from targets."""
    return [
        {"name": "eligible", "transformer": "DropMissingRows", "params": {"subset": ["x"]}},
        {"name": "xy", "transformer": "feature_target_split", "params": {"target_column": "y"}},
    ]


def _dataset(engine, *, empty_validation=False):
    """Keep training complete while each held-out split contains rejected rows."""
    frame = pl.DataFrame if engine == "polars" else pd.DataFrame
    return SplitDataset(
        train=frame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]}),
        test=frame({"x": [1.0, None, 3.0], "y": [2.0, 50.0, 6.0]}),
        validation=frame({"x": [None, None if empty_validation else 4.0], "y": [70.0, 8.0]}),
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pipeline_reports_original_heldout_denominator(engine):
    """Excellent eligible-row scores must disclose excluded test and validation rows."""
    pipeline = SkyulfPipeline(
        {"preprocessing": _steps(), "modeling": {"type": "linear_regression"}}
    )
    result = pipeline.fit(_dataset(engine), "y")["modeling"]
    for split, rows, scored in [("test", 3, 2), ("validation", 2, 1)]:
        coverage = result["splits"][split].coverage
        assert coverage["input_rows"] == rows
        assert coverage["scored_rows"] == scored
        assert coverage["excluded_rows"] == 1
        assert result["raw_data"]["splits"][split]["coverage"] == coverage


def test_split_coverage_survives_feature_target_split_and_copy():
    """A later node or copied dataset must not erase or alias earlier exclusion counts."""
    engineer = FeatureEngineer(_steps())
    original = _dataset("pandas")
    transformed, _ = engineer.fit_transform(original, target_column="y")
    assert transformed.evaluation_coverage["test"]["excluded_rows"] == 1
    copied = transformed.copy()
    copied.evaluation_coverage["test"]["steps"].clear()
    assert transformed.evaluation_coverage["test"]["steps"]
    assert len(original.test) == 3


def test_legacy_split_artifact_loads_without_new_coverage_field():
    """Previously persisted split containers remain usable after coverage was introduced."""
    original = _dataset("pandas")
    del original.evaluation_coverage
    restored = pickle.loads(pickle.dumps(original))
    assert restored.evaluation_coverage == {}
    copied = restored.copy()
    result = SkyulfPipeline(
        {
            "preprocessing": _steps(),
            "modeling": {"type": "linear_regression"},
        }
    ).fit(copied, "y")["modeling"]
    assert result["splits"]["test"].coverage["excluded_rows"] == 1


def test_fully_excluded_validation_is_explicitly_unavailable():
    """An empty eligible population is disclosed instead of silently losing the split."""
    pipeline = SkyulfPipeline(
        {"preprocessing": _steps(), "modeling": {"type": "linear_regression"}}
    )
    result = pipeline.fit(_dataset("pandas", empty_validation=True), "y")["modeling"]
    report = result["splits"]["validation"]
    assert report.coverage["input_rows"] == report.coverage["excluded_rows"] == 2
    assert report.coverage["scored_rows"] == 0
    assert report.metrics == {}
    assert "No eligible rows" in report.omitted_metrics["evaluation"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_tuned_final_test_can_be_completely_excluded(engine):
    """Successful CV fitting still reports a final external split with no eligible rows."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": _steps(),
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": "ridge_regression"},
                "strategy": "grid",
                "metric": "r2",
                "search_space": {"alpha": [1.0]},
                "cv_folds": 2,
            },
        }
    )
    dataset = _dataset(engine)
    frame = pl.DataFrame if engine == "polars" else pd.DataFrame
    dataset.test = frame({"x": [None, None], "y": [1.0, 2.0]})
    dataset.validation = None
    report = pipeline.fit(dataset, "y")["modeling"]["splits"]["test"]
    assert report.coverage["input_rows"] == report.coverage["excluded_rows"] == 2
    assert report.coverage["scored_rows"] == 0
    assert report.metrics == {}


@pytest.mark.parametrize("unknown_population", [False, True])
def test_empty_clustering_split_keeps_coverage_and_clustering_payload(unknown_population):
    """A fully excluded clustering split remains visible without supervised output keys."""
    dataset = SplitDataset(
        train=pd.DataFrame({"x": [0.0, 0.1, 0.2, 8.0, 8.1, 8.2]}),
        test=pd.DataFrame({"x": [None, None]}, dtype=float),
    )
    if unknown_population:
        dataset.evaluation_coverage["test"] = {
            "input_rows": None,
            "scored_rows": 2,
            "excluded_rows": None,
            "reason": "Original evaluation population unavailable after merging preprocessing branches",
        }
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [_steps()[0]],
            "modeling": {"type": "kmeans", "params": {"n_clusters": 2, "random_state": 42}},
        }
    )
    result = pipeline.fit(dataset, "")["modeling"]
    report = result["splits"]["test"]
    assert report.coverage["scored_rows"] == 0
    assert report.coverage["input_rows"] == (None if unknown_population else 2)
    assert report.metrics == {}
    assert "No eligible rows" in report.omitted_metrics["evaluation"]
    assert result["raw_data"]["splits"]["test"] == {"labels": [], "coverage": report.coverage}
