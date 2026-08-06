"""Native Polars parity tests for ``evaluate_clustering_model``.

Covers the Wave 2 "Candidate B" gate from
``temp/skyulf-core-pandas-polars-audit-2026-08-05.md``: raw and wrapped
Polars frames must reproduce the exact Pandas-path metrics, centroids,
profiles, and reference crosstab without a full-frame ``to_pandas()``
conversion.
"""

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.modeling._evaluation.clustering import evaluate_clustering_model

_FIXTURE = {
    "id": ["r1", "r2", "r3", "r4", "r5", "r6"],
    "x": [0.0, 0.0, 10.0, 10.0, 20.0, 20.0],
    "y": [0.0, 2.0, 10.0, 12.0, 20.0, 22.0],
    "flag": [True, False, True, False, True, False],
    "note": ["a", "b", "c", "d", "e", "f"],
    "species": ["setosa", "setosa", "versicolor", "versicolor", "virginica", "virginica"],
}
_LABELS = [0, 0, 1, 1, 2, 2]

_GOLDEN_METRICS = {
    "n_clusters": 3.0,
    "silhouette_score": 0.8394922657564039,
    "silhouette_sample_size": 6.0,
    "calinski_harabasz_score": 160.0,
    "davies_bouldin_score": 0.15811388300841897,
}
_GOLDEN_CENTROIDS = [
    {"cluster_id": 0, "size": 2, "percentage": 33.33, "center": {"x": 0.0, "y": 1.0, "flag": 0.5}},
    {
        "cluster_id": 1,
        "size": 2,
        "percentage": 33.33,
        "center": {"x": 10.0, "y": 11.0, "flag": 0.5},
    },
    {
        "cluster_id": 2,
        "size": 2,
        "percentage": 33.33,
        "center": {"x": 20.0, "y": 21.0, "flag": 0.5},
    },
]
_GOLDEN_CROSSTAB = {
    "0": {"setosa": 2},
    "1": {"versicolor": 2},
    "2": {"virginica": 2},
}


def _frames() -> dict[str, object]:
    return {
        "pandas": pd.DataFrame(_FIXTURE),
        "raw_polars": pl.DataFrame(_FIXTURE),
        "wrapped_polars": SkyulfPolarsWrapper(pl.DataFrame(_FIXTURE)),
    }


@pytest.mark.parametrize("engine", ["pandas", "raw_polars", "wrapped_polars"])
def test_evaluate_clustering_model_matches_golden_metrics(engine: str) -> None:
    df = _frames()[engine]
    report = evaluate_clustering_model(
        None, df, _LABELS, dataset_name="fixture", reference_column="species"
    )
    assert report.metrics == _GOLDEN_METRICS


@pytest.mark.parametrize("engine", ["pandas", "raw_polars", "wrapped_polars"])
def test_evaluate_clustering_model_matches_golden_centroids(engine: str) -> None:
    df = _frames()[engine]
    report = evaluate_clustering_model(
        None, df, _LABELS, dataset_name="fixture", reference_column="species"
    )
    assert report.clustering is not None
    centroids = [
        {"cluster_id": c.cluster_id, "size": c.size, "percentage": c.percentage, "center": c.center}
        for c in report.clustering.centroids
    ]
    assert centroids == _GOLDEN_CENTROIDS
    assert [c.profile for c in report.clustering.centroids] == [
        "Low x, Low y",
        "Low x, Low y",
        "High x, High y",
    ]


@pytest.mark.parametrize("engine", ["pandas", "raw_polars", "wrapped_polars"])
def test_evaluate_clustering_model_matches_golden_crosstab(engine: str) -> None:
    df = _frames()[engine]
    report = evaluate_clustering_model(
        None, df, _LABELS, dataset_name="fixture", reference_column="species"
    )
    assert report.clustering is not None
    assert report.clustering.reference_crosstab == _GOLDEN_CROSSTAB
    assert report.clustering.reference_column == "species"


@pytest.mark.parametrize("engine", ["pandas", "raw_polars", "wrapped_polars"])
def test_evaluate_clustering_model_missing_reference_column_is_none(engine: str) -> None:
    df = _frames()[engine]
    report = evaluate_clustering_model(
        None, df, _LABELS, dataset_name="fixture", reference_column="missing"
    )
    assert report.clustering is not None
    assert report.clustering.reference_crosstab is None
    assert report.clustering.reference_column == "missing"


@pytest.mark.parametrize("engine", ["pandas", "raw_polars"])
def test_evaluate_clustering_model_nan_labels_raise_value_error(engine: str) -> None:
    df = _frames()[engine]
    with pytest.raises(ValueError, match="Input y contains NaN"):
        evaluate_clustering_model(
            None,
            df,
            [0, math.nan, 1, 1, 2, 2],
            dataset_name="fixture",
            reference_column="species",
        )


@pytest.mark.parametrize("engine", ["pandas", "raw_polars"])
def test_evaluate_clustering_model_no_numeric_columns_raises_like_sklearn(engine: str) -> None:
    """Non-numeric-only frames (after excluding the reference column) must fail
    identically on both engines, not with a Polars-specific row-count error."""
    fixture = {"id": ["a", "b", "c"], "note": ["x", "y", "z"], "ref": ["p", "q", "r"]}
    df = pd.DataFrame(fixture) if engine == "pandas" else pl.DataFrame(fixture)
    with pytest.raises(ValueError, match="0 feature"):
        evaluate_clustering_model(None, df, [0, 1, 1], dataset_name="empty", reference_column="ref")


def test_evaluate_clustering_model_polars_reference_dtype_preserved() -> None:
    """Reference values keep their original dtype through the native crosstab path."""
    fixture = dict(_FIXTURE)
    fixture["numeric_ref"] = [1, 1, 2, 2, 3, 3]
    df = pl.DataFrame(fixture)
    report = evaluate_clustering_model(
        None, df, _LABELS, dataset_name="fixture", reference_column="numeric_ref"
    )
    assert report.clustering is not None
    assert report.clustering.reference_crosstab == {"0": {"1": 2}, "1": {"2": 2}, "2": {"3": 2}}


def test_evaluate_clustering_model_raw_polars_avoids_full_frame_pandas_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native Polars path must not call ``to_pandas()`` on the full input frame."""
    df = pl.DataFrame(_FIXTURE)
    original_to_pandas = pl.DataFrame.to_pandas
    called = {"count": 0}

    def _tracking_to_pandas(self: pl.DataFrame, *args: object, **kwargs: object) -> pd.DataFrame:
        called["count"] += 1
        return original_to_pandas(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", _tracking_to_pandas)
    evaluate_clustering_model(None, df, _LABELS, dataset_name="fixture", reference_column="species")
    assert called["count"] == 0


def test_evaluate_clustering_model_polars_path_peak_memory_not_worse_than_pandas() -> None:
    """The native Polars path must not use materially more peak memory than the
    existing Pandas path on the same representative mixed frame.

    Both paths run KMeans-scale silhouette computation (O(n^2) pairwise
    distances at this row count), so this compares relative peak memory
    between engines rather than asserting an absolute magic-number bound.
    """
    import gc
    import tracemalloc

    rng = np.random.default_rng(42)
    rows, numeric_cols = 2_000, 30
    data = {f"num_{i}": rng.random(rows) for i in range(numeric_cols)}
    data["id"] = [f"r{i}" for i in range(rows)]
    data["species"] = rng.choice(["a", "b", "c"], size=rows)
    labels = rng.integers(0, 3, size=rows)

    def _peak_for(df: object) -> int:
        gc.collect()
        tracemalloc.start()
        try:
            evaluate_clustering_model(
                None, df, labels, dataset_name="wide", reference_column="species"
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    pandas_peak = _peak_for(pd.DataFrame(data))
    polars_peak = _peak_for(pl.DataFrame(data))

    # Generous multiplier: the goal is catching an accidental full extra
    # Pandas-sized copy on the Polars path, not a tight performance gate.
    assert polars_peak <= pandas_peak * 1.5
