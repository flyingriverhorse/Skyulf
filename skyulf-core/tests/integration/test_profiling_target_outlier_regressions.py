"""Public EDA regressions for aggregation names and sampled row provenance."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.ensemble import IsolationForest

from skyulf.profiling.analyzer import EDAAnalyzer


def _target_frame() -> pl.DataFrame:
    """Keep null and constant controls alongside two separated observed classes."""
    return pl.DataFrame(
        {
            "species": ["setosa"] * 24 + ["virginica"] * 24 + [None] * 24,
            "measurement": [1.0, 2.0, 3.0, None] * 6 + [10.0, 11.0, 12.0, None] * 6 + [1000.0] * 24,
            "constant": [7.0] * 72,
            "all_missing": [None] * 72,
        },
        schema_overrides={"species": pl.Categorical, "all_missing": pl.Float64},
    )


@pytest.mark.parametrize(
    "target_name", ["species", "mean", "n", "min", "q1", "median", "q3", "max", "group"]
)
def test_target_aggregation_names_preserve_public_statistics(target_name) -> None:
    """Renaming a categorical target must retain its associations and interaction summaries."""
    frame = _target_frame().rename({"species": target_name})
    analyzer = EDAAnalyzer(frame)

    profile = analyzer.analyze(target_col=target_name)

    # On complete pairs, SS_between=729 and SS_total=753; missing targets add neither.
    assert profile.target_correlations == pytest.approx(
        {"measurement": np.sqrt(243 / 251), "constant": 0.0}
    )
    assert profile.target_interactions is not None
    interactions = {interaction.feature: interaction for interaction in profile.target_interactions}
    assert set(interactions) == {"measurement", "constant"}
    boxes = {box.name: box.stats.model_dump() for box in interactions["measurement"].data}
    assert boxes == {
        "setosa": {"min": 1.0, "q1": 1.0, "median": 2.0, "q3": 3.0, "max": 3.0},
        "virginica": {"min": 10.0, "q1": 10.0, "median": 11.0, "q3": 12.0, "max": 12.0},
    }
    assert interactions["measurement"].p_value is not None
    assert 0.0 <= interactions["measurement"].p_value <= 1.0
    assert_frame_equal(analyzer.df, frame)


@pytest.mark.parametrize("group_name", ["species", "min", "q1", "median", "q3", "max", "group"])
def test_numeric_target_group_names_preserve_interactions(group_name) -> None:
    """Categorical feature names must not collide with numeric-target box-plot fields."""
    frame = _target_frame().select("species", "measurement").rename({"species": group_name})

    profile = EDAAnalyzer(frame).analyze(target_col="measurement")

    assert profile.target_interactions is not None
    assert len(profile.target_interactions) == 1
    interaction = profile.target_interactions[0]
    assert interaction.feature == group_name
    assert {box.name: box.stats.median for box in interaction.data} == {
        "setosa": 2.0,
        "virginica": 11.0,
    }


@pytest.mark.parametrize("target_name", ["species", "count", "value"])
@pytest.mark.parametrize("labels", [[0, 1, 2] * 40, ["count", "value", "other"] * 40])
def test_count_target_preserves_balanced_class_recommendations(target_name, labels) -> None:
    """A target named count must profile successfully and keep balanced class advice."""
    rng = np.random.default_rng(245)
    frame = pl.DataFrame(
        {target_name: labels, "x": rng.normal(size=120), "z": rng.normal(size=120)}
    )
    analyzer = EDAAnalyzer(frame)

    profile = analyzer.analyze(target_col=target_name)

    balanced = [rec for rec in profile.recommendations if rec.reason == "Balanced Target"]
    assert len(balanced) == 1
    assert balanced[0].column == target_name
    assert balanced[0].action == "Info"
    assert "Ratio: 1.00" in balanced[0].suggestion
    assert set(profile.columns) == set(frame.columns)
    assert profile.sample_data == frame.to_dicts()
    assert_frame_equal(analyzer.df, frame)


@pytest.mark.parametrize("target_name", ["species", "count"])
def test_count_target_preserves_imbalance_ratio_excluding_missing_labels(target_name) -> None:
    """Literal count/value classes and missing labels must retain the observed 5/100 ratio."""
    labels = ["count"] * 100 + ["value"] * 10 + ["rare"] * 5 + [None]
    rng = np.random.default_rng(245)
    frame = pl.DataFrame(
        {target_name: labels, "x": rng.normal(size=116), "z": rng.normal(size=116)}
    )

    profile = EDAAnalyzer(frame).analyze(target_col=target_name)

    imbalanced = [rec for rec in profile.recommendations if rec.reason == "Imbalanced Target"]
    assert len(imbalanced) == 1
    assert imbalanced[0].column == target_name
    assert imbalanced[0].action == "Resample"
    assert "Ratio: 0.05" in imbalanced[0].suggestion
    assert profile.sample_data == frame.to_dicts()


@pytest.mark.parametrize("row_count", [50000, 50001])
@pytest.mark.parametrize("filtered_prefix", [0, 7])
def test_outlier_indices_refer_to_filtered_input_without_changing_scores(
    row_count, filtered_prefix, monkeypatch
) -> None:
    """Every reported outlier must locate its own row without adding a detector feature."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "4")
    rng = np.random.default_rng(74)
    total_rows = row_count + filtered_prefix
    measurements = rng.normal(size=total_rows)
    measurements[-100:] += 40
    numeric_cols = ["row_id", "measurement", "__skyulf_outlier_row__"]
    frame = pl.DataFrame(
        {
            "row_id": np.arange(total_rows),
            "measurement": measurements,
            "__skyulf_outlier_row__": rng.normal(size=total_rows),
            "species": ["setosa" if i % 2 else "virginica" for i in range(total_rows)],
        }
    )
    filters = (
        [{"column": "row_id", "operator": ">=", "value": filtered_prefix}]
        if filtered_prefix
        else None
    )
    filtered = frame.slice(filtered_prefix)
    analyzer = EDAAnalyzer(frame)

    profile = analyzer.analyze(target_col="species", filters=filters)

    assert profile.row_count == row_count
    assert profile.outliers is not None
    assert profile.outliers.top_outliers
    expected_sample = filtered.select(numeric_cols)
    if row_count > 50000:
        expected_sample = expected_sample.sample(n=50000, with_replacement=False, seed=42)
    detector = IsolationForest(random_state=42, contamination=0.05, n_jobs=1)
    detector.fit(expected_sample.to_numpy())
    expected_scores = dict(
        zip(
            expected_sample["row_id"],
            detector.decision_function(expected_sample.to_numpy()),
            strict=True,
        )
    )
    for point in profile.outliers.top_outliers:
        assert point.index == point.values["row_id"] - filtered_prefix
        assert point.values == filtered.select(numeric_cols).row(point.index, named=True)
        assert point.score == pytest.approx(expected_scores[point.values["row_id"]])
    assert_frame_equal(analyzer.df, filtered)
