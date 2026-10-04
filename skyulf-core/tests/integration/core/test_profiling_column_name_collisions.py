"""Profiling helper names must not collide with user columns or category labels."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.drift import DriftCalculator


@pytest.mark.parametrize("column", ["count", "value", "count__count", "__skyulf_count__"])
def test_profile_categorical_names_preserve_values_and_frequencies(column: str) -> None:
    """User names resembling aggregate fields must retain the correct category/count pairs."""
    frame = pl.DataFrame({column: ["count"] * 96 + ["value"] * 3 + ["__skyulf_count__"]})

    profile = EDAAnalyzer(frame).analyze()

    stats = profile.columns[column].categorical_stats
    assert stats is not None
    assert stats.top_k == [
        {"value": "count", "count": 96},
        {"value": "value", "count": 3},
        {"value": "__skyulf_count__", "count": 1},
    ]
    assert stats.unique_count == 3
    assert stats.rare_labels_count == 2
    assert profile.column_count == 1
    assert profile.sample_data == frame.to_dicts()


@pytest.mark.parametrize("column", ["count", "value", "count__count", "__skyulf_count__"])
@pytest.mark.parametrize("shifted", [False, True])
def test_categorical_drift_names_preserve_distribution_identity(column: str, shifted: bool) -> None:
    """A category/count name collision must neither abort drift nor change the measured PSI."""
    reference = pl.DataFrame({column: ["count"] * 80 + ["value"] * 20})
    current = pl.DataFrame(
        {column: ["count"] * (20 if shifted else 80) + ["value"] * (80 if shifted else 20)}
    )
    original_reference, original_current = reference.clone(), current.clone()

    report = DriftCalculator(reference, current).calculate_drift()

    result = report.column_drifts[column]
    assert result.column == column
    assert len(result.metrics) == 1
    assert result.metrics[0].metric == "psi_categorical"
    assert result.metrics[0].value == pytest.approx(1.2 * np.log(4) if shifted else 0)
    assert result.drift_detected is shifted
    assert report.drifted_columns_count == int(shifted)
    assert_frame_equal(reference, original_reference)
    assert_frame_equal(current, original_current)


@pytest.mark.parametrize("boolean_target", [False, True])
@pytest.mark.parametrize("exclude_candidate", [False, True])
def test_profile_nominal_target_preserves_real_features_across_repeated_analysis(
    boolean_target: bool, exclude_candidate: bool
) -> None:
    """Nominal targets must preserve real encoded-name features and their correlation matrices."""
    rng = np.random.default_rng(190198)
    n_rows = 120
    frame = pl.DataFrame(
        {
            "x": rng.normal(size=n_rows),
            "target": ([False, True] if boolean_target else ["left", "right"]) * (n_rows // 2),
            "target_encoded": rng.normal(100, 10, size=n_rows),
            "target_encoded_1": rng.normal(200, 20, size=n_rows),
            "target_encoded_2": rng.normal(300, 30, size=n_rows),
            "__skyulf_target_encoded__": rng.normal(400, 40, size=n_rows),
        }
    )
    excluded = ["target_encoded_2"] if exclude_candidate else []
    selected = [col for col in frame.columns if col not in excluded]
    features = [col for col in selected if col != "target"]
    expected_correlations = np.corrcoef(frame.select(features).to_numpy(), rowvar=False)
    analyzer = EDAAnalyzer(frame)

    for _ in range(2):
        profile = analyzer.analyze(target_col="target", exclude_cols=excluded)

        assert_frame_equal(analyzer.df, frame)
        assert_frame_equal(analyzer.lazy_df.collect(), frame)
        assert profile.column_count == len(selected)
        assert profile.sample_data == frame.select(selected).to_dicts()
        assert profile.correlations is not None
        assert profile.correlations.columns == features
        np.testing.assert_allclose(profile.correlations.values, expected_correlations, atol=1e-12)
        assert profile.correlations_with_target is None
        assert profile.causal_target_exclusion_reason == "categorical"


def test_profile_target_name_collision_keeps_causal_graph_available() -> None:
    """Real encoded-name features must retain their identity when a nominal target is omitted."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(198)
    frame = pl.DataFrame(
        {
            "x": rng.normal(size=120),
            "target": ["left", "right"] * 60,
            "target_encoded": rng.normal(100, 10, size=120),
            "target_encoded_1": rng.normal(200, 20, size=120),
        }
    )

    profile = EDAAnalyzer(frame).analyze(target_col="target")

    assert profile.causal_graph is not None
    assert profile.correlations is not None
    assert profile.correlations_with_target is None
    node_names = [node.id for node in profile.causal_graph.nodes]
    assert node_names == profile.correlations.columns == ["x", "target_encoded", "target_encoded_1"]
    assert len(set(node_names)) == 3


def test_profile_nominal_target_preserves_frame_when_analysis_fails(monkeypatch) -> None:
    """A later analysis error must not add target codes or overwrite source state."""
    frame = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "target": ["left", "left", "right"], "target_encoded": [5, 7, 9]}
    )
    analyzer = EDAAnalyzer(frame)

    def fail_multivariate(*args, **kwargs):
        """Simulate a downstream analytics failure after target preparation."""
        raise RuntimeError("multivariate analysis failed")

    monkeypatch.setattr(analyzer, "_compute_multivariate", fail_multivariate)

    with pytest.raises(RuntimeError, match="multivariate analysis failed"):
        analyzer.analyze(target_col="target")

    assert_frame_equal(analyzer.df, frame)
    assert_frame_equal(analyzer.lazy_df.collect(), frame)
