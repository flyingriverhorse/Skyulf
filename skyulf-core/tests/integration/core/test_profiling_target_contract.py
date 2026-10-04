"""Target selection must preserve category semantics and column exclusions."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.datasets import load_iris

from skyulf.profiling import EDAAnalyzer


@pytest.mark.parametrize("kind", ["string", "categorical", "enum", "boolean", "codes"])
def test_nominal_targets_do_not_become_numeric_analysis_variables(kind: str) -> None:
    """Category labels must retain associations without arbitrary Pearson or causal codes."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(236)
    target = pl.Series("species", ["a", "b", "c"] * 40)
    if kind == "categorical":
        target = target.cast(pl.Categorical)
    elif kind == "enum":
        target = target.cast(pl.Enum(["a", "b", "c"]))
    elif kind == "boolean":
        target = pl.Series("species", [True, False] * 60)
    elif kind == "codes":
        target = pl.Series("species", [0, 1, 2] * 40)
    frame = pl.DataFrame({"x": rng.normal(size=120), "z": rng.normal(size=120)}).with_columns(
        target
    )
    analyzer = EDAAnalyzer(frame)

    profile = analyzer.analyze(target_col="species")

    assert profile.correlations_with_target is None
    assert profile.causal_target_exclusion_reason == "categorical"
    assert profile.causal_graph is not None
    assert [node.id for node in profile.causal_graph.nodes] == ["x", "z"]
    assert [node.label for node in profile.causal_graph.nodes] == ["x", "z"]
    assert profile.target_correlations is not None
    assert profile.target_interactions is not None
    assert set(profile.target_correlations) == {"x", "z"}
    assert len(profile.target_interactions) == 2
    assert profile.sample_data == frame.to_dicts()
    assert_frame_equal(analyzer.df, frame)


def test_iris_row_order_preserves_category_associations_and_numeric_graph() -> None:
    """Reordering identical classes must not change the numeric analysis through category codes."""
    pytest.importorskip("causallearn")
    iris = load_iris()
    results = []
    for order in ([0, 1, 2], [0, 2, 1]):
        indices = np.concatenate([np.flatnonzero(iris.target == group) for group in order])
        frame = pl.DataFrame(iris.data[indices], schema=list(iris.feature_names)).with_columns(
            pl.Series("species", iris.target_names[iris.target[indices]])
        )
        results.append(EDAAnalyzer(frame).analyze(target_col="species"))

    first, second = results
    assert first.correlations_with_target is second.correlations_with_target is None
    assert second.target_correlations is not None
    assert first.target_correlations == pytest.approx(second.target_correlations)
    assert first.causal_graph is not None and second.causal_graph is not None
    assert first.causal_graph == second.causal_graph
    assert [node.id for node in first.causal_graph.nodes] == list(iris.feature_names)


@pytest.mark.parametrize("task_type", [None, "Regression", "Classification"])
def test_numeric_target_eligibility_respects_the_selected_task(task_type: str | None) -> None:
    """Classification must not restore ordinal target semantics through a numeric storage dtype."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(237)
    frame = pl.DataFrame({name: rng.normal(size=120) for name in ["x", "z", "response"]})

    profile = EDAAnalyzer(frame).analyze(target_col="response", task_type=task_type)

    assert profile.causal_graph is not None
    names = [node.id for node in profile.causal_graph.nodes]
    if task_type == "Classification":
        assert profile.correlations_with_target is None
        assert profile.causal_target_exclusion_reason == "categorical"
        assert names == ["x", "z"]
        assert profile.target_correlations == pytest.approx({"x": 1.0, "z": 1.0})
    else:
        assert profile.correlations_with_target is not None
        assert profile.correlations_with_target.columns == ["x", "z", "response"]
        assert profile.causal_target_exclusion_reason is None
        assert names == ["x", "z", "response"]


def test_excluded_selected_target_cannot_drive_rules_or_inferred_task() -> None:
    """An excluded target must not appear in rule text after it disappears from column profiles."""
    rng = np.random.default_rng(239)
    frame = pl.DataFrame(
        {
            "x": np.r_[rng.normal(4, 0.1, 30), rng.normal(8, 0.1, 30)],
            "species": ["setosa"] * 30 + ["virginica"] * 30,
        }
    )
    analyzer = EDAAnalyzer(frame)

    for exclusions in (["species"], None):
        profile = analyzer.analyze(target_col="species", exclude_cols=exclusions)
        assert profile.rule_tree is None
        assert profile.task_type is None
        assert profile.target_correlations == {}
        assert profile.target_interactions is None
        assert profile.causal_target_exclusion_reason == "excluded"
        assert "species" not in profile.columns
        assert profile.sample_data is not None
        assert all("species" not in row for row in profile.sample_data)


def test_explicit_regression_retains_a_numeric_target_with_few_distinct_values() -> None:
    """A regression choice must retain measured numeric targets despite cardinality inference."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(236238)
    frame = pl.DataFrame(
        {"x": rng.normal(size=120), "z": rng.normal(size=120), "rating": [1, 2, 3] * 40}
    )

    profile = EDAAnalyzer(frame).analyze(target_col="rating", task_type="Regression")

    assert profile.correlations_with_target is not None
    assert profile.correlations_with_target.columns == ["x", "z", "rating"]
    assert profile.causal_target_exclusion_reason is None
    assert profile.causal_graph is not None
    assert [node.id for node in profile.causal_graph.nodes] == ["x", "z", "rating"]


def test_target_omission_reason_survives_an_unavailable_graph() -> None:
    """A report without enough numeric variables must still explain an omitted categorical target."""
    frame = pl.DataFrame({"species": ["setosa", "virginica"] * 30})

    profile = EDAAnalyzer(frame).analyze(target_col="species")

    assert profile.causal_graph is None
    assert profile.correlations_with_target is None
    assert profile.causal_target_exclusion_reason == "categorical"


def test_unsupported_target_does_not_fail_or_get_synthetic_codes() -> None:
    """An absent target must leave feature analysis available without manufacturing a variable."""
    rng = np.random.default_rng(238)
    frame = pl.DataFrame({"x": rng.normal(size=60), "z": rng.normal(size=60)})

    profile = EDAAnalyzer(frame).analyze(target_col="absent")

    assert profile.correlations is not None
    assert profile.correlations_with_target is None
    assert profile.rule_tree is None
    assert profile.causal_target_exclusion_reason == "unsupported"
