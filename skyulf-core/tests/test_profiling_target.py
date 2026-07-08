"""Tests for skyulf.profiling._analyzer.target.TargetMixin."""

import numpy as np
import polars as pl
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling.analyzer import EDAAnalyzer


def _numeric_target_df(n: int = 60) -> pl.DataFrame:
    """Numeric features correlated with a numeric target."""
    rng = np.random.default_rng(31)
    a = rng.normal(0, 1, n)
    b = a * 2 + rng.normal(0, 0.1, n)
    target = a * 3 + rng.normal(0, 0.2, n)
    return pl.DataFrame({"a": a, "b": b, "target": target})


def test_calculate_target_correlations_happy_path() -> None:
    """Correlations should be returned sorted by absolute magnitude, descending."""
    analyzer = EDAAnalyzer(_numeric_target_df())
    corrs = analyzer._calculate_target_correlations("target", ["a", "b", "target"])
    assert set(corrs.keys()) == {"a", "b"}
    values = list(corrs.values())
    assert abs(values[0]) >= abs(values[-1])


def test_calculate_target_correlations_no_features_returns_empty() -> None:
    """numeric_cols containing only the target itself should return {} (line 21)."""
    analyzer = EDAAnalyzer(_numeric_target_df())
    assert analyzer._calculate_target_correlations("target", ["target"]) == {}


def test_calculate_target_correlations_exception_returns_empty(monkeypatch) -> None:
    """An internal error (e.g. lazy_df.select raising) should be caught (lines 40-42)."""
    analyzer = EDAAnalyzer(_numeric_target_df())

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(analyzer.lazy_df, "select", _boom)
    assert analyzer._calculate_target_correlations("target", ["a", "b"]) == {}


def test_calculate_categorical_target_associations_happy_path() -> None:
    """Categorical target vs numeric features should return eta-values in [0, 1]."""
    rng = np.random.default_rng(32)
    n = 60
    cat_target = rng.choice(["low", "high"], size=n)
    a = np.where(cat_target == "high", rng.normal(5, 1, n), rng.normal(0, 1, n))
    df = pl.DataFrame({"a": a, "target": cat_target})
    analyzer = EDAAnalyzer(df)
    assoc = analyzer._calculate_categorical_target_associations("target", ["a"])
    assert "a" in assoc
    assert 0.0 <= assoc["a"] <= 1.0


def test_calculate_categorical_target_associations_constant_feature() -> None:
    """A constant feature column (ss_total == 0) should yield an association of 0.0."""
    n = 40
    df = pl.DataFrame(
        {
            "a": [1.0] * n,
            "target": ["x" if i % 2 == 0 else "y" for i in range(n)],
        }
    )
    analyzer = EDAAnalyzer(df)
    assoc = analyzer._calculate_categorical_target_associations("target", ["a"])
    assert assoc["a"] == 0.0


def test_calculate_categorical_target_associations_exception_returns_empty(monkeypatch) -> None:
    """An internal error should be caught and an empty dict returned (lines 82-84)."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "target": ["x", "y", "x"]})
    analyzer = EDAAnalyzer(df)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pl.DataFrame, "group_by", _boom)
    assert analyzer._calculate_categorical_target_associations("target", ["a"]) == {}


def test_calculate_target_interactions_numeric_target() -> None:
    """Numeric target with a categorical feature should produce box-plot interactions."""
    rng = np.random.default_rng(33)
    n = 60
    cat = rng.choice(["A", "B", "C"], size=n)
    target = np.where(cat == "A", rng.normal(0, 1, n), rng.normal(5, 1, n))
    df = pl.DataFrame({"cat": cat, "target": target})
    analyzer = EDAAnalyzer(df)
    interactions = analyzer._calculate_target_interactions(
        "target", ["cat"], is_target_numeric=True
    )
    assert len(interactions) == 1
    assert interactions[0].feature == "cat"
    assert interactions[0].plot_type == "boxplot"
    assert interactions[0].p_value is not None


def test_calculate_target_interactions_skips_high_cardinality_group() -> None:
    """A grouping column with >20 unique values should be skipped (line 104)."""
    rng = np.random.default_rng(34)
    n = 60
    high_card = [f"id_{i}" for i in range(n)]
    target = rng.normal(0, 1, n)
    df = pl.DataFrame({"high_card": high_card, "target": target})
    analyzer = EDAAnalyzer(df)
    interactions = analyzer._calculate_target_interactions(
        "target", ["high_card"], is_target_numeric=True
    )
    assert interactions == []


def test_calculate_target_interactions_anova_failure_is_caught(monkeypatch) -> None:
    """f_oneway raising should be caught, leaving p_value=None (lines 168-169)."""
    rng = np.random.default_rng(35)
    n = 60
    cat = rng.choice(["A", "B", "C"], size=n)
    target = rng.normal(0, 1, n)
    df = pl.DataFrame({"cat": cat, "target": target})
    analyzer = EDAAnalyzer(df)

    import scipy.stats

    def _boom(*args, **kwargs):
        raise RuntimeError("anova boom")

    monkeypatch.setattr(scipy.stats, "f_oneway", _boom)
    interactions = analyzer._calculate_target_interactions(
        "target", ["cat"], is_target_numeric=True
    )
    assert len(interactions) == 1
    assert interactions[0].p_value is None


def test_calculate_target_interactions_categorical_target() -> None:
    """A categorical target with a numeric feature should group by target (lines 99-100)."""
    rng = np.random.default_rng(36)
    n = 60
    target = rng.choice(["low", "high"], size=n)
    feature = np.where(target == "high", rng.normal(5, 1, n), rng.normal(0, 1, n))
    df = pl.DataFrame({"feature": feature, "target": target})
    analyzer = EDAAnalyzer(df)
    interactions = analyzer._calculate_target_interactions(
        "target", ["feature"], is_target_numeric=False
    )
    assert len(interactions) == 1
    assert interactions[0].feature == "feature"


def test_calculate_target_interactions_skips_null_group_rows() -> None:
    """A group with an entirely-null value column should be skipped (line 130)."""
    n = 40
    cat = ["A"] * (n // 2) + ["B"] * (n // 2)
    # Group "B" has an entirely-null target so its min/median etc. are all None.
    target = [1.0] * (n // 2) + [None] * (n // 2)
    df = pl.DataFrame({"cat": cat, "target": target})
    analyzer = EDAAnalyzer(df)
    interactions = analyzer._calculate_target_interactions(
        "target", ["cat"], is_target_numeric=True
    )
    assert len(interactions) == 1
    names = {plot.name for plot in interactions[0].data}
    assert "B" not in names
    assert "A" in names


def test_calculate_target_interactions_outer_exception(monkeypatch) -> None:
    """An internal error (e.g. n_unique raising) should be caught, returning [] (lines 183-185)."""
    df = pl.DataFrame({"cat": ["A", "B", "A"], "target": [1.0, 2.0, 3.0]})
    analyzer = EDAAnalyzer(df)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pl.Series, "n_unique", _boom)
    interactions = analyzer._calculate_target_interactions(
        "target", ["cat"], is_target_numeric=True
    )
    assert interactions == []


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing ``age``/``income`` rows — closer to production data than
    the small synthetic frames used elsewhere in this file.
    """

    def test_churn_target_correlations_handle_missing_values(self) -> None:
        df = load_sample_dataset("customers", engine="polars")
        analyzer = EDAAnalyzer(df)

        # churned is a 0/1 numeric label; age/income both have missing rows,
        # which pl.corr should silently exclude pairwise.
        corrs = analyzer._calculate_target_correlations("churned", ["age", "income", "churned"])
        assert set(corrs.keys()) == {"age", "income"}
        assert all(-1.0 <= v <= 1.0 for v in corrs.values())
