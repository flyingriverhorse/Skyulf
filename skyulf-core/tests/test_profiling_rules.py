"""Tests for skyulf.profiling._analyzer.rules.RulesMixin."""

import numpy as np
import polars as pl
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling._analyzer import rules as rules_mod
from skyulf.profiling.analyzer import EDAAnalyzer


def _classification_df(n: int = 80) -> pl.DataFrame:
    """Numeric features + a low-cardinality string target for classification rules."""
    rng = np.random.default_rng(21)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    target = np.where(a + b > 0, "high", "low")
    return pl.DataFrame({"a": a, "b": b, "target": target})


def _regression_df(n: int = 80) -> pl.DataFrame:
    """Numeric features + a numeric target for regression rules."""
    rng = np.random.default_rng(22)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    target = a * 2 + b + rng.normal(0, 0.1, n)
    return pl.DataFrame({"a": a, "b": b, "target": target})


def test_discover_rules_classification_auto_detected() -> None:
    """Auto task-type detection should classify a string target as classification."""
    analyzer = EDAAnalyzer(_classification_df())
    tree = analyzer._discover_rules(["a", "b"], "target")
    assert tree is not None
    assert tree.rules
    assert tree.accuracy is not None
    assert 0.0 <= tree.accuracy <= 1.0


def test_discover_rules_regression_auto_detected() -> None:
    """Auto task-type detection should treat a numeric target as regression."""
    analyzer = EDAAnalyzer(_regression_df())
    tree = analyzer._discover_rules(["a", "b"], "target")
    assert tree is not None
    assert tree.rules is not None
    assert any("Value" in r for r in tree.rules)


def test_discover_rules_explicit_classification_task_type() -> None:
    """Explicit task_type='classification' should force is_regression=False (lines 40-41)."""
    analyzer = EDAAnalyzer(_classification_df())
    tree = analyzer._discover_rules(["a", "b"], "target", task_type="classification")
    assert tree is not None
    assert tree.rules is not None
    assert any("Confidence" in r for r in tree.rules)


def test_discover_rules_explicit_regression_task_type() -> None:
    """Explicit task_type='regression' should force is_regression=True."""
    analyzer = EDAAnalyzer(_regression_df())
    tree = analyzer._discover_rules(["a", "b"], "target", task_type="regression")
    assert tree is not None
    assert tree.rules is not None
    assert any("Value" in r for r in tree.rules)


def test_discover_rules_caps_high_cardinality_classes() -> None:
    """More than 10 distinct target classes should be capped to top 10 + 'Other' (lines 84-91)."""
    rng = np.random.default_rng(23)
    n = 300
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    # 15 categories with skewed frequency so some fall outside the top 10.
    categories = [f"cat_{i}" for i in range(15)]
    weights = np.array([30 - i for i in range(15)], dtype=float)
    weights /= weights.sum()
    target = rng.choice(categories, size=n, p=weights)
    df = pl.DataFrame({"a": a, "b": b, "target": target})
    analyzer = EDAAnalyzer(df)
    tree = analyzer._discover_rules(["a", "b"], "target", task_type="classification")
    assert tree is not None
    assert tree.rules is not None
    # Rules should reference at most 10 real categories plus a possible "Other" bucket.
    referenced = {r.split(" THEN ")[1].split(" (")[0] for r in tree.rules}
    assert len(referenced) <= 11


def test_discover_rules_with_categorical_feature() -> None:
    """A categorical feature column should be ordinal-encoded via factorize (lines 65-69)."""
    rng = np.random.default_rng(24)
    n = 80
    a = rng.normal(0, 1, n)
    cat = rng.choice(["red", "green", "blue"], size=n)
    target = np.where(a > 0, "high", "low")
    df = pl.DataFrame({"a": a, "cat": cat, "target": target})
    analyzer = EDAAnalyzer(df)
    tree = analyzer._discover_rules(["a", "cat"], "target")
    assert tree is not None
    assert tree.rules


def test_discover_rules_sklearn_unavailable(monkeypatch) -> None:
    """SKLEARN_AVAILABLE=False should short-circuit to None (line 23)."""
    analyzer = EDAAnalyzer(_classification_df())
    monkeypatch.setattr(rules_mod, "SKLEARN_AVAILABLE", False)
    assert analyzer._discover_rules(["a", "b"], "target") is None


def test_discover_rules_missing_column_returns_none() -> None:
    """A non-existent feature column should raise internally and be caught (lines 214-216)."""
    analyzer = EDAAnalyzer(_classification_df())
    tree = analyzer._discover_rules(["does_not_exist"], "target")
    assert tree is None


def test_discover_rules_outer_exception(monkeypatch) -> None:
    """DecisionTreeClassifier.fit raising should be caught by the outer except (214-216)."""
    analyzer = EDAAnalyzer(_classification_df())
    from sklearn.tree import DecisionTreeClassifier

    def _boom(self, *args, **kwargs):
        raise RuntimeError("tree exploded")

    monkeypatch.setattr(DecisionTreeClassifier, "fit", _boom)
    tree = analyzer._discover_rules(["a", "b"], "target")
    assert tree is None


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing ``age``/``income`` feature values — closer to
    production data than the small synthetic frames used elsewhere in this
    file.
    """

    def test_discover_rules_predicts_churned_from_age_and_income(self) -> None:
        df = load_sample_dataset("customers", engine="polars")
        analyzer = EDAAnalyzer(df)
        tree = analyzer._discover_rules(["age", "income"], "churned", task_type="classification")

        # Missing age/income are mean-imputed internally, so all 15 rows train.
        assert tree is not None
        assert tree.rules
        assert tree.accuracy is not None
        assert 0.0 <= tree.accuracy <= 1.0
