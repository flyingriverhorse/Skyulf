"""Tests for skyulf.profiling._analyzer.rules.RulesMixin."""

import numpy as np
import polars as pl
import pytest
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


@pytest.mark.parametrize("public_profile", [False, True], ids=["rules", "profile"])
@pytest.mark.parametrize("target_kind", ["string", "numeric", "categorical_subset", "missing"])
def test_rule_labels_use_only_observed_target_classes(
    public_profile: bool, target_kind: str
) -> None:
    """Shared category dictionaries must not substitute unrelated or unused target labels."""
    held = pl.Series(["unrelated_1", "unrelated_2"]).cast(pl.Categorical)
    target = pl.Series("target", ["yes"] * 30 + ["no"] * 30)
    expected_labels = ["yes", "no"]
    if target_kind == "numeric":
        target = pl.Series("target", [20] * 30 + [10] * 30)
        expected_labels = ["20", "10"]
    elif target_kind == "categorical_subset":
        target = pl.Series("target", ["unused_target"] + target.to_list()).cast(pl.Categorical)
        target = target.slice(1)
    elif target_kind == "missing":
        target = pl.Series("target", [None] * 30 + ["no"] * 30)
        expected_labels = ["Missing", "no"]
    df = pl.DataFrame({"x": np.arange(60, dtype=float), "target": target})
    analyzer = EDAAnalyzer(df)

    tree = (
        analyzer.analyze(target_col="target", task_type="classification").rule_tree
        if public_profile
        else analyzer._discover_rules(["x"], "target", "classification")
    )

    assert tree is not None
    assert tree.accuracy == 1.0
    leaves = [node for node in tree.nodes if node.is_leaf]
    assert [node.class_name for node in leaves] == expected_labels
    assert tree.rules is not None
    assert [rule.split(" THEN ")[1].split(" (")[0] for rule in tree.rules] == expected_labels
    assert {node.class_name for node in tree.nodes}.isdisjoint(held.to_list())


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_rule_text_reports_actual_leaf_sample_counts(task_type: str) -> None:
    """Rule support must count rows while confidence remains the winning class proportion."""
    target = (
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 1] if task_type == "classification" else [10.0] * 4 + [20.0] * 6
    )
    df = pl.DataFrame({"x": [0.0] * 4 + [1.0] * 6, "target": target})

    tree = EDAAnalyzer(df).analyze(target_col="target", task_type=task_type).rule_tree

    assert tree is not None
    leaves = [node for node in tree.nodes if node.is_leaf]
    assert [node.samples for node in leaves] == [4, 6]
    assert tree.rules is not None
    assert [int(rule.split("Samples: ")[1].rstrip(")")) for rule in tree.rules] == [4, 6]
    if task_type == "classification":
        assert [rule.split("Confidence: ")[1].split("%")[0] for rule in tree.rules] == [
            "75.0",
            "66.7",
        ]
    else:
        assert [node.class_name for node in leaves] == ["10.00", "20.00"]


@pytest.mark.parametrize("target_col", ["target", "count"])
def test_rule_class_cap_keeps_target_labels_with_reserved_column_name(target_col: str) -> None:
    """A target named count must retain its top ten labels and the combined Other class."""
    labels = [f"class_{i}" for i in range(15) for _ in range(40 - i)]
    feature = [float(i) for i in range(15) for _ in range(40 - i)]
    df = pl.DataFrame({"x": feature, target_col: labels})

    tree = EDAAnalyzer(df)._discover_rules(["x"], target_col, "classification")

    assert tree is not None
    assert tree.rules is not None
    predicted = {rule.split(" THEN ")[1].split(" (")[0] for rule in tree.rules}
    assert predicted <= {f"class_{i}" for i in range(10)} | {"Other"}
    assert "Other" in predicted


@pytest.mark.parametrize(
    "feature_kind, public_profile",
    [
        ("string", False),
        ("categorical_subset", False),
        ("missing", False),
        ("numeric", False),
        ("numeric", True),
    ],
)
def test_rule_conditions_use_only_observed_feature_categories(
    feature_kind: str, public_profile: bool
) -> None:
    """Rule conditions must match observed feature values without unrelated dictionary entries."""
    held = pl.Series(["unrelated_1", "unrelated_2"]).cast(pl.Categorical)
    feature = pl.Series("feature", ["red"] * 30 + ["blue"] * 30)
    expected_categories = ["red", "blue"]
    if feature_kind == "categorical_subset":
        feature = pl.Series("feature", ["unused_feature"] + feature.to_list()).cast(pl.Categorical)
        feature = feature.slice(1)
    elif feature_kind == "missing":
        feature = pl.Series("feature", [None] * 30 + ["blue"] * 30)
        expected_categories = ["Missing", "blue"]
    elif feature_kind == "numeric":
        feature = pl.Series("feature", [10] * 30 + [20] * 30)
        expected_categories = ["10", "20"]
    df = pl.DataFrame({"feature": feature, "target": ["yes"] * 29 + ["no"] * 30 + ["yes"]})
    analyzer = EDAAnalyzer(df)

    tree = (
        analyzer.analyze(target_col="target", task_type="classification").rule_tree
        if public_profile
        else analyzer._discover_rules(["feature"], "target", "classification")
    )

    assert tree is not None
    assert tree.categories is not None
    assert set(tree.categories["feature"]) == set(expected_categories)
    assert tree.rules is not None
    conditions_and_labels = {
        (rule.split(" THEN ")[0], rule.split(" THEN ")[1].split(" (")[0]) for rule in tree.rules
    }
    assert conditions_and_labels == {
        (f"IF feature in ['{expected_categories[0]}']", "yes"),
        (f"IF feature in ['{expected_categories[1]}']", "no"),
    }
    assert not any(label in rule for label in held.to_list() for rule in tree.rules)


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


def test_discover_rules_categorical_split_uses_category_names_not_raw_codes() -> None:
    """When the surrogate tree actually splits on a categorical feature, the
    human-readable rule text must show category names (e.g. "color in
    ['red', 'blue']"), not the meaningless raw ordinal-encoding threshold
    (e.g. "color <= 2.00") — regression guard for the misleading-threshold fix.
    """
    rng = np.random.default_rng(25)
    n = 90
    # Target is driven entirely by the categorical feature, so the tree must
    # split on it to get any accuracy.
    color = rng.choice(["red", "green", "blue"], size=n)
    target = np.where(color == "red", "positive", "negative")
    noise = rng.normal(0, 1, n)
    df = pl.DataFrame({"color": color, "noise": noise, "target": target})
    analyzer = EDAAnalyzer(df)
    tree = analyzer._discover_rules(["color", "noise"], "target")

    assert tree is not None
    assert tree.categories is not None
    assert "color" in tree.categories
    assert set(tree.categories["color"]) == {"red", "green", "blue"}

    # At least one rule must reference the categorical feature using the
    # "in [...]" form rather than a raw numeric threshold.
    color_rules = [r for r in (tree.rules or []) if "color" in r]
    assert color_rules
    assert any("color in [" in r for r in color_rules)
    assert not any("color <=" in r or "color >" in r for r in color_rules)


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
