"""Tests for skyulf.profiling._analyzer.recommendations.RecommendationsMixin.

Unit-tests ``_generate_recommendations`` directly against hand-built
``ColumnProfile`` objects (per the schema in skyulf.profiling.schemas),
covering the skewness "Transform", high-cardinality "Encode", and
"is_constant" -> "Drop" branches that aren't exercised elsewhere.
"""

import polars as pl
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import CategoricalStats, ColumnProfile, NumericStats


def _analyzer() -> EDAAnalyzer:
    """A trivial analyzer instance; _generate_recommendations only needs `self` for the method."""
    return EDAAnalyzer(pl.DataFrame({"x": [1, 2, 3]}))


def test_generate_recommendations_high_skewness_triggers_transform() -> None:
    """A numeric column with |skewness| > 1.5 should get a 'Transform' recommendation (line 47)."""
    profile = ColumnProfile(
        name="skewed",
        dtype="Numeric",
        missing_count=0,
        missing_percentage=0.0,
        numeric_stats=NumericStats(skewness=2.3),
    )
    analyzer = _analyzer()
    recs = analyzer._generate_recommendations({"skewed": profile}, [], None)

    transform_recs = [r for r in recs if r.action == "Transform"]
    assert len(transform_recs) == 1
    assert transform_recs[0].column == "skewed"
    assert "skewness" in transform_recs[0].reason.lower()


def test_generate_recommendations_high_cardinality_categorical_triggers_encode() -> None:
    """A Categorical column with unique_count > 50 should get an 'Encode' recommendation (line 62)."""
    profile = ColumnProfile(
        name="high_card",
        dtype="Categorical",
        missing_count=0,
        missing_percentage=0.0,
        categorical_stats=CategoricalStats(unique_count=75, top_k=[{"value": "a", "count": 1}]),
    )
    analyzer = _analyzer()
    recs = analyzer._generate_recommendations({"high_card": profile}, [], None)

    encode_recs = [r for r in recs if r.action == "Encode"]
    assert len(encode_recs) == 1
    assert encode_recs[0].column == "high_card"
    assert "cardinality" in encode_recs[0].reason.lower()


def test_generate_recommendations_constant_column_triggers_drop() -> None:
    """is_constant=True should produce a 'Drop' recommendation with 'Constant value' reason (line 88)."""
    profile = ColumnProfile(
        name="const_col",
        dtype="Numeric",
        missing_count=0,
        missing_percentage=0.0,
        is_constant=True,
    )
    analyzer = _analyzer()
    recs = analyzer._generate_recommendations({"const_col": profile}, [], None)

    drop_recs = [r for r in recs if r.action == "Drop" and r.column == "const_col"]
    assert len(drop_recs) == 1
    assert drop_recs[0].reason == "Constant value"


def test_generate_recommendations_is_unique_triggers_drop() -> None:
    """is_unique=True with an ID-like dtype should produce a 'Drop' / 'Likely ID column' rec (line 88)."""
    profile = ColumnProfile(
        name="row_id",
        dtype="Categorical",
        missing_count=0,
        missing_percentage=0.0,
        is_unique=True,
    )
    analyzer = _analyzer()
    recs = analyzer._generate_recommendations({"row_id": profile}, [], None)

    drop_recs = [r for r in recs if r.action == "Drop" and r.column == "row_id"]
    assert len(drop_recs) == 1
    assert drop_recs[0].reason == "Likely ID column"


def test_generate_recommendations_no_critical_issues_returns_keep() -> None:
    """A clean profile (no missing/constant/skew/high-cardinality) yields a 'Keep' recommendation."""
    profile = ColumnProfile(
        name="clean",
        dtype="Numeric",
        missing_count=0,
        missing_percentage=0.0,
        numeric_stats=NumericStats(skewness=0.1),
    )
    analyzer = _analyzer()
    recs = analyzer._generate_recommendations({"clean": profile}, [], None)

    assert any(r.action == "Keep" for r in recs)


def test_generate_recommendations_combines_all_branches_for_one_column() -> None:
    """A single pathological column can trigger Transform/Encode/Drop simultaneously."""
    profile = ColumnProfile(
        name="messy",
        dtype="Categorical",
        missing_count=0,
        missing_percentage=0.0,
        numeric_stats=NumericStats(skewness=-2.0),
        categorical_stats=CategoricalStats(unique_count=60, top_k=[{"value": "a", "count": 1}]),
        is_constant=True,
    )
    analyzer = _analyzer()
    recs = analyzer._generate_recommendations({"messy": profile}, [], None)

    actions = {r.action for r in recs}
    assert "Transform" in actions
    assert "Encode" in actions
    assert "Drop" in actions


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    run through the full :meth:`EDAAnalyzer.analyze` pipeline so
    ``_generate_recommendations`` sees real column profiles instead of
    hand-built ``ColumnProfile`` objects.
    """

    def test_analyze_flags_missing_value_columns_for_imputation(self) -> None:
        df = load_sample_dataset("customers", engine="polars")
        analyzer = EDAAnalyzer(df)
        profile = analyzer.analyze(target_col="churned")

        recs_by_column = {r.column: r for r in profile.recommendations}

        # age/income/city/lat/lon each have some missing values (but not more
        # than 50%) -> every one should be flagged for imputation, not dropped.
        for col in ("age", "income", "city", "lat", "lon"):
            assert recs_by_column[col].action == "Impute"
            assert "Missing values" in recs_by_column[col].reason

        # customer_id is a small (15-row) unique sequence: below the
        # unique_count > 50 threshold used to flag "likely ID" columns, so it
        # is not flagged here even though it is a perfect identifier.
        assert "customer_id" not in recs_by_column
