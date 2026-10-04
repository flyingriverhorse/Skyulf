"""Observed string categories must reach accurate target statistics (OC-110/247)."""

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize(
    ("row_count", "class_count"), [(50, 3), (100, 5), (100, 6), (100, 10), (40, 20)]
)
def test_small_string_target_reaches_classification(row_count: int, class_count: int) -> None:
    """Repeated class labels must retain target analysis above the 5% cardinality ratio."""
    frame = pl.DataFrame(
        {
            "signal": [10.0 * (i % class_count) + 0.01 * i for i in range(row_count)],
            "label": [f"class_{i % class_count}" for i in range(row_count)],
        }
    )
    analyzer = EDAAnalyzer(frame)

    profile = analyzer.analyze(target_col="label")

    column = profile.columns["label"]
    assert column.dtype == "Categorical"
    assert analyzer._get_semantic_type(analyzer.df["label"]) == "Categorical"
    assert column.categorical_stats is not None
    assert column.categorical_stats.unique_count == class_count
    assert column.text_stats is None
    assert profile.task_type == "Classification"
    assert profile.rule_tree is not None
    assert profile.target_correlations is not None
    assert 0.99 < profile.target_correlations["signal"] <= 1.0
    assert any(
        r.column == "label" and r.reason == "Balanced Target" for r in profile.recommendations
    )


@pytest.mark.parametrize(
    ("values", "expected_type"),
    [
        pytest.param(["a", "b", "a", "b"], "Categorical", id="small-repeated-labels"),
        pytest.param(["a", "a", None], "Categorical", id="repeated-label-with-null"),
        pytest.param(
            [f"class_{i}" for i in range(20)] * 2 + [None],
            "Categorical",
            id="twenty-labels-plus-null",
        ),
        pytest.param(["a", "b", "c"], "Text", id="unique-small-strings"),
        pytest.param(["a", "b", None, None], "Text", id="nulls-are-not-repeated-labels"),
        pytest.param(["a"] + [None] * 99, "Text", id="nulls-cannot-dilute-cardinality"),
        pytest.param([None] * 4, "Text", id="all-null-small"),
        pytest.param([None] * 100, "Text", id="all-null-large"),
        pytest.param(["one sentence"], "Text", id="single-observation"),
        pytest.param(["constant"] * 4, "Categorical", id="constant-label"),
        pytest.param(
            [f"class_{i}" for i in range(21)] * 2, "Text", id="above-small-vocabulary-limit"
        ),
        pytest.param(
            [f"class_{i}" for i in range(60)] * 25,
            "Categorical",
            id="large-vocabulary-low-ratio",
        ),
        pytest.param([f"sentence number {i}" for i in range(30)], "Text", id="unique-text"),
    ],
)
def test_string_inference_uses_observed_categories(
    values: list[str | None], expected_type: str
) -> None:
    """Both inference paths must distinguish repeated labels from missing or unique text."""
    series = pl.Series("value", values, dtype=pl.String)
    analyzer = EDAAnalyzer(series.to_frame())

    profile = analyzer.analyze()

    column = profile.columns["value"]
    assert analyzer._get_semantic_type(series) == expected_type
    assert column.dtype == expected_type
    assert column.missing_count == series.null_count()
    if expected_type == "Categorical":
        assert column.categorical_stats is not None
        assert column.text_stats is None
    else:
        assert column.text_stats is not None
        assert column.categorical_stats is None


@pytest.mark.parametrize("dtype", [pl.String, pl.Categorical, pl.Enum(["a"])])
@pytest.mark.parametrize(("observed_count", "rare_count"), [(2, 1), (99, 0)])
def test_categorical_counts_exclude_missing_values(
    dtype: pl.DataType | type[pl.DataType], observed_count: int, rare_count: int
) -> None:
    """A missing value must not inflate the number of categories or rare labels."""
    series = pl.Series("value", ["a"] * observed_count + [None], dtype=dtype)

    column = EDAAnalyzer(series.to_frame()).analyze().columns["value"]

    stats = column.categorical_stats
    assert stats is not None
    assert stats.unique_count == 1
    assert stats.rare_labels_count == rare_count
    assert stats.top_k == [{"value": "a", "count": observed_count}]
    assert column.missing_count == 1


def test_missing_values_do_not_displace_top_categories() -> None:
    """Missing rows must not consume a frequency slot reserved for an observed label."""
    labels = [f"class_{i}" for i in range(10)]
    frame = pl.DataFrame({"value": labels * 2 + [None] * 100})

    column = EDAAnalyzer(frame).analyze().columns["value"]

    stats = column.categorical_stats
    assert stats is not None
    assert len(stats.top_k) == 10
    assert {entry["value"]: entry["count"] for entry in stats.top_k} == dict.fromkeys(labels, 2)
    assert stats.unique_count == 10
    assert stats.rare_labels_count == 10
    assert column.missing_count == 100


def test_all_null_native_category_has_no_observed_labels() -> None:
    """An explicit categorical dtype can retain its type without inventing a null label."""
    frame = pl.DataFrame({"value": pl.Series([None] * 3, dtype=pl.Categorical)})

    column = EDAAnalyzer(frame).analyze().columns["value"]

    assert column.categorical_stats is not None
    assert column.categorical_stats.unique_count == 0
    assert column.categorical_stats.rare_labels_count == 0
    assert column.categorical_stats.top_k == []
    assert column.missing_count == 3
