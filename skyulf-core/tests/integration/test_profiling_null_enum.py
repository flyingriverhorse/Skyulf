"""Regression coverage for null-only and explicitly categorical profile columns."""

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize(
    ("series", "expected_type"),
    [
        pytest.param(pl.Series("value", [None] * 4), "Unknown", id="untyped-null"),
        pytest.param(
            pl.Series("value", [None] * 4, dtype=pl.Float64), "Numeric", id="typed-null-float"
        ),
        pytest.param(
            pl.Series("value", [None] * 4, dtype=pl.String), "Text", id="typed-null-string"
        ),
        pytest.param(
            pl.Series("value", ["a", "b", "a", None], dtype=pl.Categorical),
            "Categorical",
            id="categorical",
        ),
        pytest.param(
            pl.Series("value", ["a", "b", "a", None], dtype=pl.Enum(["a", "b"])),
            "Categorical",
            id="enum",
        ),
        pytest.param(
            pl.Series("value", [None] * 4, dtype=pl.Enum(["a", "b"])),
            "Categorical",
            id="typed-null-enum",
        ),
    ],
)
def test_analyze_preserves_null_and_enum_columns(series: pl.Series, expected_type: str) -> None:
    """Valid null and Enum dtypes must not abort the entire dataset profile."""
    df = pl.DataFrame({"value": series, "signal": [1.0, 2.0, 4.0, 8.0]})
    analyzer = EDAAnalyzer(df)

    profile = analyzer.analyze()

    column = profile.columns["value"]
    assert column.dtype == expected_type
    assert analyzer._get_semantic_type(series) == expected_type
    assert column.missing_count == series.null_count()
    assert column.missing_percentage == series.null_count() / len(series) * 100
    assert profile.row_count == 4
    assert profile.sample_data == df.to_dicts()
    assert profile.columns["signal"].numeric_stats is not None
    assert profile.columns["signal"].numeric_stats.mean == 3.75
    if expected_type == "Unknown":
        assert column.numeric_stats is None
        assert column.categorical_stats is None
        assert column.text_stats is None
        assert column.histogram is None
        assert any(
            alert.column == "value" and alert.type == "High Null" for alert in profile.alerts
        )
    elif expected_type == "Categorical":
        assert column.categorical_stats is not None
        counts = {item["value"]: item["count"] for item in column.categorical_stats.top_k}
        assert counts == ({} if series.null_count() == len(series) else {"a": 2, "b": 1})
        assert column.text_stats is None
    else:
        assert (
            getattr(column, "numeric_stats" if expected_type == "Numeric" else "text_stats")
            is not None
        )
    assert analyzer.df.equals(df)


def test_analyze_null_only_frame_preserves_missing_alerts() -> None:
    """A wholly empty-valued dataset still needs a report and missing-data advice."""
    profile = EDAAnalyzer(pl.DataFrame({"empty": [None, None]})).analyze()

    assert profile.columns["empty"].dtype == "Unknown"
    assert profile.missing_cells_percentage == 100.0
    assert profile.columns["empty"].missing_count == 2
    assert any(alert.type == "High Null" for alert in profile.alerts)
    assert profile.sample_data == [{"empty": None}, {"empty": None}]


def test_analyze_enum_target_infers_classification() -> None:
    """An explicitly categorical target must reach classification rule discovery."""
    df = pl.DataFrame(
        {
            "signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "label": pl.Series(["a", "a", "a", "b", "b", "b"], dtype=pl.Enum(["a", "b"])),
        }
    )

    profile = EDAAnalyzer(df).analyze(target_col="label")

    assert profile.columns["label"].dtype == "Categorical"
    assert profile.task_type == "Classification"
    assert profile.rule_tree is not None
