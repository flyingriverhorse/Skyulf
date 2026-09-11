"""Profile dtype regressions through the backend analysis and serialization path."""

import polars as pl

from backend.eda.tasks import _run_eda_analyzer


def test_eda_report_serializes_null_and_enum_columns() -> None:
    """Valid source dtypes must produce a usable report for API consumers."""
    df = pl.DataFrame(
        {
            "empty": [None, None, None],
            "label": pl.Series(["a", "b", "a"], dtype=pl.Enum(["a", "b"])),
        }
    )

    result = _run_eda_analyzer(df, None).model_dump(mode="json")

    assert result["row_count"] == 3
    assert result["columns"]["empty"]["dtype"] == "Unknown"
    assert result["columns"]["empty"]["missing_percentage"] == 100.0
    assert result["columns"]["empty"]["text_stats"] is None
    assert result["columns"]["label"]["dtype"] == "Categorical"
    assert result["columns"]["label"]["categorical_stats"]["top_k"] == [
        {"value": "a", "count": 2},
        {"value": "b", "count": 1},
    ]
    assert result["sample_data"] == df.to_dicts()
