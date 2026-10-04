"""Engineered weekend flags must preserve missing dates and existing valid-date dtypes."""

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("parsed", [False, True])
@pytest.mark.parametrize("case", ["valid", "mixed", "all-missing", "empty"])
def test_weekend_preserves_missing_dates_and_observed_values(engine, parsed, case, monkeypatch):
    """Missing and invalid dates stay unknown while valid weekend flags retain integer values."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    cases = {
        "valid": (["2026-10-03", "2026-10-04", "2026-10-05"], [1, 1, 0]),
        "mixed": (["2026-10-03", "2026-10-05", None, "invalid"], [1, 0, None, None]),
        "all-missing": ([None, None], [None, None]),
        "empty": ([], []),
    }
    dates, expected = cases[case]
    source = pd.DataFrame({"date": pd.Series(dates, dtype="string")})
    if parsed:
        source["date"] = pd.to_datetime(source["date"], errors="coerce", utc=True)
    source.index = pd.Index(range(20, 20 + len(source)), name="row")
    original = source.copy(deep=True)
    frame = source if engine == "pandas" else pl.from_pandas(source)
    config = {
        "operations": [
            {
                "operation_type": "datetime_extract",
                "input_columns": ["date"],
                "datetime_features": ["is_weekend"],
            }
        ]
    }
    params = FeatureGenerationCalculator().fit(frame, config)

    output = FeatureGenerationApplier().apply(frame, params)

    actual = [
        None if pd.isna(value) else int(value) for value in output["date_is_weekend"].to_list()
    ]
    assert actual == expected
    pd.testing.assert_frame_equal(source, original)
    if engine == "pandas":
        pd.testing.assert_index_equal(output.index, source.index)
        expected_dtype = "Int64" if None in expected else str(pd.Series(dtype=int).dtype)
        assert str(output["date_is_weekend"].dtype) == expected_dtype
    assert list(output.columns) == ["date", "date_is_weekend"]
