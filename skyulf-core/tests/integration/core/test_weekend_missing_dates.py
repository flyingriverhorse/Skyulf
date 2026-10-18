"""Weekend flags must preserve missing dates and their saved artifact's dtype contract."""

import pickle

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
@pytest.mark.parametrize("legacy", [False, True])
def test_weekend_preserves_missing_dates_and_observed_values(
    engine, parsed, case, legacy, monkeypatch
):
    """Missing dates stay unknown; new fits use nullable integers and old states keep their dtype."""
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
    if legacy:
        params = {
            "type": "feature_generation",
            "operations": config["operations"],
            "epsilon": 1e-9,
            "allow_overwrite": False,
        }
        assert "operation_contract_version" not in params
    else:
        params = dict(FeatureGenerationCalculator().fit(frame, config))
        assert params["operation_contract_version"] == 1
    saved = pickle.dumps(params)
    params = pickle.loads(saved)

    output = FeatureGenerationApplier().apply(frame, params)

    actual = [
        None if pd.isna(value) else int(value) for value in output["date_is_weekend"].to_list()
    ]
    assert actual == expected
    pd.testing.assert_frame_equal(source, original)
    if engine == "pandas":
        pd.testing.assert_index_equal(output.index, source.index)
        expected_dtype = (
            str(pd.Series(dtype=int).dtype) if legacy and None not in expected else "Int64"
        )
        assert str(output["date_is_weekend"].dtype) == expected_dtype
    else:
        assert output["date_is_weekend"].dtype == pl.Int64
    assert pickle.dumps(params) == saved
    assert list(output.columns) == ["date", "date_is_weekend"]
