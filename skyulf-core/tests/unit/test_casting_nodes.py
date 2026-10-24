import pandas as pd

from skyulf.preprocessing.casting import CastingApplier, CastingCalculator


def test_casting():
    df = pd.DataFrame(
        {
            "A": ["1", "2", "3"],
            "B": [1, 2, 3],
            "C": ["2021-01-01", "2021-01-02", "not_a_date"],
        }
    )

    # 1. Fit
    calc = CastingCalculator()
    config = {"column_types": {"A": "int", "B": "float", "C": "datetime"}}
    params = calc.fit(df, config)

    assert params["type_map"]["A"] == "int64"
    assert params["type_map"]["B"] == "float64"

    # 2. Apply
    applier = CastingApplier()
    transformed = applier.apply(df, params)

    assert pd.api.types.is_integer_dtype(transformed["A"])
    assert pd.api.types.is_float_dtype(transformed["B"])
    assert pd.api.types.is_datetime64_any_dtype(transformed["C"])

    # Check values
    assert transformed["A"].iloc[0] == 1
    assert transformed["B"].iloc[0] == 1.0
    assert pd.isna(transformed["C"].iloc[2])  # Coerced error


import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_integral_float_codes_cast_to_string_without_decimal(engine):
    """A nullable INT code arrives as float; "81.0" would miss fixed one-hot categories."""
    frame = pd.DataFrame({"code": [81.0, np.nan, 7.5, 12.0]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    state = CastingCalculator().fit(native, {"column_types": {"code": "string"}})
    assert state["integral_floats_as_int"] is True
    result = CastingApplier().apply(native, state)
    values = result["code"].to_list() if engine == "polars" else result["code"].tolist()
    assert values[0] == "81" and values[2] == "7.5" and values[3] == "12"
    assert values[1] is None or pd.isna(values[1])


def test_saved_casts_without_the_flag_keep_their_old_text():
    """Models trained before the change must replay exactly the strings they learned on."""
    frame = pd.DataFrame({"code": [81.0, 12.0]})
    state = dict(CastingCalculator().fit(frame, {"column_types": {"code": "string"}}))
    del state["integral_floats_as_int"]
    CastingApplier.validate_inference_state(state)
    assert CastingApplier().apply(frame, state)["code"].tolist() == ["81.0", "12.0"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_integral_text_is_identical_when_scored_one_row_at_a_time(engine):
    """Serving probes score rows in chunks of 1; each row must match the whole-frame result."""
    frame = pd.DataFrame({"code": [81.0, np.nan, 12.0, 7.5]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    state = CastingCalculator().fit(native, {"column_types": {"code": "string"}})
    whole = CastingApplier().apply(native, state)["code"].to_list()
    for index in range(len(frame)):
        row = native[index : index + 1]
        assert CastingApplier().apply(row, state)["code"].to_list() == [whole[index]]
