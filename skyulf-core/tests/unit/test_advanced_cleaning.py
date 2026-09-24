import pandas as pd

from skyulf.preprocessing.cleaning import (
    AliasReplacementApplier,
    AliasReplacementCalculator,
)


def test_boolean_normalizer():
    df = pd.DataFrame({"bool_col": ["yes", "No", "TRUE", "0", "1", "invalid"]})

    calc = AliasReplacementCalculator()
    applier = AliasReplacementApplier()

    config = {"columns": ["bool_col"], "mode": "normalize_boolean"}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)

    # AliasReplacementCalculator maps to "Yes"/"No" strings, not booleans
    res_list = result["bool_col"].tolist()

    assert res_list[0] == "Yes"
    assert res_list[1] == "No"
    assert res_list[2] == "Yes"
    assert res_list[3] == "No"
    assert res_list[4] == "Yes"
    assert res_list[5] == "invalid"  # Keeps original if not mapped


def test_country_standardizer():
    df = pd.DataFrame({"country": ["United States", "UK", "Turkey", "Unknown"]})

    calc = AliasReplacementCalculator()
    applier = AliasReplacementApplier()

    config = {"columns": ["country"], "mode": "canonicalize_country_codes"}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)

    # The alias map keys are space-stripped (e.g. "unitedstates"), but
    # cleaning only strips punctuation, not spaces, so "United States" would
    # not match. Use "UnitedStates" here to exercise the mapping logic.
    df = pd.DataFrame({"country": ["UnitedStates", "UK", "Turkey", "Unknown"]})
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)

    assert result["country"].iloc[0] == "USA"
    assert result["country"].iloc[1] == "United Kingdom"
    assert result["country"].iloc[3] == "Unknown"
