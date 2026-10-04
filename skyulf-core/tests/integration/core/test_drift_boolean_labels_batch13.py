"""Boolean serialization must not manufacture categorical distribution changes."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.profiling.drift import DriftCalculator


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [pl.String, pl.Categorical, pl.Enum(["True", "FALSE"])])
def test_boolean_text_casing_preserves_distribution(reverse, dtype):
    """Native flags and their textual serialization describe the same observations."""
    flags = pl.DataFrame({"flag": [True, False, None] * 20})
    text = pl.DataFrame({"flag": pl.Series(["True", "FALSE", None] * 20, dtype=dtype)})
    before_flags, before_text = flags.clone(), text.clone()
    reference, current = (text, flags) if reverse else (flags, text)

    report = DriftCalculator(reference, current).calculate_drift()

    column = report.column_drifts["flag"]
    assert [(metric.metric, metric.value) for metric in column.metrics] == [
        ("psi_categorical", 0.0)
    ]
    assert report.drifted_columns_count == 0
    assert_frame_equal(flags, before_flags)
    assert_frame_equal(text, before_text)


@pytest.mark.parametrize("reverse", [False, True])
def test_boolean_text_distribution_change_remains_visible(reverse):
    """Normalizing the spelling must retain a real change in flag frequencies."""
    flags = pl.DataFrame({"flag": [True] * 50 + [False] * 50})
    text = pl.DataFrame({"flag": ["TRUE"] * 90 + ["false"] * 10})
    reference, current = (text, flags) if reverse else (flags, text)

    report = DriftCalculator(reference, current).calculate_drift()

    column = report.column_drifts["flag"]
    assert column.metrics[0].metric == "psi_categorical"
    assert column.metrics[0].value == pytest.approx(0.8788898309344878)
    assert report.drifted_columns_count == 1


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("labels", [["True", "unknown"], ["1", "0"], ["yes", "no"]])
def test_non_boolean_text_is_not_silently_coerced(reverse, labels):
    """Unrecognized labels must yield type drift even under permissive PSI thresholds."""
    flags = pl.DataFrame({"flag": [True, False] * 20})
    text = pl.DataFrame({"flag": labels * 20})
    reference, current = (text, flags) if reverse else (flags, text)

    report = DriftCalculator(reference, current).calculate_drift(thresholds={"psi": 100})

    assert report.column_drifts["flag"].metrics[0].metric == "type_drift"
    assert report.drifted_columns_count == 1


def test_text_only_categories_remain_case_sensitive():
    """String categories must not be merged solely because they resemble flag names."""
    reference = pl.DataFrame({"flag": ["True", "False"] * 20})
    current = pl.DataFrame({"flag": ["true", "false"] * 20})

    report = DriftCalculator(reference, current).calculate_drift()

    assert report.column_drifts["flag"].metrics[0].metric == "psi_categorical"
    assert report.drifted_columns_count == 1
