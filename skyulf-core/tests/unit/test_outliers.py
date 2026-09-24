import pandas as pd
import pytest

from skyulf.preprocessing.outliers import (
    IQRApplier,
    IQRCalculator,
    ZScoreApplier,
    ZScoreCalculator,
)


@pytest.fixture
def outlier_df():
    return pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 100], "B": [10, 10, 10, 10, 10, 10]}  # 100 is outlier
    )


def test_iqr_clip(outlier_df):
    # Despite the name, IQRApplier only supports dropping rows outside bounds
    # (no clip/mask mode), so this test verifies the drop behavior.

    calc = IQRCalculator()
    params = calc.fit(outlier_df, {"multiplier": 1.5, "columns": ["A"]})

    applier = IQRApplier()
    res = applier.apply(outlier_df, params)

    # Should drop the outlier
    assert len(res) == 5
    assert 100 not in res["A"].values


def test_zscore_drop(outlier_df):
    # Similarly, ZScore likely drops.
    calc = ZScoreCalculator()
    params = calc.fit(outlier_df, {"threshold": 2.0, "columns": ["A"]})

    applier = ZScoreApplier()
    res = applier.apply(outlier_df, params)

    assert len(res) == 5
    assert 100 not in res["A"].values
