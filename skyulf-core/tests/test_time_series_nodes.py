"""Tests for the time-series preprocessing nodes (Lag, Rolling, DateFeatures).

Each node is exercised on both engines (pandas + polars) to confirm parity,
plus the ``infer_output_schema`` predictions and registry wiring.
"""

import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.core.schema import SkyulfSchema
from skyulf.preprocessing import (
    DateFeaturesApplier,
    DateFeaturesCalculator,
    LagFeaturesApplier,
    LagFeaturesCalculator,
    RollingAggregateApplier,
    RollingAggregateCalculator,
)
from skyulf.registry import NodeRegistry

_lag_coerces_lags_cases = TestCaseLoader(
    "preprocessing/time_series_nodes", group="lag_calculator_fit_coerces_lags"
).load()
_rolling_filters_aggregations_cases = TestCaseLoader(
    "preprocessing/time_series_nodes", group="rolling_calculator_fit_filters_aggregations"
).load()


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b"],
            "v": [1.0, 2.0, 3.0, 10.0, 20.0],
            "d": pd.to_datetime(
                ["2021-01-01", "2021-02-15", "2021-03-31", "2021-06-30", "2021-12-25"]
            ),
        }
    )


def test_nodes_registered():
    for name in ("LagFeatures", "RollingAggregate", "DateFeatures"):
        assert NodeRegistry.get_calculator(name)
        assert NodeRegistry.get_applier(name)


def test_lag_features_parity_with_groups():
    df = _frame()
    art = LagFeaturesCalculator().fit(
        df, {"columns": ["v"], "lags": [1], "group_by": ["g"], "sort_by": None}
    )
    pandas_out = LagFeaturesApplier().apply(df, art)
    polars_out = LagFeaturesApplier().apply(pl.from_pandas(df), art)
    # Lag is computed within each group, so the first row of every group is null.
    assert pandas_out["v_lag_1"].fillna(-1).tolist() == [-1.0, 1.0, 2.0, -1.0, 10.0]
    pl_vals = [v if v is not None else -1.0 for v in polars_out["v_lag_1"].to_list()]
    assert pl_vals == pandas_out["v_lag_1"].fillna(-1).tolist()


@pytest.mark.parametrize(*_lag_coerces_lags_cases)
def test_lag_features_coerces_and_dedups_lags(lags_input, expected_lags):
    """``LagFeaturesCalculator.fit`` coerces the ``lags`` config via ``coerce_lags``.

    Loaded from ``tests/test_cases/preprocessing/time_series_nodes.json``
    (group ``lag_calculator_fit_coerces_lags``).
    """
    art = LagFeaturesCalculator().fit(_frame(), {"columns": ["v"], "lags": lags_input})
    assert art["lags"] == expected_lags


def test_rolling_aggregate_parity():
    df = _frame()
    art = RollingAggregateCalculator().fit(
        df, {"columns": ["v"], "window": 2, "aggregations": ["mean", "sum"], "min_periods": 1}
    )
    pandas_out = RollingAggregateApplier().apply(df, art)
    polars_out = RollingAggregateApplier().apply(pl.from_pandas(df), art)
    assert pandas_out["v_roll_mean_2"].tolist() == [1.0, 1.5, 2.5, 6.5, 15.0]
    assert polars_out["v_roll_mean_2"].to_list() == pandas_out["v_roll_mean_2"].tolist()
    assert "v_roll_sum_2" in pandas_out.columns


@pytest.mark.parametrize(*_rolling_filters_aggregations_cases)
def test_rolling_aggregate_filters_unknown_aggs(aggregations_input, expected_aggregations):
    """``RollingAggregateCalculator.fit`` filters unrecognised aggregation names.

    Loaded from ``tests/test_cases/preprocessing/time_series_nodes.json``
    (group ``rolling_calculator_fit_filters_aggregations``).
    """
    art = RollingAggregateCalculator().fit(
        _frame(), {"columns": ["v"], "aggregations": aggregations_input}
    )
    assert art["aggregations"] == expected_aggregations


def test_date_features_parity():
    df = _frame()
    feats = ["year", "month", "dayofweek", "is_weekend", "is_month_end"]
    art = DateFeaturesCalculator().fit(df, {"columns": ["d"], "features": feats})
    pandas_out = DateFeaturesApplier().apply(df, art)
    polars_out = DateFeaturesApplier().apply(pl.from_pandas(df), art)
    assert pandas_out["d_year"].tolist() == [2021] * 5
    # 2021-03-31 and 2021-06-30 are month-ends.
    assert pandas_out["d_is_month_end"].tolist() == [0, 0, 1, 1, 0]
    assert polars_out["d_is_month_end"].to_list() == pandas_out["d_is_month_end"].tolist()
    assert polars_out["d_dayofweek"].to_list() == pandas_out["d_dayofweek"].tolist()


def test_date_features_drop_original():
    art = DateFeaturesCalculator().fit(
        _frame(), {"columns": ["d"], "features": ["year"], "drop_original": True}
    )
    out = DateFeaturesApplier().apply(_frame(), art)
    assert "d" not in out.columns
    assert "d_year" in out.columns


def test_lag_infer_output_schema_extends_columns():
    schema = SkyulfSchema.from_columns(["v"], {"v": "float64"})
    out = LagFeaturesCalculator().infer_output_schema(schema, {"columns": ["v"], "lags": [1, 2]})
    assert out is not None
    assert "v_lag_1" in out and "v_lag_2" in out
    assert out.dtypes["v_lag_1"] == "float64"


def test_date_features_infer_output_schema_drops_original():
    schema = SkyulfSchema.from_columns(["d"], {"d": "datetime"})
    out = DateFeaturesCalculator().infer_output_schema(
        schema, {"columns": ["d"], "features": ["year"], "drop_original": True}
    )
    assert out is not None
    assert "d" not in out
    assert out.dtypes["d_year"] == "int64"


# ---------------------------------------------------------------------------
# Real-shaped dataset integration check
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.

    Verifies that DateFeaturesCalculator extracts calendar features from the
    real-world ``signup_date`` string column and that the pandas and polars
    apply paths return identical results for all 15 rows.
    """

    def test_date_features_on_signup_date_pandas_polars_parity(self) -> None:
        df = load_sample_dataset("customers")
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        art = DateFeaturesCalculator().fit(
            df, {"columns": ["signup_date"], "features": ["year", "month", "dayofweek"]}
        )
        pd_out = DateFeaturesApplier().apply(df, art)
        pl_out = DateFeaturesApplier().apply(pl.from_pandas(df), art)
        # All signup years must fall in the observed range.
        assert pd_out["signup_date_year"].between(2018, 2023).all()
        # Polars and pandas paths must agree exactly.
        assert pl_out["signup_date_year"].to_list() == pd_out["signup_date_year"].tolist()

    def test_date_features_on_raw_string_dates_polars_matches_pandas(self) -> None:
        """Regression test: polars must not silently null out plain date strings.

        Unlike the parity test above (which pre-converts the column via
        ``pd.to_datetime`` before building the polars frame, masking the bug),
        this builds the polars frame directly from raw ``YYYY-MM-DD`` strings
        — the common shape of a column freshly loaded from CSV.
        """
        df = load_sample_dataset("customers")
        assert df["signup_date"].dtype == object  # raw strings, not yet parsed
        art = DateFeaturesCalculator().fit(
            df, {"columns": ["signup_date"], "features": ["year", "month", "dayofweek"]}
        )
        pd_out = DateFeaturesApplier().apply(df, art)

        pl_raw = pl.DataFrame({"signup_date": df["signup_date"].tolist()})
        pl_out = DateFeaturesApplier().apply(pl_raw, art)

        assert not pl_out["signup_date_year"].is_null().any()
        assert pl_out["signup_date_year"].to_list() == pd_out["signup_date_year"].tolist()
        assert pl_out["signup_date_month"].to_list() == pd_out["signup_date_month"].tolist()
