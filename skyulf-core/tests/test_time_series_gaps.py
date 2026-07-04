"""Coverage-gap tests for time_series: lag, rolling, date_features, _common.

Complements ``test_time_series_nodes.py`` with edge cases: insufficient
history for lag/rolling windows, sort_by-triggered reordering, drop_na,
weekofyear/is_month_start features, and the shared ``_common.py`` coercion
helpers.
"""

import pandas as pd
import polars as pl

from skyulf.preprocessing.time_series._common import (
    coerce_aggregations,
    coerce_lags,
    resolve_columns,
    sort_pandas,
)
from skyulf.preprocessing.time_series.date_features import (
    DateFeaturesApplier,
    DateFeaturesCalculator,
)
from skyulf.preprocessing.time_series.lag import LagFeaturesApplier, LagFeaturesCalculator
from skyulf.preprocessing.time_series.rolling import (
    RollingAggregateApplier,
    RollingAggregateCalculator,
)

# ---------------------------------------------------------------------------
# _common.py helpers
# ---------------------------------------------------------------------------


def test_resolve_columns_filters_to_available() -> None:
    """Only columns present in ``available`` are kept, order preserved."""
    assert resolve_columns(["a", "missing", "b"], ["a", "b"]) == ["a", "b"]


def test_resolve_columns_empty_input_returns_empty() -> None:
    """An empty/None ``columns`` config resolves to an empty list."""
    assert resolve_columns(None, ["a"]) == []
    assert resolve_columns([], ["a"]) == []


def test_coerce_lags_accepts_single_int() -> None:
    """A bare int ``lags`` value is wrapped into a single-element list."""
    assert coerce_lags(5) == [5]


def test_coerce_lags_drops_non_positive_and_dedupes() -> None:
    """Zero/negative lags are dropped and duplicates are removed, sorted."""
    assert coerce_lags([2, 2, 0, -1, 1]) == [1, 2]


def test_coerce_aggregations_accepts_single_string() -> None:
    """A bare string ``aggregations`` value is wrapped into a list."""
    assert coerce_aggregations("mean") == ["mean"]


def test_coerce_aggregations_filters_unknown_names() -> None:
    """Unrecognised aggregation names are dropped, order preserved."""
    assert coerce_aggregations(["mean", "bogus", "sum"]) == ["mean", "sum"]


def test_sort_pandas_noop_without_sort_by() -> None:
    """Without a ``sort_by`` column, the frame is returned unchanged."""
    df = pd.DataFrame({"v": [3, 1, 2]})
    out = sort_pandas(df, None)
    assert out["v"].tolist() == [3, 1, 2]


def test_sort_pandas_sorts_stable() -> None:
    """A configured ``sort_by`` column stable-sorts the frame."""
    df = pd.DataFrame({"t": [3, 1, 2], "v": ["c", "a", "b"]})
    out = sort_pandas(df, "t")
    assert out["v"].tolist() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# LagFeatures edge cases
# ---------------------------------------------------------------------------


def test_lag_features_insufficient_history_produces_nan() -> None:
    """A lag larger than the frame's row count produces all-NaN lag values."""
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    art = LagFeaturesCalculator().fit(df, {"columns": ["v"], "lags": [5]})
    out = LagFeaturesApplier().apply(df, art)
    assert out["v_lag_5"].isna().all()


def test_lag_features_drop_na_removes_incomplete_rows() -> None:
    """``drop_na=True`` removes rows containing a NaN lag value."""
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    art = LagFeaturesCalculator().fit(df, {"columns": ["v"], "lags": [1], "drop_na": True})
    out = LagFeaturesApplier().apply(df, art)
    assert len(out) == 2
    assert out["v_lag_1"].notna().all()


def test_lag_features_sort_by_reorders_before_lagging() -> None:
    """``sort_by`` sorts rows before computing lags (polars + pandas parity)."""
    df = pd.DataFrame({"t": [3, 1, 2], "v": [30.0, 10.0, 20.0]})
    art = LagFeaturesCalculator().fit(df, {"columns": ["v"], "lags": [1], "sort_by": "t"})
    pandas_out = LagFeaturesApplier().apply(df, art)
    polars_out = LagFeaturesApplier().apply(pl.from_pandas(df), art)
    assert pandas_out["v_lag_1"].fillna(-1).tolist() == [-1.0, 10.0, 20.0]
    pl_vals = [v if v is not None else -1.0 for v in polars_out["v_lag_1"].to_list()]
    assert pl_vals == [-1.0, 10.0, 20.0]


def test_lag_features_apply_noop_without_columns_or_lags() -> None:
    """Missing ``columns``/``lags`` leaves the frame unchanged."""
    df = pd.DataFrame({"v": [1.0, 2.0]})
    out = LagFeaturesApplier().apply(df, {"columns": [], "lags": [1]})
    assert list(out.columns) == ["v"]


def test_lag_features_infer_output_schema_skips_unresolved_columns() -> None:
    """Configured columns absent from the schema are skipped, not erroring."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["v"], {"v": "float64"})
    out = LagFeaturesCalculator().infer_output_schema(
        schema, {"columns": ["v", "missing"], "lags": [1]}
    )
    assert out is not None
    assert "v_lag_1" in out
    assert "missing_lag_1" not in out


# ---------------------------------------------------------------------------
# RollingAggregate edge cases
# ---------------------------------------------------------------------------


def test_rolling_aggregate_insufficient_history_with_min_periods() -> None:
    """min_periods below window still produces values for early rows."""
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    art = RollingAggregateCalculator().fit(
        df, {"columns": ["v"], "window": 5, "aggregations": ["mean"], "min_periods": 1}
    )
    out = RollingAggregateApplier().apply(df, art)
    assert out["v_roll_mean_5"].notna().all()


def test_rolling_aggregate_min_periods_forces_nan_when_unmet() -> None:
    """When ``min_periods`` exceeds available history, early rows are NaN."""
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    art = RollingAggregateCalculator().fit(
        df, {"columns": ["v"], "window": 3, "aggregations": ["mean"], "min_periods": 3}
    )
    out = RollingAggregateApplier().apply(df, art)
    assert out["v_roll_mean_3"].isna().tolist() == [True, True, False]


def test_rolling_aggregate_group_by_computed_within_groups() -> None:
    """``group_by`` computes the rolling aggregate independently per group."""
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 100.0, 200.0]})
    art = RollingAggregateCalculator().fit(
        df,
        {
            "columns": ["v"],
            "window": 2,
            "aggregations": ["sum"],
            "min_periods": 1,
            "group_by": ["g"],
        },
    )
    out = RollingAggregateApplier().apply(df, art)
    assert out["v_roll_sum_2"].tolist() == [1.0, 4.0, 100.0, 300.0]


def test_rolling_aggregate_apply_noop_without_columns_or_aggs() -> None:
    """Missing ``columns``/``aggregations`` leaves the frame unchanged."""
    df = pd.DataFrame({"v": [1.0, 2.0]})
    out = RollingAggregateApplier().apply(df, {"columns": [], "aggregations": ["mean"]})
    assert list(out.columns) == ["v"]


# ---------------------------------------------------------------------------
# DateFeatures edge cases
# ---------------------------------------------------------------------------


def test_date_features_weekofyear_and_is_month_start() -> None:
    """weekofyear and is_month_start are computed correctly, pandas + polars."""
    df = pd.DataFrame({"d": pd.to_datetime(["2021-01-01", "2021-01-04"])})
    art = DateFeaturesCalculator().fit(
        df, {"columns": ["d"], "features": ["weekofyear", "is_month_start"]}
    )
    pandas_out = DateFeaturesApplier().apply(df, art)
    polars_out = DateFeaturesApplier().apply(pl.from_pandas(df), art)
    assert pandas_out["d_is_month_start"].tolist() == [1, 0]
    assert polars_out["d_is_month_start"].to_list() == pandas_out["d_is_month_start"].tolist()


def test_date_features_default_features_when_unspecified() -> None:
    """Without a ``features`` config, the default calendar parts are used."""
    art = DateFeaturesCalculator().fit(
        pd.DataFrame({"d": pd.to_datetime(["2021-01-01"])}), {"columns": ["d"]}
    )
    assert art["features"] == ["year", "month", "day", "dayofweek"]


def test_date_features_invalid_feature_name_filtered_out() -> None:
    """Unknown feature names are silently dropped at fit time."""
    art = DateFeaturesCalculator().fit(
        pd.DataFrame({"d": pd.to_datetime(["2021-01-01"])}),
        {"columns": ["d"], "features": ["bogus", "year"]},
    )
    assert art["features"] == ["year"]


def test_date_features_apply_noop_without_columns_or_features() -> None:
    """Missing ``columns``/``features`` leaves the frame unchanged."""
    df = pd.DataFrame({"d": pd.to_datetime(["2021-01-01"])})
    out = DateFeaturesApplier().apply(df, {"columns": [], "features": ["year"]})
    assert list(out.columns) == ["d"]


def test_date_features_infer_output_schema_skips_unresolved_columns() -> None:
    """Columns absent from the input schema are skipped during inference."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["d"], {"d": "datetime"})
    out = DateFeaturesCalculator().infer_output_schema(
        schema, {"columns": ["d", "missing"], "features": ["year"]}
    )
    assert out is not None
    assert "d_year" in out
    assert "missing_year" not in out
