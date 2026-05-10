"""Reproduce + verify FeatureGeneration `group_agg` op for pandas + polars."""

import pandas as pd
import polars as pl

from skyulf.preprocessing.feature_generation import (
    _featgen_apply_pandas,
    _featgen_apply_polars,
)


def _build_data():
    return {
        "dept": ["A", "A", "B", "B", "B"],
        "salary": [100.0, 200.0, 300.0, 400.0, 500.0],
    }


def test_pandas_group_agg_mean():
    df = pd.DataFrame(_build_data())
    params = {
        "operations": [
            {
                "operation_type": "group_agg",
                "method": "mean",
                "input_columns": ["dept"],
                "secondary_columns": ["salary"],
                "output_column": "dept_mean_salary",
            }
        ]
    }
    out, _ = _featgen_apply_pandas(df, None, params)
    assert "dept_mean_salary" in out.columns
    assert out["dept_mean_salary"].tolist() == [150.0, 150.0, 400.0, 400.0, 400.0]


def test_polars_group_agg_mean():
    df = pl.DataFrame(_build_data())
    params = {
        "operations": [
            {
                "operation_type": "group_agg",
                "method": "mean",
                "input_columns": ["dept"],
                "secondary_columns": ["salary"],
                "output_column": "dept_mean_salary",
            }
        ]
    }
    out, _ = _featgen_apply_polars(df, None, params)
    assert "dept_mean_salary" in out.columns
    vals = out["dept_mean_salary"].to_list()
    assert vals == [150.0, 150.0, 400.0, 400.0, 400.0]


def test_pandas_group_agg_count():
    df = pd.DataFrame(_build_data())
    params = {
        "operations": [
            {
                "operation_type": "group_agg",
                "method": "count",
                "input_columns": ["dept"],
                "secondary_columns": ["salary"],
                "output_column": "dept_n",
            }
        ]
    }
    out, _ = _featgen_apply_pandas(df, None, params)
    assert out["dept_n"].tolist() == [2, 2, 3, 3, 3]


if __name__ == "__main__":
    test_pandas_group_agg_mean()
    test_polars_group_agg_mean()
    test_pandas_group_agg_count()
    print("OK")
