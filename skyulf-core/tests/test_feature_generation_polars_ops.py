"""Tests for the Polars-engine op handlers in feature_generation/_polars_ops.py.

Covers arithmetic, ratio, similarity, group_agg, datetime extraction, and
the top-level `_featgen_apply_polars` dispatcher, including edge cases
(division by zero, missing columns, unknown op types, round_digits).
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.feature_generation import (
    _featgen_apply_pandas,
    _featgen_apply_polars,
)
from skyulf.preprocessing.feature_generation._polars_ops import (
    _POLARS_DT_FEATURES,
    _polars_arith,
    _polars_arith_terms,
    _polars_datetime_apply,
    _polars_divide,
    _polars_group_agg,
    _polars_ratio,
    _polars_similarity,
    _register_polars_dt,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_DF = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 0.0, 40.0]})

_arith_value_cases = TestCaseLoader(
    "preprocessing/feature_generation_polars_ops", group="arith_values"
).load()
_arith_none_cases = TestCaseLoader(
    "preprocessing/feature_generation_polars_ops", group="arith_none"
).load()
_group_agg_value_cases = TestCaseLoader(
    "preprocessing/feature_generation_polars_ops", group="group_agg_values"
).load()
_group_agg_none_cases = TestCaseLoader(
    "preprocessing/feature_generation_polars_ops", group="group_agg_none"
).load()
_ratio_none_cases = TestCaseLoader(
    "preprocessing/feature_generation_polars_ops", group="ratio_none"
).load()


def _make_pl() -> pl.DataFrame:
    """Return a small deterministic Polars DataFrame."""
    return pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})


# ---------------------------------------------------------------------------
# _polars_arith_terms
# ---------------------------------------------------------------------------


class TestPolarsArithTerms:
    def test_valid_columns_returned(self) -> None:
        """Only columns present in `existing` should appear in col_exprs."""
        op = {"input_columns": ["a", "b", "missing"], "constants": [5.0]}
        col_exprs, const_vals = _polars_arith_terms(op, ["a", "b"])
        assert len(col_exprs) == 2
        assert const_vals == [5.0]

    def test_secondary_columns_included(self) -> None:
        """secondary_columns are merged with input_columns for arith terms."""
        op = {"input_columns": ["a"], "secondary_columns": ["b"]}
        col_exprs, _ = _polars_arith_terms(op, ["a", "b"])
        assert len(col_exprs) == 2

    def test_fillna_custom_value(self) -> None:
        """Custom fillna should propagate through to expressions without error."""
        op = {"input_columns": ["a"], "fillna": 99.0}
        col_exprs, _ = _polars_arith_terms(op, ["a"])
        # Evaluate to confirm no expression-build error.
        result = _make_pl().select(col_exprs[0].alias("v"))
        assert result["v"].to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_empty_op_yields_empty_lists(self) -> None:
        """An op with no columns and no constants returns two empty lists."""
        col_exprs, const_vals = _polars_arith_terms({}, ["a", "b"])
        assert col_exprs == []
        assert const_vals == []


# ---------------------------------------------------------------------------
# Arithmetic operations via _polars_arith
# ---------------------------------------------------------------------------


class TestPolarsArithOps:
    _EPS = 1e-9

    def _eval(self, op: dict) -> list:
        """Helper: run op through _polars_arith and return result list."""
        expr = _polars_arith(op, list(_make_pl().columns), self._EPS)
        assert expr is not None
        return _make_pl().select(expr.alias("r"))["r"].to_list()

    @pytest.mark.parametrize(*_arith_value_cases)
    def test_arith_ops_produce_expected_values(
        self, method: str, input_columns: list[str], constants: list[float], expected: list[float]
    ) -> None:
        """Success-path arithmetic ops (add/subtract/multiply, column or constant operands)."""
        vals = self._eval(
            {"method": method, "input_columns": input_columns, "constants": constants}
        )
        assert vals == expected

    def test_divide_two_columns(self) -> None:
        """a / b element-wise (no zero denominators here)."""
        vals = self._eval({"method": "divide", "input_columns": ["a", "b"]})
        expected = [1 / 10, 2 / 20, 3 / 30, 4 / 40]
        np.testing.assert_allclose(vals, expected, rtol=1e-9)

    def test_divide_zero_denominator_safe(self) -> None:
        """Zero denominator must not produce inf; epsilon guard kicks in."""
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [0.0, 0.0]})
        expr = _polars_arith(
            {"method": "divide", "input_columns": ["a", "b"]},
            ["a", "b"],
            1e-9,
        )
        assert expr is not None
        result = df.select(expr.alias("r"))["r"].to_list()
        # Must not be inf or NaN.
        assert all(abs(v) < 1e18 for v in result)

    @pytest.mark.parametrize(*_arith_none_cases)
    def test_arith_returns_none_for_invalid_ops(
        self, method: str, input_columns: list[str]
    ) -> None:
        """Unrecognised method or unresolvable columns must return None, not raise."""
        result = _polars_arith({"method": method, "input_columns": input_columns}, ["a", "b"], 1e-9)
        assert result is None


# ---------------------------------------------------------------------------
# _polars_divide edge cases
# ---------------------------------------------------------------------------


class TestPolarsDivide:
    _EPS = 1e-9

    def test_constants_only(self) -> None:
        """divide with constants only (no column exprs) should work."""
        result = _polars_divide([], [10.0, 2.0], self._EPS)
        assert result is not None
        val = pl.DataFrame({"dummy": [1]}).select(result.alias("r"))["r"][0]
        assert val == pytest.approx(5.0)

    def test_no_cols_no_consts_returns_none(self) -> None:
        """Calling divide with empty lists must return None, not raise."""
        assert _polars_divide([], [], self._EPS) is None

    def test_near_zero_constant_uses_epsilon(self) -> None:
        """Constants smaller than epsilon should be replaced by epsilon."""
        result = _polars_divide([], [1.0, 0.0], self._EPS)
        assert result is not None
        # 1 / epsilon = 1e9; just ensure no crash and value is finite.
        val = pl.DataFrame({"dummy": [1]}).select(result.alias("r"))["r"][0]
        assert abs(val) < 1e18


# ---------------------------------------------------------------------------
# _polars_ratio
# ---------------------------------------------------------------------------


class TestPolarsRatio:
    def test_basic_ratio(self) -> None:
        """Ratio of a / b with no near-zero denominators."""
        op = {"input_columns": ["a"], "secondary_columns": ["b"]}
        expr = _polars_ratio(op, ["a", "b"], 1e-9)
        assert expr is not None
        result = _make_pl().select(expr.alias("r"))["r"].to_list()
        np.testing.assert_allclose(result, [0.1, 0.1, 0.1, 0.1], rtol=1e-6)

    def test_zero_denominator_safe(self) -> None:
        """Zero denominator column must not produce inf."""
        df = pl.DataFrame({"n": [1.0, 2.0], "d": [0.0, 0.0]})
        op = {"input_columns": ["n"], "secondary_columns": ["d"]}
        expr = _polars_ratio(op, ["n", "d"], 1e-9)
        assert expr is not None
        result = df.select(expr.alias("r"))["r"].to_list()
        assert all(abs(v) < 1e18 for v in result)

    @pytest.mark.parametrize(*_ratio_none_cases)
    def test_ratio_returns_none_for_unresolvable_columns(
        self, op: dict, existing: list[str]
    ) -> None:
        """A ratio with an unresolvable numerator or denominator column returns None."""
        assert _polars_ratio(op, existing, 1e-9) is None


# ---------------------------------------------------------------------------
# _polars_similarity
# ---------------------------------------------------------------------------


class TestPolarsSimilarity:
    def test_identical_strings(self) -> None:
        """Identical strings should give 100.0 similarity score."""
        df = pl.DataFrame({"x": ["hello", "world"], "y": ["hello", "world"]})
        op = {"input_columns": ["x", "y"]}
        expr = _polars_similarity(op, ["x", "y"], 1e-9)
        assert expr is not None
        result = df.select(expr.alias("r"))["r"].to_list()
        assert all(v == 100.0 for v in result)

    def test_both_empty_gives_100(self) -> None:
        """Two empty strings should be 100% similar by convention."""
        df = pl.DataFrame({"x": ["", ""], "y": ["", ""]})
        op = {"input_columns": ["x", "y"]}
        expr = _polars_similarity(op, ["x", "y"], 1e-9)
        assert expr is not None
        result = df.select(expr.alias("r"))["r"].to_list()
        assert all(v == 100.0 for v in result)

    def test_one_empty_gives_0(self) -> None:
        """One empty side with a non-empty other side means zero similarity."""
        df = pl.DataFrame({"x": ["hello", ""], "y": ["", "world"]})
        op = {"input_columns": ["x", "y"]}
        expr = _polars_similarity(op, ["x", "y"], 1e-9)
        assert expr is not None
        result = df.select(expr.alias("r"))["r"].to_list()
        assert all(v == 0.0 for v in result)

    def test_missing_column_returns_none(self) -> None:
        """A similarity op referencing a non-existent column returns None."""
        op = {"input_columns": ["x", "missing"]}
        assert _polars_similarity(op, ["x"], 1e-9) is None

    def test_scores_in_range(self) -> None:
        """All computed scores must lie within [0, 100]."""
        df = pl.DataFrame({"x": ["abc", "foo bar"], "y": ["xyz", "foo"]})
        op = {"input_columns": ["x", "y"]}
        expr = _polars_similarity(op, ["x", "y"], 1e-9)
        assert expr is not None
        result = df.select(expr.alias("r"))["r"].to_list()
        assert all(0.0 <= v <= 100.0 for v in result)


# ---------------------------------------------------------------------------
# _polars_group_agg — aggregation methods not covered by existing tests
# ---------------------------------------------------------------------------


class TestPolarsGroupAgg:
    _DATA = {"dept": ["A", "A", "B", "B", "B"], "salary": [100.0, 200.0, 300.0, 400.0, 500.0]}

    def _run_agg(self, method: str) -> list:
        """Run a group_agg op and return result column as list."""
        df = pl.DataFrame(self._DATA)
        params = {
            "operations": [
                {
                    "operation_type": "group_agg",
                    "method": method,
                    "input_columns": ["dept"],
                    "secondary_columns": ["salary"],
                    "output_column": f"out_{method}",
                }
            ]
        }
        out, _ = _featgen_apply_polars(df, None, params)
        return out[f"out_{method}"].to_list()

    @pytest.mark.parametrize(*_group_agg_value_cases)
    def test_group_agg_produces_expected_values(self, method: str, expected: list[float]) -> None:
        """Group-aggregation methods (sum/min/max/median) must broadcast per-group results."""
        vals = self._run_agg(method)
        assert vals == expected

    def test_std_returns_floats(self) -> None:
        """Group std must return a float column without raising."""
        vals = self._run_agg("std")
        assert all(isinstance(v, float) for v in vals)

    @pytest.mark.parametrize(*_group_agg_none_cases)
    def test_group_agg_returns_none_for_invalid_ops(
        self, method: str, input_columns: list[str]
    ) -> None:
        """An unregistered method or unresolvable group column must return None, not raise."""
        op = {"method": method, "input_columns": input_columns, "secondary_columns": ["salary"]}
        result = _polars_group_agg(op, ["dept", "salary"], 1e-9)
        assert result is None


# ---------------------------------------------------------------------------
# _polars_datetime_apply
# ---------------------------------------------------------------------------


class TestPolarsDatetimeApply:
    def test_datetime_column_extracts_year_month(self) -> None:
        """Datetime extraction must produce correctly named and valued columns."""
        df = pl.DataFrame({"dt": pl.Series(["2024-01-15", "2024-07-04"]).str.to_datetime()})
        op = {
            "input_columns": ["dt"],
            "datetime_features": ["year", "month"],
        }
        out = _polars_datetime_apply(op, df)
        assert "dt_year" in out.columns
        assert "dt_month" in out.columns
        assert out["dt_year"].to_list() == [2024, 2024]
        assert out["dt_month"].to_list() == [1, 7]

    def test_string_datetime_column_cast(self) -> None:
        """String-typed datetime columns must be cast before extraction."""
        df = pl.DataFrame({"ts": ["2023-12-31", "2022-03-07"]})
        assert df.schema["ts"] == pl.String
        op = {"input_columns": ["ts"], "datetime_features": ["year", "day"]}
        out = _polars_datetime_apply(op, df)
        assert "ts_year" in out.columns
        assert out["ts_year"].to_list() == [2023, 2022]

    def test_no_valid_columns_passthrough(self) -> None:
        """When input_columns don't exist, frame is returned unchanged."""
        df = pl.DataFrame({"a": [1, 2]})
        op = {"input_columns": ["nonexistent"], "datetime_features": ["year"]}
        out = _polars_datetime_apply(op, df)
        assert list(out.columns) == ["a"]

    def test_is_weekend_flag(self) -> None:
        """is_weekend must be 1 for Sunday (2023-12-31) and 0 for Monday."""
        df = pl.DataFrame({"dt": pl.Series(["2023-12-31", "2024-01-15"]).str.to_datetime()})
        op = {"input_columns": ["dt"], "datetime_features": ["is_weekend"]}
        out = _polars_datetime_apply(op, df)
        assert out["dt_is_weekend"].to_list() == [1, 0]

    def test_bad_column_does_not_drop_good_column_features(self) -> None:
        """A column that fails datetime extraction must not prevent other
        columns in the same op from producing their features (per-column
        isolation, matching the pandas engine's behaviour)."""
        df = pl.DataFrame(
            {
                "good": pl.Series(["2024-01-15", "2024-07-04"]).str.to_datetime(),
                "bad": [1, 2],
            }
        )
        op = {"input_columns": ["bad", "good"], "datetime_features": ["year", "month"]}
        out = _polars_datetime_apply(op, df)
        assert "good_year" in out.columns
        assert "good_month" in out.columns
        assert out["good_year"].to_list() == [2024, 2024]
        assert "bad_year" not in out.columns

    def test_register_polars_dt_idempotent(self) -> None:
        """Calling _register_polars_dt multiple times must not cause errors."""
        _register_polars_dt()
        _register_polars_dt()
        assert "year" in _POLARS_DT_FEATURES
        assert "month_name" in _POLARS_DT_FEATURES


# ---------------------------------------------------------------------------
# _featgen_apply_polars — integration / edge cases
# ---------------------------------------------------------------------------


class TestFeatgenApplyPolars:
    def test_empty_operations_returns_unchanged(self) -> None:
        """No ops must return the original frame untouched."""
        df = _make_pl()
        out, _ = _featgen_apply_polars(df, None, {"operations": []})
        assert out.shape == df.shape

    def test_unknown_op_type_skipped_gracefully(self) -> None:
        """Unknown op types must be skipped silently; original columns preserved."""
        df = _make_pl()
        out, _ = _featgen_apply_polars(
            df,
            None,
            {"operations": [{"operation_type": "black_magic", "input_columns": ["a"]}]},
        )
        assert "a" in out.columns
        assert out.shape[0] == df.shape[0]

    def test_round_digits_applied(self) -> None:
        """round_digits on an arithmetic result must limit decimal precision."""
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 7.0]})
        params = {
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "divide",
                    "input_columns": ["a", "b"],
                    "output_column": "result",
                    "round_digits": 2,
                }
            ]
        }
        out, _ = _featgen_apply_polars(df, None, params)
        for v in out["result"].to_list():
            assert v == round(v, 2)

    def test_allow_overwrite_replaces_column(self) -> None:
        """allow_overwrite=True must let two ops use the same output_column."""
        df = _make_pl()
        params = {
            "allow_overwrite": True,
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "add",
                    "input_columns": ["a", "b"],
                    "output_column": "out",
                },
                {
                    "operation_type": "arithmetic",
                    "method": "subtract",
                    "input_columns": ["a", "b"],
                    "output_column": "out",
                },
            ],
        }
        out, _ = _featgen_apply_polars(df, None, params)
        # Second op overwrites; "out" exists and holds subtract values.
        assert "out" in out.columns
        assert out["out"].to_list() == [-9.0, -18.0, -27.0, -36.0]

    def test_y_passed_through_unchanged(self) -> None:
        """y is never modified by _featgen_apply_polars."""
        df = _make_pl()
        y = pl.Series("target", [0, 1, 0, 1])
        params = {
            "operations": [
                {"operation_type": "arithmetic", "method": "add", "input_columns": ["a", "b"]}
            ]
        }
        _, y_out = _featgen_apply_polars(df, y, params)
        assert y_out.to_list() == [0, 1, 0, 1]

    def test_malformed_op_skipped_without_raising(self) -> None:
        """A completely broken op dict should not crash the pipeline."""
        df = _make_pl()
        out, _ = _featgen_apply_polars(df, None, {"operations": [{"operation_type": None}]})
        assert "a" in out.columns


# ---------------------------------------------------------------------------
# Engine parity: _featgen_apply_polars vs _featgen_apply_pandas
# ---------------------------------------------------------------------------

_FINITE_FLOAT = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


@st.composite
def _two_col_frame(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a small two-column pandas DataFrame for parity testing."""
    n = draw(st.integers(min_value=5, max_value=30))
    a = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
    # Avoid zeros in b so divide doesn't hit the epsilon guard asymmetrically.
    b = draw(
        st.lists(
            st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return pd.DataFrame({"a": a, "b": b})


@pytest.mark.parametrize(
    "op_type,method",
    [
        ("arithmetic", "add"),
        ("arithmetic", "subtract"),
        ("arithmetic", "multiply"),
        ("arithmetic", "divide"),
    ],
)
@settings(max_examples=25, deadline=None)
@given(df_pd=_two_col_frame())
def test_arith_engine_parity(op_type: str, method: str, df_pd: pd.DataFrame) -> None:
    """Polars and pandas arithmetic ops must produce numerically identical outputs."""
    params = {
        "operations": [
            {
                "operation_type": op_type,
                "method": method,
                "input_columns": ["a", "b"],
                "output_column": "result",
            }
        ]
    }
    out_pd, _ = _featgen_apply_pandas(df_pd, None, params)
    df_pl = pl.from_pandas(df_pd)
    out_pl, _ = _featgen_apply_polars(df_pl, None, params)

    pd_vals = out_pd["result"].to_numpy()
    pl_vals = out_pl["result"].to_numpy()
    np.testing.assert_allclose(pd_vals, pl_vals, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Real-shaped dataset: null handling in polars arithmetic ops
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Verify that _featgen_apply_polars handles the customers.csv sample,
    which contains null (NaN) values in age/income, without crashing and
    preserves row count — nulls must propagate in arithmetic results, not
    cause silent row removal.
    """

    def test_add_op_on_null_containing_columns_preserves_all_rows(self) -> None:
        """An arithmetic add on age+income must not drop rows that have nulls —
        row count must be identical to the input even when null-filling is applied."""
        df = load_sample_dataset("customers", engine="polars")
        params = {
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "add",
                    "input_columns": ["age", "income"],
                    "output_column": "age_plus_income",
                }
            ]
        }
        out, _ = _featgen_apply_polars(df, None, params)
        # Row count must be preserved — nulls must not cause row removal.
        assert out.shape[0] == df.shape[0]
        assert "age_plus_income" in out.columns
        # Rows where both age and income have values must produce a non-null result.
        both_present = df["age"].is_not_null() & df["income"].is_not_null()
        assert out.filter(both_present)["age_plus_income"].is_not_null().all()
