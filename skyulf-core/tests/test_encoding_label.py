"""Tests for the LabelEncoder Calculator/Applier (skyulf.preprocessing.encoding.label)."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.encoding.label import (
    LabelEncoderApplier,
    LabelEncoderCalculator,
)

_maybe_pull_y_cases = TestCaseLoader("preprocessing/encoding_label").load_with_ids()

settings.register_profile(
    "encoding_label",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("encoding_label")


def test_fit_apply_roundtrip_sorts_classes_alphabetically() -> None:
    """LabelEncoder assigns codes in sklearn's default alphabetical class order."""
    df = pd.DataFrame({"category": ["b", "a", "c", "a", "b"]})
    calc = LabelEncoderCalculator()
    applier = LabelEncoderApplier()

    params = calc.fit(df, {"columns": ["category"]})
    result = applier.apply(df, dict(params))

    assert params["classes_count"] == {"category": 3}
    expected = {"a": 0, "b": 1, "c": 2}
    for raw, code in zip(df["category"], result["category"], strict=True):
        assert code == expected[raw]


def test_unseen_category_at_apply_maps_to_missing_code() -> None:
    """A category unseen during fit maps to the configured missing_code (-1 by default)."""
    train = pd.DataFrame({"category": ["a", "b", "a"]})
    calc = LabelEncoderCalculator()
    applier = LabelEncoderApplier()
    params = calc.fit(train, {"columns": ["category"]})

    test_df = pd.DataFrame({"category": ["a", "unseen"]})
    result = applier.apply(test_df, dict(params))

    assert result["category"].iloc[0] == 0
    assert result["category"].iloc[1] == -1


def test_custom_missing_code_is_respected() -> None:
    """A non-default missing_code configuration flows through fit and apply."""
    train = pd.DataFrame({"category": ["a", "b"]})
    calc = LabelEncoderCalculator()
    applier = LabelEncoderApplier()
    params = calc.fit(train, {"columns": ["category"], "missing_code": -99})

    assert params["missing_code"] == -99
    result = applier.apply(pd.DataFrame({"category": ["unseen"]}), dict(params))
    assert result["category"].iloc[0] == -99


def test_target_column_named_in_columns_gets_encoded() -> None:
    """When y's name is listed in `columns`, the target is also label-encoded."""
    X = pd.DataFrame({"category": ["a", "b", "a"]})
    y = pd.Series(["yes", "no", "yes"], name="target")
    calc = LabelEncoderCalculator()
    applier = LabelEncoderApplier()

    params = calc.fit((X, y), {"columns": ["category", "target"]})
    assert "__target__" in params["encoders"]

    X_out, y_out = applier.apply((X, y), dict(params))
    # "no" < "yes" alphabetically -> no=0, yes=1
    assert list(y_out) == [1, 0, 1]
    assert list(X_out["category"]) == [0, 1, 0]


def test_target_encoded_when_no_feature_columns_configured() -> None:
    """`_maybe_fit_target` fits y whenever `cols` is falsy, regardless of y.name."""
    X = pd.DataFrame({"category": ["a", "b"]})
    y = pd.Series(["p", "q"], name="anything")
    calc = LabelEncoderCalculator()

    params = calc.fit((X, y), {})
    assert "__target__" in params["encoders"]
    assert params["classes_count"]["__target__"] == 2


def test_empty_dataframe_with_no_columns_returns_empty_artifact() -> None:
    """Fitting with no configured columns and no rows still returns a valid (empty) artifact."""
    df = pd.DataFrame({"category": pd.Series([], dtype=object)})
    calc = LabelEncoderCalculator()

    params = calc.fit(df, {"columns": None})
    assert params["encoders"] == {}
    assert params["classes_count"] == {}


def test_no_columns_and_no_target_warns_about_noop(caplog: pytest.LogCaptureFixture) -> None:
    """Fitting with no configured columns and no y logs a warning about the no-op."""
    df = pd.DataFrame({"category": ["a", "b", "c"]})
    calc = LabelEncoderCalculator()

    with caplog.at_level("WARNING", logger="skyulf.preprocessing.encoding.label"):
        params = calc.fit(df, {"columns": None})

    assert params["encoders"] == {}
    assert any("no-op" in record.message for record in caplog.records)


def test_single_row_dataframe() -> None:
    """A single-row frame produces exactly one class."""
    df = pd.DataFrame({"category": ["only"]})
    calc = LabelEncoderCalculator()
    applier = LabelEncoderApplier()

    params = calc.fit(df, {"columns": ["category"]})
    result = applier.apply(df, dict(params))

    assert params["classes_count"] == {"category": 1}
    assert result["category"].iloc[0] == 0


def test_all_nan_column_becomes_single_string_class() -> None:
    """An all-NaN column stringifies to a constant "nan" class."""
    df = pd.DataFrame({"category": [np.nan, np.nan, np.nan]})
    calc = LabelEncoderCalculator()
    applier = LabelEncoderApplier()

    params = calc.fit(df, {"columns": ["category"]})
    result = applier.apply(df, dict(params))

    assert params["classes_count"] == {"category": 1}
    assert (result["category"] == 0).all()


def test_constant_column_encodes_to_single_code() -> None:
    """A constant categorical column collapses to exactly one class."""
    df = pd.DataFrame({"category": ["same"] * 5})
    calc = LabelEncoderCalculator()
    applier = LabelEncoderApplier()

    params = calc.fit(df, {"columns": ["category"]})
    result = applier.apply(df, dict(params))

    assert params["classes_count"] == {"category": 1}
    assert (result["category"] == 0).all()


def test_unconfigured_column_is_left_untouched() -> None:
    """Columns absent from `encoders` (e.g. filtered out) pass through unchanged."""
    df = pd.DataFrame({"category": ["a", "b"], "other": [1, 2]})
    calc = LabelEncoderCalculator()
    applier = LabelEncoderApplier()

    params = calc.fit(df, {"columns": ["category"]})
    result = applier.apply(df, dict(params))

    assert list(result["other"]) == [1, 2]


# ---------------------------------------------------------------------------
# Polars parity
# ---------------------------------------------------------------------------


@st.composite
def _category_frame(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a small frame with a categorical column drawn from a fixed alphabet."""
    n = draw(st.integers(min_value=3, max_value=30))
    cats = draw(st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=n, max_size=n))
    return pd.DataFrame({"category": cats})


@given(df=_category_frame())
def test_label_fit_engine_parity(df: pd.DataFrame) -> None:
    """Pandas and polars fits must assign identical classes and codes."""
    config: dict[str, Any] = {"columns": ["category"]}
    pd_params = LabelEncoderCalculator().fit(df, dict(config))
    pl_params = LabelEncoderCalculator().fit(pl.from_pandas(df), dict(config))

    assert pd_params["classes_count"] == pl_params["classes_count"]
    pd_le = pd_params["encoders"]["category"]
    pl_le = pl_params["encoders"]["category"]
    assert list(pd_le.classes_) == list(pl_le.classes_)


def test_label_apply_engine_parity_on_unseen_category() -> None:
    """Unknown-category fallback must match between pandas and polars apply paths."""
    train = pd.DataFrame({"category": ["a", "b", "a"]})
    config: dict[str, Any] = {"columns": ["category"], "missing_code": -1}

    pd_params = dict(LabelEncoderCalculator().fit(train, dict(config)))
    pl_params = dict(LabelEncoderCalculator().fit(pl.from_pandas(train), dict(config)))

    test_pd = pd.DataFrame({"category": ["a", "unseen"]})
    test_pl = pl.DataFrame({"category": ["a", "unseen"]})

    out_pd = LabelEncoderApplier().apply(test_pd, pd_params)
    out_pl = LabelEncoderApplier().apply(test_pl, pl_params)

    np.testing.assert_allclose(
        out_pd["category"].to_numpy().astype(float),
        out_pl["category"].to_numpy().astype(float),
    )


def test_polars_apply_target_column_is_encoded() -> None:
    """The polars apply path encodes y via `__target__` when configured."""
    X = pd.DataFrame({"category": ["a", "b", "a"]})
    y = pd.Series(["yes", "no", "yes"], name="target")
    params = dict(LabelEncoderCalculator().fit((X, y), {"columns": ["category", "target"]}))

    X_pl = pl.from_pandas(X)
    y_pl = pl.Series("target", y)
    X_out, y_out = LabelEncoderApplier().apply((X_pl, y_pl), dict(params))

    # "no" < "yes" alphabetically -> no=0, yes=1
    assert list(y_out) == [1, 0, 1]
    assert list(X_out["category"]) == [0, 1, 0]


def test_y_to_str_array_without_to_numpy_uses_np_array_fallback() -> None:
    """_y_to_str_array falls back to np.array(y) for plain Python sequences (no `.to_numpy`)."""
    from skyulf.preprocessing.encoding.label import _y_to_str_array

    result = _y_to_str_array([1, 2, 3])
    np.testing.assert_array_equal(result, np.array(["1", "2", "3"]))


class TestMaybePullYExtractsTargetColumn:
    """`_maybe_pull_y_pandas`/`_maybe_pull_y_polars` pull the target column out of X
    when y is missing. Scenarios (pandas/polars) loaded from
    ``tests/test_cases/preprocessing/encoding_label.json``.
    """

    @pytest.mark.parametrize(
        _maybe_pull_y_cases[0], _maybe_pull_y_cases[1], ids=_maybe_pull_y_cases[2]
    )
    def test_maybe_pull_y_extracts_column_from_x(self, engine: str) -> None:
        data = {"category": ["a", "b"], "target": ["yes", "no"]}
        X = pl.DataFrame(data) if engine == "polars" else pd.DataFrame(data)
        params = LabelEncoderCalculator().fit(
            X, {"columns": ["category", "target"], "target_column": "target"}
        )
        assert "__target__" in params["encoders"]


def test_maybe_fit_target_skips_when_y_name_not_in_columns() -> None:
    """_maybe_fit_target does nothing when `cols` is set but y's name isn't among them."""
    X = pd.DataFrame({"category": ["a", "b"]})
    y = pd.Series(["p", "q"], name="unrelated")
    params = LabelEncoderCalculator().fit((X, y), {"columns": ["category"]})
    assert "__target__" not in params["encoders"]


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.
    ``city`` has one missing value (NaN) — verifies that the LabelEncoder
    stringifies NaN to a "nan" class rather than propagating it as missing.
    """

    def test_city_label_encoding_treats_nan_as_string_class(self) -> None:
        """``city`` contains one NaN; LabelEncoder stringifies it and assigns a valid code.

        Ensures that the NaN row receives a deterministic non-null integer code
        and that all same-city rows share the same code.
        """
        df = load_sample_dataset("customers")
        params = LabelEncoderCalculator().fit(df, {"columns": ["city"]})
        result = LabelEncoderApplier().apply(df, dict(params))

        # No raw NaN must survive encoding.
        assert result["city"].isna().sum() == 0
        # The single NaN row must map to exactly one consistent code.
        nan_mask = df["city"].isna()
        assert result.loc[nan_mask, "city"].nunique() == 1
        # Known cities must each map to a distinct, consistent code.
        for val in df.loc[~nan_mask, "city"].unique():
            mask = df["city"] == val
            assert result.loc[mask, "city"].nunique() == 1
