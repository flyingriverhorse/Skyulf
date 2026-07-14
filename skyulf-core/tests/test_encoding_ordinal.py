"""Tests for the OrdinalEncoder Calculator/Applier (skyulf.preprocessing.encoding.ordinal)."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.encoding.ordinal import (
    OrdinalEncoderApplier,
    OrdinalEncoderCalculator,
)

# Reuse the project's shared hypothesis profile instead of re-registering it.
settings.register_profile(
    "encoding_ordinal",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("encoding_ordinal")

_apply_features_exception_cases = TestCaseLoader(
    "preprocessing/encoding_ordinal", group="apply_features_exception"
).load_with_ids()
_apply_target_exception_cases = TestCaseLoader(
    "preprocessing/encoding_ordinal", group="apply_target_exception"
).load_with_ids()


def _fit_apply(df: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the calculator/applier fit->apply round trip and return (params, result)."""
    calc = OrdinalEncoderCalculator()
    applier = OrdinalEncoderApplier()
    params = calc.fit(df, config)
    result = applier.apply(df, dict(params))
    return dict(params), result


def test_fit_apply_roundtrip_sorts_categories_alphabetically() -> None:
    """Ordinal codes follow sklearn's default alphabetical category order."""
    df = pd.DataFrame({"category": ["b", "a", "c", "a", "b"]})
    params, result = _fit_apply(df, {"columns": ["category"]})

    assert params["columns"] == ["category"]
    assert params["categories_count"] == [3]
    expected = {"a": 0.0, "b": 1.0, "c": 2.0}
    for raw, code in zip(df["category"], result["category"], strict=True):
        assert code == expected[raw]


def test_unseen_category_at_apply_maps_to_unknown_value() -> None:
    """A category never seen during fit falls back to the configured unknown_value."""
    train = pd.DataFrame({"category": ["a", "b", "a", "b"]})
    calc = OrdinalEncoderCalculator()
    applier = OrdinalEncoderApplier()
    params = calc.fit(train, {"columns": ["category"], "unknown_value": -1})

    test_df = pd.DataFrame({"category": ["a", "z"]})
    result = applier.apply(test_df, dict(params))

    assert result["category"].iloc[0] == 0.0  # "a" is known, sorted first
    assert result["category"].iloc[1] == -1.0  # "z" is unseen


def test_custom_unknown_value_is_respected() -> None:
    """A non-default unknown_value configuration is passed through to sklearn."""
    train = pd.DataFrame({"category": ["a", "b"]})
    calc = OrdinalEncoderCalculator()
    applier = OrdinalEncoderApplier()
    params = calc.fit(train, {"columns": ["category"], "unknown_value": -99})

    result = applier.apply(pd.DataFrame({"category": ["unseen"]}), dict(params))
    assert result["category"].iloc[0] == -99.0


def test_invalid_handle_unknown_falls_back_to_use_encoded_value() -> None:
    """An unsupported handle_unknown value silently falls back to a safe default."""
    train = pd.DataFrame({"category": ["a", "b"]})
    calc = OrdinalEncoderCalculator()
    params = calc.fit(train, {"columns": ["category"], "handle_unknown": "bogus"})

    encoder = params["encoder_object"]
    assert encoder.handle_unknown == "use_encoded_value"


def test_empty_dataframe_raises_on_fit() -> None:
    """sklearn's OrdinalEncoder requires >=1 sample; an empty frame surfaces that error."""
    df = pd.DataFrame({"category": pd.Series([], dtype=object)})
    calc = OrdinalEncoderCalculator()

    with pytest.raises(ValueError, match="minimum of 1"):
        calc.fit(df, {"columns": ["category"]})


def test_single_row_dataframe() -> None:
    """A single-row frame produces a single category and a valid code."""
    df = pd.DataFrame({"category": ["only"]})
    params, result = _fit_apply(df, {"columns": ["category"]})

    assert params["categories_count"] == [1]
    assert result["category"].iloc[0] == 0.0


def test_all_nan_column_becomes_single_string_category() -> None:
    """An all-NaN column is stringified to a constant "nan" category."""
    df = pd.DataFrame({"category": [np.nan, np.nan, np.nan]})
    params, result = _fit_apply(df, {"columns": ["category"]})

    assert params["categories_count"] == [1]
    assert (result["category"] == 0.0).all()


def test_constant_column_encodes_to_single_code() -> None:
    """A constant categorical column collapses to exactly one category."""
    df = pd.DataFrame({"category": ["same"] * 5})
    params, result = _fit_apply(df, {"columns": ["category"]})

    assert params["categories_count"] == [1]
    assert (result["category"] == 0.0).all()


def test_user_picked_no_columns_still_encodes_target() -> None:
    """columns=[] means "skip features" but the target is still encoded if present."""
    X = pd.DataFrame({"category": ["a", "b"]})
    y = pd.Series(["yes", "no"], name="target")
    calc = OrdinalEncoderCalculator()

    params = calc.fit((X, y), {"columns": []})

    assert params["columns"] == []
    assert params["encoder_object"] is None
    assert "__target__" in params["encoders"]


def test_target_column_named_in_columns_gets_encoded() -> None:
    """When target_column is listed in `columns` but absent from X, encode y too."""
    X = pd.DataFrame({"category": ["a", "b", "a"]})
    y = pd.Series(["yes", "no", "yes"], name="target")
    calc = OrdinalEncoderCalculator()
    applier = OrdinalEncoderApplier()

    config = {"columns": ["category", "target"], "target_column": "target"}
    params = calc.fit((X, y), config)

    assert params["columns"] == ["category"]
    assert "__target__" in params["encoders"]

    X_out, y_out = applier.apply((X, y), dict(params))
    # "no" < "yes" alphabetically -> no=0, yes=1
    assert list(y_out) == [1.0, 0.0, 1.0]
    assert list(X_out["category"]) == [0.0, 1.0, 0.0]


def test_fit_transform_roundtrip_with_target_tuple() -> None:
    """fit() followed by apply() on an (X, y) tuple preserves tuple shape."""
    X = pd.DataFrame({"category": ["x", "y", "x", "z"]})
    y = pd.Series([1, 0, 1, 0], name="label")
    calc = OrdinalEncoderCalculator()
    applier = OrdinalEncoderApplier()

    params = calc.fit((X, y), {"columns": ["category"]})
    X_out, y_out = applier.apply((X, y), dict(params))

    assert isinstance(X_out, pd.DataFrame)
    assert list(y_out) == [1, 0, 1, 0]  # y untouched: not configured as target
    assert list(X_out["category"]) == [0.0, 1.0, 0.0, 2.0]


def test_unresolved_columns_short_circuit_apply() -> None:
    """Applying params whose columns no longer exist in X is a no-op."""
    df = pd.DataFrame({"other": [1, 2, 3]})
    calc = OrdinalEncoderCalculator()
    applier = OrdinalEncoderApplier()
    params = calc.fit(pd.DataFrame({"category": ["a", "b"]}), {"columns": ["category"]})

    result = applier.apply(df, dict(params))
    pd.testing.assert_frame_equal(result, df)


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
def test_ordinal_fit_engine_parity(df: pd.DataFrame) -> None:
    """Pandas and polars fits must assign identical category counts and codes."""
    config = {"columns": ["category"]}
    pd_params = OrdinalEncoderCalculator().fit(df, dict(config))
    pl_params = OrdinalEncoderCalculator().fit(pl.from_pandas(df), dict(config))

    assert pd_params["categories_count"] == pl_params["categories_count"]
    pd_enc = pd_params["encoder_object"]
    pl_enc = pl_params["encoder_object"]
    assert [list(c) for c in pd_enc.categories_] == [list(c) for c in pl_enc.categories_]

    sample = df[["category"]].astype(str).values
    np.testing.assert_allclose(pd_enc.transform(sample), pl_enc.transform(sample))


def test_ordinal_apply_engine_parity_on_unseen_category() -> None:
    """Unknown-category fallback must match between pandas and polars apply paths."""
    train = pd.DataFrame({"category": ["a", "b", "a"]})
    config = {"columns": ["category"], "unknown_value": -1}

    pd_params = dict(OrdinalEncoderCalculator().fit(train, dict(config)))
    pl_params = dict(OrdinalEncoderCalculator().fit(pl.from_pandas(train), dict(config)))

    test_pd = pd.DataFrame({"category": ["a", "unseen"]})
    test_pl = pl.DataFrame({"category": ["a", "unseen"]})

    out_pd = OrdinalEncoderApplier().apply(test_pd, pd_params)
    out_pl = OrdinalEncoderApplier().apply(test_pl, pl_params)

    np.testing.assert_allclose(out_pd["category"].to_numpy(), out_pl["category"].to_numpy())


# ---------------------------------------------------------------------------
# Apply-time error handling and target-column apply paths
# ---------------------------------------------------------------------------


class TestApplyFeaturesExceptionPropagates:
    """A transform-time exception in the feature-apply path propagates (raised),
    with the dispatcher logging the failure. Scenarios (pandas/polars) loaded from
    ``tests/test_cases/preprocessing/encoding_ordinal.json`` (group ``apply_features_exception``).
    """

    @pytest.mark.parametrize(
        _apply_features_exception_cases[0],
        _apply_features_exception_cases[1],
        ids=_apply_features_exception_cases[2],
    )
    def test_apply_features_exception_propagates(
        self, engine: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _BrokenEncoder:
            def transform(self, _x: Any) -> Any:
                raise ValueError("boom")

        params = {"columns": ["category"], "encoder_object": _BrokenEncoder()}
        with caplog.at_level("ERROR"), pytest.raises(ValueError, match="boom"):
            if engine == "polars":
                X = pl.DataFrame({"category": ["a", "b"]})
                OrdinalEncoderApplier().apply(X, dict(params))
            else:
                X = pd.DataFrame({"category": ["a", "b"]})
                OrdinalEncoderApplier().apply(X, dict(params))

        assert any("engine apply failed" in rec.message for rec in caplog.records)


def test_apply_target_polars_encodes_y_correctly() -> None:
    """The polars target-apply path encodes y using the fitted target encoder."""
    X = pd.DataFrame({"category": ["a", "b"]})
    y = pd.Series(["yes", "no"], name="target")
    config = {"columns": ["category", "target"], "target_column": "target"}
    params = dict(OrdinalEncoderCalculator().fit((X, y), config))

    X_pl = pl.from_pandas(X)
    y_pl = pl.Series("target", y)
    X_out, y_out = OrdinalEncoderApplier().apply((X_pl, y_pl), dict(params))

    # "no" < "yes" alphabetically -> no=0, yes=1
    assert list(y_out) == [1.0, 0.0]
    assert list(X_out["category"]) == [0.0, 1.0]


class TestApplyTargetExceptionPropagates:
    """A transform-time exception in the target-apply path propagates (raised),
    with the dispatcher logging the failure. Scenarios (pandas/polars) loaded from
    ``tests/test_cases/preprocessing/encoding_ordinal.json`` (group ``apply_target_exception``).
    """

    @pytest.mark.parametrize(
        _apply_target_exception_cases[0],
        _apply_target_exception_cases[1],
        ids=_apply_target_exception_cases[2],
    )
    def test_apply_target_exception_propagates(
        self, engine: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _BrokenEncoder:
            def transform(self, _x: Any) -> Any:
                raise ValueError("boom")

        params = {
            "columns": [],
            "encoder_object": None,
            "encoders": {"__target__": _BrokenEncoder()},
        }
        with caplog.at_level("ERROR"), pytest.raises(ValueError, match="boom"):
            if engine == "polars":
                y = pl.Series("target", ["yes", "no"])
                X = pl.DataFrame({"other": [1, 2]})
            else:
                y = pd.Series(["yes", "no"], name="target")
                X = pd.DataFrame({"other": [1, 2]})
            OrdinalEncoderApplier().apply((X, y), dict(params))

        assert any("engine apply failed" in rec.message for rec in caplog.records)


def test_polars_apply_returns_input_when_nothing_to_do() -> None:
    """No valid feature columns and no target encoder: apply is a pure no-op."""
    X_pl = pl.DataFrame({"other": [1, 2]})
    params = {"columns": ["category"], "encoder_object": None, "encoders": {}}
    out_pl = OrdinalEncoderApplier().apply(X_pl, dict(params))
    assert out_pl.equals(X_pl)


# ---------------------------------------------------------------------------
# Fit-time edge cases (target categories resolution, empty selections)
# ---------------------------------------------------------------------------


def test_resolve_target_categories_uses_last_row_when_counts_match() -> None:
    """categories_order with an extra trailing row supplies explicit target categories."""
    X = pd.DataFrame({"category": ["a", "b", "a"]})
    y = pd.Series(["yes", "no", "yes"], name="target")
    config = {
        "columns": ["category", "target"],
        "target_column": "target",
        "categories_order": "a,b\nno,yes",
    }
    params = OrdinalEncoderCalculator().fit((X, y), config)
    target_enc = params["encoders"]["__target__"]
    assert list(target_enc.categories_[0]) == ["no", "yes"]


def test_no_feature_columns_and_no_target_encoding_returns_empty() -> None:
    """When resolve_columns yields nothing and the target isn't configured, fit() returns {}."""
    X = pd.DataFrame({"amount": [1, 2, 3]})
    params = OrdinalEncoderCalculator().fit(X, {})
    assert params == {}


def test_should_encode_target_true_when_target_named_in_columns() -> None:
    """_should_encode_target fires when target_column is listed in `columns` but absent from X."""
    X = pd.DataFrame({"category": ["a", "b"]})
    y = pd.Series(["yes", "no"], name="target")
    config = {"columns": ["category", "target"], "target_column": "target"}
    params = OrdinalEncoderCalculator().fit((X, y), config)
    assert "__target__" in params["encoders"]


def test_should_encode_target_false_when_no_target_name_resolvable() -> None:
    """_should_encode_target returns False when neither target_column nor y.name is set."""
    X = pd.DataFrame({"category": ["a", "b"]})
    y = pd.Series(["yes", "no"])  # unnamed series
    params = OrdinalEncoderCalculator().fit((X, y), {"columns": ["category"]})
    assert "__target__" not in params["encoders"]


def test_no_feature_columns_but_target_encoding_fits_target_only() -> None:
    """When no feature columns resolve but the target is configured, only y gets encoded."""
    X = pd.DataFrame({"amount": [1, 2, 3]})
    y = pd.Series(["yes", "no", "yes"], name="target")
    config = {"columns": ["target"], "target_column": "target"}
    params = OrdinalEncoderCalculator().fit((X, y), config)

    assert params["columns"] == []
    assert params["encoder_object"] is None
    assert "__target__" in params["encoders"]


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.
    ``plan_type`` (no NaN, 3 categories: basic/enterprise/premium) exercises the
    full fit→apply round-trip on production-like data with a multi-class column.
    """

    def test_plan_type_ordinal_encoding_produces_valid_codes(self) -> None:
        """OrdinalEncoder on ``plan_type`` assigns codes 0.0/1.0/2.0 in alphabetical order.

        Verifies the three-category case on real data: every row gets a code
        and same plan_type values share the same ordinal code.
        """
        df = load_sample_dataset("customers")
        params, result = _fit_apply(df, {"columns": ["plan_type"]})

        assert params["categories_count"] == [3]  # basic, enterprise, premium
        assert result["plan_type"].between(0.0, 2.0).all()
        for val in df["plan_type"].unique():
            mask = df["plan_type"] == val
            assert result.loc[mask, "plan_type"].nunique() == 1
